"""
API Views
=========
POST /api/load-video/   – validate URL, fetch transcript, build FAISS index
POST /api/chat/         – answer a question about a loaded video
GET  /api/video-status/ – check whether a video index already exists
"""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .utils import (
    extract_video_id,
    fetch_transcript,
    build_faiss_index,
    answer_question,
    index_exists,
)


class LoadVideoView(APIView):
    """
    Accept a YouTube URL, extract its transcript, build and persist a FAISS index.

    Request body:
        { "url": "https://www.youtube.com/watch?v=..." }

    Response (200):
        { "video_id": "...", "message": "Video loaded successfully.", "chunk_count": N }
    """

    def post(self, request):
        url = request.data.get("url", "").strip()
        if not url:
            return Response(
                {"error": "A YouTube URL is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        video_id = extract_video_id(url)
        if not video_id:
            return Response(
                {"error": "Could not extract a video ID from the provided URL."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Skip re-indexing if we already have the FAISS index
        if index_exists(video_id):
            return Response(
                {
                    "video_id": video_id,
                    "message": "Video was already loaded. Ready to chat!",
                    "already_indexed": True,
                },
                status=status.HTTP_200_OK,
            )

        try:
            transcript = fetch_transcript(video_id)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)

        try:
            vectorstore = build_faiss_index(video_id, transcript)
            chunk_count = vectorstore.index.ntotal
        except Exception as exc:
            return Response(
                {"error": f"Failed to build index: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "video_id": video_id,
                "message": "Video loaded successfully. Ready to chat!",
                "chunk_count": chunk_count,
                "already_indexed": False,
            },
            status=status.HTTP_200_OK,
        )


class ChatView(APIView):
    """
    Answer a question about a previously loaded YouTube video.

    Request body:
        { "video_id": "...", "question": "What is this video about?" }

    Response (200):
        { "answer": "...", "source_chunks": [...] }
    """

    def post(self, request):
        video_id = request.data.get("video_id", "").strip()
        question = request.data.get("question", "").strip()

        if not video_id:
            return Response(
                {"error": "video_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not question:
            return Response(
                {"error": "question is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not index_exists(video_id):
            return Response(
                {"error": "Video not loaded yet. Please call /api/load-video/ first."},
                status=status.HTTP_404_NOT_FOUND,
            )

        try:
            result = answer_question(video_id, question)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
        except Exception as exc:
            return Response(
                {"error": f"Failed to generate answer: {exc}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(result, status=status.HTTP_200_OK)


class VideoStatusView(APIView):
    """
    Check whether a video has already been indexed.

    Query param:  ?video_id=...   OR   ?url=...

    Response:
        { "video_id": "...", "indexed": true/false }
    """

    def get(self, request):
        video_id = request.query_params.get("video_id", "").strip()
        url = request.query_params.get("url", "").strip()

        if not video_id and url:
            video_id = extract_video_id(url)

        if not video_id:
            return Response(
                {"error": "Provide video_id or url as a query parameter."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {"video_id": video_id, "indexed": index_exists(video_id)},
            status=status.HTTP_200_OK,
        )
