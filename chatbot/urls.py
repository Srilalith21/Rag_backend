from django.urls import path
from .views import LoadVideoView, ChatView, VideoStatusView

urlpatterns = [
    path('load-video/', LoadVideoView.as_view(), name='load-video'),
    path('chat/', ChatView.as_view(), name='chat'),
    path('video-status/', VideoStatusView.as_view(), name='video-status'),
]
