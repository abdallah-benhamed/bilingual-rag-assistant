from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/handoff/", views.api_handoff, name="api_handoff"),
    path("kb/", views.kb_index, name="kb_index"),
    path("kb/<str:lang>/<slug:doc>/", views.kb_doc, name="kb_doc"),
    path("health/", views.health, name="health"),
]
