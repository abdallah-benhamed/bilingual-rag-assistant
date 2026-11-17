import uuid

from django.db import models


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session_key = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Conversation {str(self.id)[:8]} ({self.created_at:%Y-%m-%d %H:%M})"


class Message(models.Model):
    class Role(models.TextChoices):
        USER = "user", "Customer"
        ASSISTANT = "assistant", "Assistant"

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=Role.choices)
    text = models.TextField()
    lang = models.CharField(max_length=2, blank=True)
    mode = models.CharField(max_length=12, blank=True, help_text="generated, extractive or handoff")
    retrieval = models.CharField(max_length=10, blank=True, help_text="keyword, hybrid or none")
    handoff = models.BooleanField(default=False)
    reason = models.CharField(max_length=40, blank=True)
    sources = models.JSONField(default=list, blank=True)
    scores = models.JSONField(default=list, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.get_role_display()}: {self.text[:60]}"


class HandoffRequest(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=100)
    contact = models.CharField(max_length=150, help_text="Phone or email")
    question = models.TextField(blank=True)
    resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["resolved", "-created_at"]

    def __str__(self):
        return f"{self.name} ({self.contact})"
