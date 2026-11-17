from django.contrib import admin

from .models import Conversation, HandoffRequest, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    can_delete = False
    fields = ("created_at", "role", "text", "mode", "retrieval", "handoff", "reason", "sources", "latency_ms")
    readonly_fields = fields


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("__str__", "created_at", "message_count", "had_handoff")
    inlines = [MessageInline]

    @admin.display(description="Messages")
    def message_count(self, obj):
        return obj.messages.count()

    @admin.display(boolean=True, description="Handed off")
    def had_handoff(self, obj):
        return obj.messages.filter(handoff=True).exists()


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("created_at", "role", "short_text", "lang", "mode", "retrieval", "handoff", "reason", "latency_ms")
    list_filter = ("role", "mode", "retrieval", "handoff", "lang")
    search_fields = ("text",)
    readonly_fields = [f.name for f in Message._meta.fields]

    @admin.display(description="Text")
    def short_text(self, obj):
        return obj.text[:80]


@admin.register(HandoffRequest)
class HandoffRequestAdmin(admin.ModelAdmin):
    list_display = ("created_at", "name", "contact", "question", "resolved")
    list_filter = ("resolved",)
    list_editable = ("resolved",)
