import json
import logging

from django.conf import settings
from django.core.cache import cache
from django.core.mail import send_mail
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from . import pipeline
from .models import Conversation, HandoffRequest, Message
from .retrieval import all_documents

logger = logging.getLogger(__name__)

SAMPLE_QUESTIONS = [
    "How much does delivery cost between Zarzis and Djerba?",
    "My order was 90 minutes late. Do I get anything back?",
    "Do you deliver to Sfax?",
    "كم سعر التوصيل داخل جرجيس؟",
    "وصلني منتج تالف، ماذا أفعل؟",
    "What will the weather be tomorrow?",
]


def _json_body(request) -> dict:
    try:
        return json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return {}


def _conversation(request) -> Conversation:
    if not request.session.session_key:
        request.session.save()
    conversation_id = request.session.get("conversation_id")
    conversation = Conversation.objects.filter(id=conversation_id).first() if conversation_id else None
    if conversation is None:
        conversation = Conversation.objects.create(session_key=request.session.session_key)
        request.session["conversation_id"] = str(conversation.id)
    return conversation


def _rate_limited(request) -> bool:
    key = f"chat-rate:{request.session.session_key}"
    count = cache.get(key, 0)
    if count >= settings.CHAT_MESSAGES_PER_HOUR:
        return True
    cache.set(key, count + 1, timeout=3600)
    return False


@require_GET
def index(request):
    return render(request, "assistant/chat.html", {"samples": SAMPLE_QUESTIONS})


@require_POST
def api_chat(request):
    question = str(_json_body(request).get("message", "")).strip()
    if not question:
        return JsonResponse({"error": "Empty message."}, status=400)
    conversation = _conversation(request)
    if _rate_limited(request):
        return JsonResponse({"error": "Too many messages. Please try again later."}, status=429)

    Message.objects.create(conversation=conversation, role=Message.Role.USER, text=question[:2000])
    result = pipeline.answer(question)
    Message.objects.create(
        conversation=conversation,
        role=Message.Role.ASSISTANT,
        text=result.answer,
        lang=result.lang,
        mode=result.mode,
        retrieval=result.retrieval,
        handoff=result.handoff,
        reason=result.reason,
        sources=result.sources,
        scores=result.scores,
        latency_ms=result.latency_ms,
    )
    data = result.to_dict()
    data.pop("scores")
    return JsonResponse(data)


@require_POST
def api_handoff(request):
    body = _json_body(request)
    name = str(body.get("name", "")).strip()[:100]
    contact = str(body.get("contact", "")).strip()[:150]
    if not name or not contact:
        return JsonResponse({"error": "Name and phone or email are required."}, status=400)
    conversation = _conversation(request)
    last_question = conversation.messages.filter(role=Message.Role.USER).order_by("-created_at").first()
    handoff = HandoffRequest.objects.create(
        conversation=conversation,
        name=name,
        contact=contact,
        question=last_question.text if last_question else "",
    )
    if settings.HANDOFF_EMAIL:
        try:
            send_mail(
                f"[Demo assistant] Handoff from {name}",
                f"Contact: {contact}\nQuestion: {handoff.question}\nConversation: {conversation.id}",
                None,
                [settings.HANDOFF_EMAIL],
            )
        except Exception:  # The request is saved even if email is not configured correctly.
            logger.exception("Could not send handoff email")
    return JsonResponse({"ok": True})


@require_GET
def kb_index(request):
    return render(request, "assistant/kb_index.html", {"documents": all_documents()})


@require_GET
def kb_doc(request, lang, doc):
    chunks = all_documents().get(lang, {}).get(doc)
    if not chunks:
        raise Http404("Unknown document")
    return render(
        request,
        "assistant/kb_doc.html",
        {"lang": lang, "title": chunks[0].doc_title, "chunks": chunks, "rtl": lang == "ar"},
    )


@require_GET
def health(request):
    return HttpResponse("ok", content_type="text/plain")
