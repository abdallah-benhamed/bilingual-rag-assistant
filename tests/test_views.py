import json

import pytest

from assistant.models import HandoffRequest, Message

pytestmark = pytest.mark.django_db


def post(client, url, data):
    return client.post(url, json.dumps(data), content_type="application/json")


def test_chat_page_and_knowledge_base_render(client):
    assert client.get("/").status_code == 200
    assert client.get("/kb/").status_code == 200
    page = client.get("/kb/ar/delivery-zones-and-fees/")
    assert page.status_code == 200 and 'dir="rtl"' in page.content.decode()
    assert client.get("/kb/en/nope/").status_code == 404


def test_chat_api_answers_and_logs_both_messages(client):
    response = post(client, "/api/chat/", {"message": "How much does delivery cost?"})
    data = response.json()
    assert response.status_code == 200
    assert data["sources"][0]["url"] == "/kb/en/delivery-zones-and-fees/#delivery-fees"
    assert "scores" not in data
    logged = list(Message.objects.values_list("role", "mode"))
    assert logged == [("user", ""), ("assistant", "extractive")]


def test_empty_message_is_rejected(client):
    assert post(client, "/api/chat/", {"message": "  "}).status_code == 400


def test_handoff_is_saved_with_the_last_question(client):
    post(client, "/api/chat/", {"message": "What will the weather be tomorrow?"})
    response = post(client, "/api/handoff/", {"name": "Sami", "contact": "+216 20 000 000"})
    assert response.status_code == 200
    handoff = HandoffRequest.objects.get()
    assert handoff.question == "What will the weather be tomorrow?"


def test_handoff_needs_name_and_contact(client):
    assert post(client, "/api/handoff/", {"name": "Sami"}).status_code == 400


def test_rate_limit(client, settings):
    settings.CHAT_MESSAGES_PER_HOUR = 2
    codes = [post(client, "/api/chat/", {"message": "delivery fee"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
