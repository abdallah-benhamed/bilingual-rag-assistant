import pytest

from assistant.retrieval import detect_language, load_indexes, normalize_arabic, search, tokenize


def test_every_document_exists_in_both_languages():
    indexes = load_indexes()
    docs = {lang: {c.doc for c in index.chunks} for lang, index in indexes.items()}
    assert docs["en"] == docs["ar"]
    assert len(indexes["en"].chunks) == len(indexes["ar"].chunks) == 16


@pytest.mark.parametrize(
    "text, lang",
    [("How much is delivery?", "en"), ("كم سعر التوصيل؟", "ar"), ("سعر delivery إلى جربة", "ar")],
)
def test_detect_language(text, lang):
    assert detect_language(text) == lang


def test_arabic_normalisation_removes_diacritics_and_unifies_letters():
    assert normalize_arabic("إِلْغَاءُ الطَّلَبِ") == "الغاء الطلب"
    assert normalize_arabic("مَدِينَة") == "مدينه"


def test_tokenize_drops_stop_words_and_articles():
    assert tokenize("What is the delivery fee?") == ["delivery", "fee"]
    assert tokenize("كم سعر التوصيل؟") == ["سعر", "توصيل"]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("How much does delivery cost?", "en/delivery-zones-and-fees#delivery-fees"),
        ("Can I change my address after the courier picked it up?",
         "en/orders-and-tracking#changing-the-delivery-address"),
        ("My order was 90 minutes late, do I get anything back?",
         "en/refunds-and-cancellations#late-deliveries"),
        ("Do you deliver to Sfax?", "en/delivery-zones-and-fees#zones-we-cover"),
        ("كم سعر التوصيل إلى جربة؟", "ar/delivery-zones-and-fees#سعر-التوصيل"),
        ("هل يمكنني إلغاء الطلب بعد الالتقاط؟", "ar/refunds-and-cancellations#الغاء-الطلب"),
        ("وصلني منتج تالف ماذا أفعل؟", "ar/refunds-and-cancellations#المنتجات-التالفه-او-الناقصه"),
    ],
)
def test_search_finds_the_right_section(question, expected):
    hits = search(question)
    assert hits, question
    assert hits[0].chunk.id == expected


@pytest.mark.parametrize("question", ["What is the weather in Tunis tomorrow?", "ما هي حالة الطقس غدا؟"])
def test_off_topic_questions_find_nothing(question):
    assert search(question) == []
