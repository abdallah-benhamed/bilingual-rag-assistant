"""Settings for the test suite: no secrets needed, no Gemini key, in-memory database."""
import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-key")
os.environ.pop("GEMINI_API_KEY", None)

from .settings import *  # noqa: E402,F401,F403

DEBUG = False
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
MIDDLEWARE = [m for m in MIDDLEWARE if "whitenoise" not in m]
SECURE_SSL_REDIRECT = False
