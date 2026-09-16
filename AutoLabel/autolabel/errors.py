"""Shared AutoLabel domain errors."""

from __future__ import annotations

from typing import Any, Optional


class IntentError(ValueError):
    """Intent / label parsing failed; do not invent labels locally."""


class NameSuggestError(RuntimeError):
    """Vision name-suggest failed; do not invent English SAM prompts."""


class ImageModerationBlocked(RuntimeError):
    """Vision backend rejected one image via safety moderation; skip and continue."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: Any = None,
        code: str = "moderation_blocked",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.code = code
