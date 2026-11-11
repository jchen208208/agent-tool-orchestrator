from __future__ import annotations

import logging

import openai

from backend.helpers.config import FALLBACK_MODEL

log = logging.getLogger(__name__)

_unavailable: set[str] = set()


def create_response(client, model: str, **kwargs):
    if model in _unavailable:
        model = FALLBACK_MODEL
    try:
        return client.responses.create(model=model, **kwargs)
    except (openai.NotFoundError, openai.PermissionDeniedError):
        if model == FALLBACK_MODEL:
            raise
        log.warning("model %s isn't available on this key, using %s from now on", model, FALLBACK_MODEL)
        _unavailable.add(model)
        return client.responses.create(model=FALLBACK_MODEL, **kwargs)
