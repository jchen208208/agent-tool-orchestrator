from __future__ import annotations

import logging

import openai

from backend.helpers.config import FALLBACK_MODEL

log = logging.getLogger(__name__)

_unavailable: set[str] = set()
_no_temperature: set[str] = set()


def create_response(client, model: str, **kwargs):
    if model in _unavailable:
        model = FALLBACK_MODEL
    if model in _no_temperature:
        kwargs.pop("temperature", None)
        kwargs.setdefault("reasoning", {"effort": "low"})
    try:
        return client.responses.create(model=model, **kwargs)
    except openai.BadRequestError as e:
        if "temperature" not in kwargs or "temperature" not in str(e):
            raise
        log.warning("model %s doesn't support temperature, dropping it from now on", model)
        _no_temperature.add(model)
        return create_response(client, model, **kwargs)
    except (openai.NotFoundError, openai.PermissionDeniedError):
        if model == FALLBACK_MODEL:
            raise
        log.warning("model %s isn't available on this key, using %s from now on", model, FALLBACK_MODEL)
        _unavailable.add(model)
        return create_response(client, FALLBACK_MODEL, **kwargs)
