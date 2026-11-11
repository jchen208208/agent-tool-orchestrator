from __future__ import annotations

import os
from pathlib import Path

from backend.googlecalendar_mock.state import BASE_NOW


def _load_dotenv() -> None:
    # load .env
    candidate_dir = Path(__file__).resolve().parents[2]
    for path in (candidate_dir / ".env", candidate_dir / "backend" / ".env", candidate_dir.parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# models
AGENT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", AGENT_MODEL)
FALLBACK_MODEL = "gpt-4.1-mini"  # used if this key can't access the models above (404/403)

# OpenAI client
LLM_TIMEOUT_S = 30.0
LLM_MAX_RETRIES = 2

# router
MAX_ROUTED_TOOLS = 12

# loop limits
MAX_ROUNDS = 8  # model calls per /chat request
TOOL_RESULT_MAX_CHARS = 20_000  # what the model sees, the log always keeps the full string

# mock time
NOW = BASE_NOW
NOW_TEXT = NOW.strftime("%A, %B %d, %Y %H:%M") + f" ({NOW.tzinfo.key})"