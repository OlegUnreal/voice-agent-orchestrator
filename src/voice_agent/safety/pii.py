from __future__ import annotations

import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


@dataclass
class Redaction:
    text: str
    found: list[str]


def redact_pii(text: str) -> Redaction:
    found: list[str] = []

    def _sub(pattern: re.Pattern[str], token: str, src: str) -> str:
        def repl(match: re.Match[str]) -> str:
            found.append(match.group(0))
            return token

        return pattern.sub(repl, src)

    out = _sub(_EMAIL, "[EMAIL]", text)
    out = _sub(_PHONE, "[PHONE]", out)
    out = _sub(_CARD, "[CARD]", out)
    return Redaction(text=out, found=found)
