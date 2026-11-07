"""Usage schemas: the authenticated tester's own generation allowance."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UsageResponse(BaseModel):
    """Your own generation usage — never another tester's numbers.

    `generations_remaining = max(0, generation_limit - generations_used)`.
    `generation_limit` is 0 when the tester cannot generate at all.
    """

    generation_limit: int = Field(
        ge=0, description="Effective allowance (individual override or default)."
    )
    generations_used: int = Field(
        ge=0, description="Persistent lifetime count of consumed generations."
    )
    generations_remaining: int = Field(
        ge=0, description="max(0, generation_limit - generations_used)."
    )
    max_script_characters: int = Field(
        description=(
            "Maximum script length in Unicode code points (0 = unlimited). "
            "Counted identically by the backend and the frontend."
        )
    )
