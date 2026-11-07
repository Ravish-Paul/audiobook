"""Usage endpoint: the authenticated tester's own generation allowance."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.schemas.usage import UsageResponse
from app.security import current_tester, generation_usage

router = APIRouter(tags=["Usage"])


@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Your generation allowance",
    response_description=(
        "The current tester's own limit/used/remaining + script character cap"
    ),
)
def get_usage(tester: str = Depends(current_tester)) -> UsageResponse:
    """Return the usage of the AUTHENTICATED tester only.

    The identity comes from the session/token — never from the request —
    and no other tester's identity, limit or usage is exposed. There is no
    endpoint to edit allowances: limits are configured in the environment.
    """
    return UsageResponse(**generation_usage(get_settings(), tester))
