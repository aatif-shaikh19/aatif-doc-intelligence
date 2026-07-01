import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.insights import (
    InsightsConfigurationError,
    InsightsInputError,
    InsightsResponseError,
    InsightsResult,
    InsightsServiceError,
    generate_insights,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class InsightsResponse(BaseModel):
    executive_summary: str
    risks: list[str]
    opportunities: list[str]
    missing_information: list[str]
    next_actions: list[str]


def _to_response(result: InsightsResult) -> InsightsResponse:
    return InsightsResponse(
        executive_summary=result.executive_summary,
        risks=result.risks,
        opportunities=result.opportunities,
        missing_information=result.missing_information,
        next_actions=result.next_actions,
    )


@router.post("/insights", response_model=InsightsResponse)
def insights() -> InsightsResponse:
    try:
        result = generate_insights()
    except InsightsInputError as exc:
        logger.warning("Insights rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (
        InsightsConfigurationError,
        InsightsServiceError,
        InsightsResponseError,
    ) as exc:
        logger.error("Insights failed", exc_info=True)
        raise HTTPException(status_code=500, detail="insights generation failed") from exc

    return _to_response(result)