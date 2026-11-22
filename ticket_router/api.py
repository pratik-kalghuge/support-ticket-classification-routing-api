"""FastAPI application. Train the model before starting the service."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ticket_router.data import combine_text
from ticket_router.routing import TicketRouter


class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    subject: str = Field(max_length=500)
    description: str = Field(max_length=10000)

    @model_validator(mode="after")
    def require_text(self):
        if not combine_text(self.subject, self.description):
            raise ValueError("subject and description cannot both be blank")
        return self


class Prediction(BaseModel):
    predicted_category: str
    confidence: float = Field(ge=0, le=1)
    routing_decision: Literal["auto_route", "manual_review"]
    assigned_team: str | None


def create_app(model_path: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_path or Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
        if not path.is_file():
            raise RuntimeError(f"Model missing at {path}. Run: python -m ticket_router.train")
        app.state.router = TicketRouter.load(path)
        yield
        del app.state.router

    application = FastAPI(
        title="Support Ticket Classification & Routing API", version="1.0.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health(request: Request):
        router = request.app.state.router
        return {"status": "ok", "model_loaded": True, "threshold": router.threshold}

    @application.post("/predict", response_model=Prediction)
    def predict(ticket: TicketRequest, request: Request):
        return request.app.state.router.predict(ticket.subject, ticket.description)

    return application


app = create_app()
