"""Safe error contract: every failure is {"error": {code, message, request_id, details}}.

Nothing about the internal cause (stack traces, SQL, file paths, secrets, submitted input)
is ever returned. Validation details name the field and the rule, never the value.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.logging_config import log_event

FIELD_CODES = {
    "query": ("QUERY_INVALID", "Query must contain between 2 and 500 characters."),
    "query_language": ("FILTER_INVALID", "query_language must be one of en, es, fr."),
    "target_languages": (
        "FILTER_INVALID",
        "target_languages must be 1-3 distinct values of en, es, fr.",
    ),
    "sources": ("FILTER_INVALID", "sources must be a single supported source."),
    "mode": ("MODE_INVALID", "mode must be dense, lexical or hybrid."),
    "page_size": ("PAGE_SIZE_INVALID", "page_size must be between 5 and 20."),
    "limit": ("PAGE_SIZE_INVALID", "limit must be between 5 and 20."),
    "cursor": ("CURSOR_INVALID", "The cursor is invalid."),
}


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(f"{code}: {message}")
        self.status, self.code, self.message = status, code, message
        self.details = details or {}
        self.headers = headers or {}


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_response(
    status: int,
    code: str,
    message: str,
    request_id: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details or {},
        }
    }
    return JSONResponse(body, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(
            exc.status, exc.code, exc.message, request_id_of(request), exc.details, exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        issues, code, message = [], "REQUEST_INVALID", "The request is invalid."
        for err in exc.errors():
            loc = err["loc"][1:]  # first element is where the value came from (body/query/path)
            field = str(loc[0]) if loc and isinstance(loc[0], str) else "request"
            issues.append(
                {"field": field, "issue": err["type"]}
            )  # type only: never the input value
            if field in FIELD_CODES and code == "REQUEST_INVALID":
                code, message = FIELD_CODES[field]
        return error_response(422, code, message, request_id_of(request), {"issues": issues[:10]})

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        if getattr(request.state, "body_too_large", False):
            # FastAPI turns any body-read failure into a 400; report what really happened.
            return error_response(
                413, "REQUEST_TOO_LARGE", "The request body is too large.", request_id_of(request)
            )
        codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        code = codes.get(exc.status_code, "HTTP_ERROR")
        return error_response(
            exc.status_code, code, "Request could not be served.", request_id_of(request)
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database(request: Request, exc: SQLAlchemyError):
        """Connection loss, timeouts and pool exhaustion are expected failure modes: a retryable
        503. Handling them here (instead of letting them escape) also keeps the server from
        printing a traceback whose message contains the SQL statement and its parameters."""
        log_event(
            "database_error",
            level=40,
            request_id=request_id_of(request),
            error_code=type(exc).__name__,
        )
        return error_response(
            503,
            "DATABASE_UNAVAILABLE",
            "The service is temporarily unavailable. Please try again shortly.",
            request_id_of(request),
            headers={"Retry-After": "5"},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Only the exception class is logged; the message may contain SQL, paths or user input.
        log_event(
            "unhandled_error", request_id=request_id_of(request), error_code=type(exc).__name__
        )
        return error_response(
            500, "INTERNAL_ERROR", "Something went wrong.", request_id_of(request)
        )
