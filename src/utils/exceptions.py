"""
Exceptions raised by the pipeline.

PipelineError
├── ConfigurationError
├── DataError: DataLoadError, DataValidationError, DataSchemaError
├── LLMError:  LLMAuthError, LLMRateLimitError, LLMTimeoutError, LLMResponseError
└── TaxonomyParseError
"""


class PipelineError(Exception):
    """Base exception; carries an optional context dict that is shown in the message."""

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} | context: {ctx}"
        return base


class ConfigurationError(PipelineError):
    """config.yaml is malformed or a required env var is missing."""


class DataError(PipelineError):
    pass


class DataLoadError(DataError):
    """A data file cannot be read."""


class DataValidationError(DataError):
    """Data fails a quality check."""


class DataSchemaError(DataError):
    """A required column is missing from the dataset."""

    def __init__(self, missing_columns: list[str], source: str = "") -> None:
        self.missing_columns = missing_columns
        context = {"missing": missing_columns, "source": source}
        super().__init__(f"Required columns not found: {missing_columns}", context=context)


class LLMError(PipelineError):
    pass


class LLMAuthError(LLMError):
    """The API key is invalid or missing."""


class LLMRateLimitError(LLMError):
    """Rate limit or quota exceeded (HTTP 429)."""


class LLMTimeoutError(LLMError):
    def __init__(self, timeout_seconds: float, batch_id: int | None = None) -> None:
        context = {"timeout_s": timeout_seconds, "batch_id": batch_id}
        super().__init__(f"LLM request timed out after {timeout_seconds}s", context=context)


class LLMResponseError(LLMError):
    """The LLM returned something that could not be parsed as JSON."""

    def __init__(self, raw_response: str, batch_id: int | None = None) -> None:
        self.raw_response = raw_response
        preview = raw_response[:200] if raw_response else "<empty>"
        context = {"batch_id": batch_id, "response_preview": preview}
        super().__init__("Failed to parse LLM response as JSON", context=context)


class TaxonomyParseError(PipelineError):
    """A taxonomy JSON cannot be parsed or has an invalid structure."""
