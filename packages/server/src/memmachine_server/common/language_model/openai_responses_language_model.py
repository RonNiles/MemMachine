"""OpenAI-based language model implementation."""

import asyncio
import logging
from typing import Any, TypeVar, cast
from uuid import uuid4

import json_repair
import openai
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ToolParam,
)
from pydantic import BaseModel, Field, InstanceOf

from memmachine_server.common.data_types import ExternalServiceAPIError
from memmachine_server.common.metrics_factory import MetricsFactory, OperationTracker

from .language_model import LanguageModel

T = TypeVar("T")

logger = logging.getLogger(__name__)


class OpenAIResponsesLanguageModelParams(BaseModel):
    """
    Parameters for OpenAIResponsesLanguageModel.

    Attributes:
        client (openai.AsyncOpenAI):
            AsyncOpenAI client to use for making API calls.
        model (str):
            Name of the OpenAI model to use
            (e.g. 'gpt-5-nano').
        max_retry_interval_seconds (int):
            Maximal retry interval in seconds when retrying API calls
            (default: 120).
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory
            for collecting usage metrics
            (default: None).
        reasoning_effort (str | None):
            Reasoning effort level for supported models
            (e.g. "minimal", "low", "medium", "high", "none").
            If None, the API default is used.

    """

    client: InstanceOf[openai.AsyncOpenAI] = Field(
        ...,
        description="AsyncOpenAI client to use for making API calls",
    )
    model: str = Field(
        ...,
        description="Name of the OpenAI model to use (e.g. 'gpt-5-nano')",
    )
    max_retry_interval_seconds: int = Field(
        120,
        description="Maximal retry interval in seconds when retrying API calls",
        gt=0,
    )
    max_output_tokens: int | None = Field(
        None,
        description=(
            "Maximum number of output tokens per request. Caps generation so a "
            "runaway response cannot grow toward the model's full output ceiling. "
            "If None, the provider default is used."
        ),
        gt=0,
    )
    request_timeout_seconds: float | None = Field(
        None,
        description=(
            "Per-request timeout in seconds, applied via the client's "
            "with_options(timeout=...). If None, the client default is used."
        ),
        gt=0,
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )
    reasoning_effort: str | None = Field(
        None,
        description=(
            "Reasoning effort level for supported models "
            "(e.g. 'minimal', 'low', 'medium', 'high', 'none' depend on model). "
            "If None, API default is used."
        ),
    )
    temperature: float | None = Field(
        None,
        description=(
            "Sampling temperature. 0 gives (near-)deterministic output. "
            "If None, the API default is used."
        ),
    )


class OpenAIResponsesLanguageModel(LanguageModel):
    """Language model that uses OpenAI's responses API."""

    def __init__(self, params: OpenAIResponsesLanguageModelParams) -> None:
        """
        Initialize the responses language model with configuration.

        Args:
            params (OpenAIResponsesLanguageModelParams):
                Parameters for the OpenAIResponsesLanguageModel.

        """
        super().__init__()

        self._client = params.client

        self._model = params.model

        self._max_retry_interval_seconds = params.max_retry_interval_seconds
        self._reasoning_effort = params.reasoning_effort
        self._request_timeout_seconds = params.request_timeout_seconds

        # Optional request params spread into every API call; omitted entirely
        # when None so the provider default is used. max_output_tokens caps the
        # generation length so a degenerate runaway response cannot stall the
        # caller for minutes.
        self._sampling_kwargs: dict[str, Any] = {}
        if params.temperature is not None:
            self._sampling_kwargs["temperature"] = params.temperature
        if params.max_output_tokens is not None:
            self._sampling_kwargs["max_output_tokens"] = params.max_output_tokens

        metrics_factory = params.metrics_factory

        self._tracker = OperationTracker(
            metrics_factory, prefix="language_model_openai_responses"
        )

        self._should_collect_metrics = False
        if metrics_factory is not None:
            self._should_collect_metrics = True

            self._input_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_input_tokens",
                "Number of input tokens used for OpenAI language model",
            )
            self._input_cached_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_input_cached_tokens",
                (
                    "Number of tokens retrieved from cache "
                    "used for OpenAI language model"
                ),
            )
            self._output_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_output_tokens",
                "Number of output tokens used for OpenAI language model",
            )
            self._output_reasoning_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_output_reasoning_tokens",
                ("Number of reasoning tokens used for OpenAI language model"),
            )
            self._total_tokens_usage_counter = metrics_factory.get_counter(
                "language_model_openai_responses_usage_total_tokens",
                "Number of tokens used for OpenAI language model",
            )

    def _client_with_options(
        self, *, max_retries: int | None = None
    ) -> openai.AsyncOpenAI:
        """Return the client with a per-request timeout (and optional retries).

        Applying the timeout via ``with_options`` bounds a single API call so a
        pathological generation is aborted rather than blocking for the OpenAI
        client default (600s). When no timeout is configured, the client is
        returned with only the requested retry override (if any).
        """
        options: dict[str, Any] = {}
        if self._request_timeout_seconds is not None:
            options["timeout"] = self._request_timeout_seconds
        if max_retries is not None:
            options["max_retries"] = max_retries
        return self._client.with_options(**options) if options else self._client

    async def generate_parsed_response(
        self,
        output_format: type[T],
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        max_attempts: int = 1,
    ) -> T | None:
        """Generate a structured response parsed into the given model."""
        async with self._tracker("generate_parsed_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            input_prompts = cast(
                ResponseInputParam,
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )

            generate_response_call_uuid = uuid4()

            try:
                response = await self._client_with_options(
                    max_retries=max_attempts,
                ).responses.parse(
                    model=self._model,
                    input=input_prompts,
                    store=False,
                    text_format=output_format,
                    **self._sampling_kwargs,
                )
            except openai.OpenAIError as e:
                error_message = (
                    f"[call uuid: {generate_response_call_uuid}] "
                    "Giving up generating response "
                    f"due to non-retryable {type(e).__name__}"
                )
                logger.exception(error_message)
                raise ExternalServiceAPIError(error_message) from e

            self._collect_usage_metrics(response)

            return response.output_parsed

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, list[dict[str, Any]]]:
        output, function_calls_arguments, _, _ = await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_attempts=max_attempts,
        )
        return output, function_calls_arguments

    async def generate_response_with_token_usage(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        return await self._generate_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_choice=tool_choice,
            max_attempts=max_attempts,
        )

    async def _generate_response(  # noqa: C901
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        """Generate a raw text response (and optional tool call)."""
        async with self._tracker("generate_response"):
            if max_attempts <= 0:
                raise ValueError("max_attempts must be a positive integer")

            input_prompts = cast(
                ResponseInputParam,
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
            )

            generate_response_call_uuid = uuid4()

            response: Response | None = None

            sleep_seconds = 1
            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(
                        "[call uuid: %s] "
                        "Attempting to generate response using %s OpenAI language model: "
                        "on attempt %d with max attempts %d",
                        generate_response_call_uuid,
                        self._model,
                        attempt,
                        max_attempts,
                    )
                    if tools is None:
                        response = await self._client_with_options().responses.create(
                            model=self._model,
                            input=input_prompts,
                            store=False,
                            **self._sampling_kwargs,
                        )
                    else:
                        response = await self._client_with_options().responses.create(
                            model=self._model,
                            input=input_prompts,
                            store=False,
                            tools=cast(list[ToolParam], tools),
                            tool_choice=cast(
                                Any,
                                tool_choice if tool_choice is not None else "auto",
                            ),
                            **self._sampling_kwargs,
                        )
                    break
                except (
                    openai.RateLimitError,
                    openai.APITimeoutError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ) as e:
                    # Exception may be retried.
                    if attempt >= max_attempts:
                        error_message = (
                            f"[call uuid: {generate_response_call_uuid}] "
                            "Giving up generating response "
                            f"after failed attempt {attempt} "
                            f"due to retryable {type(e).__name__}: "
                            f"max attempts {max_attempts} reached"
                        )
                        logger.exception(error_message)
                        raise ExternalServiceAPIError(error_message) from e

                    logger.info(
                        "[call uuid: %s] "
                        "Retrying generating response in %d seconds "
                        "after failed attempt %d due to retryable %s...",
                        generate_response_call_uuid,
                        sleep_seconds,
                        attempt,
                        type(e).__name__,
                    )
                    await asyncio.sleep(sleep_seconds)
                    sleep_seconds *= 2
                    sleep_seconds = min(sleep_seconds, self._max_retry_interval_seconds)
                    continue
                except openai.OpenAIError as e:
                    error_message = (
                        f"[call uuid: {generate_response_call_uuid}] "
                        "Giving up generating response "
                        f"after failed attempt {attempt} "
                        f"due to non-retryable {type(e).__name__}"
                    )
                    logger.exception(error_message)
                    raise ExternalServiceAPIError(error_message) from e

            if response is None:
                raise RuntimeError("OpenAI response was not generated")

            self._collect_usage_metrics(response)

            if response.output is None:
                return (response.output_text or "", [], 0, 0)

            function_calls_arguments: list[dict[str, Any]] = []
            try:
                for output in response.output:
                    if output.type != "function_call":
                        continue
                    function_call = cast(ResponseFunctionToolCall, output)
                    function_calls_arguments.append(
                        {
                            "call_id": function_call.call_id,
                            "function": {
                                "name": function_call.name,
                                "arguments": json_repair.loads(function_call.arguments),
                            },
                        }
                    )
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Failed to repair or parse JSON from function call arguments"
                ) from e

            return (
                response.output_text or "",
                function_calls_arguments,
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
            )

    def _collect_usage_metrics(self, response: Response) -> None:
        if not self._should_collect_metrics:
            return

        if response.usage is None:
            logger.debug("No usage information found in response")
            return

        try:
            self._input_tokens_usage_counter.increment(
                value=response.usage.input_tokens,
            )
            self._input_cached_tokens_usage_counter.increment(
                value=response.usage.input_tokens_details.cached_tokens,
            )
            self._output_tokens_usage_counter.increment(
                value=response.usage.output_tokens,
            )
            self._output_reasoning_tokens_usage_counter.increment(
                value=response.usage.output_tokens_details.reasoning_tokens,
            )
            self._total_tokens_usage_counter.increment(
                value=response.usage.total_tokens,
            )

        except Exception:
            logger.exception("Failed to collect usage metrics")
