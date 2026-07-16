from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.core.config import get_settings


RiskLevel = Literal["safe", "approval_required", "blocked"]


class ToolCallRequest(BaseModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any]
    step_number: int = Field(ge=1)
    approved: bool = False

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tool_name must not be blank")
        return value


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class ToolPolicy(BaseModel):
    risk_level: RiskLevel


class ToolGuardrailResult(BaseModel):
    passed: bool
    reason: str
    failure_type: str | None = None
    requires_approval: bool = False


class ToolExecutionState(BaseModel):
    signature_counts: dict[str, int] = Field(default_factory=dict)
    executed_steps: int = 0


@dataclass(frozen=True)
class ToolDefinition:
    input_model: type[BaseModel]
    policy: ToolPolicy
    handler: Callable[[BaseModel], Any]


class ToolGuardrailService:
    def __init__(
        self,
        tool_definitions: dict[str, ToolDefinition],
        allowed_tool_names: set[str] | None = None,
        max_steps: int | None = None,
        max_repeat_calls: int | None = None,
    ) -> None:
        settings = (
            get_settings()
            if max_steps is None or max_repeat_calls is None
            else None
        )
        self.tool_definitions = tool_definitions
        self.allowed_tool_names = (
            set(tool_definitions)
            if allowed_tool_names is None
            else set(allowed_tool_names)
        )
        self.max_steps = (
            settings.agent_max_steps
            if max_steps is None and settings is not None
            else max_steps
        )
        self.max_repeat_calls = (
            settings.agent_max_repeat_calls
            if max_repeat_calls is None and settings is not None
            else max_repeat_calls
        )
        assert self.max_steps is not None
        assert self.max_repeat_calls is not None
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.max_repeat_calls < 1:
            raise ValueError("max_repeat_calls must be at least 1")

    def validate(
        self,
        request: ToolCallRequest,
        state: ToolExecutionState,
    ) -> tuple[ToolGuardrailResult, BaseModel | None, str | None]:
        definition = self.tool_definitions.get(request.tool_name)
        if definition is None or request.tool_name not in self.allowed_tool_names:
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Tool is not allowed",
                    failure_type="tool_not_allowed",
                ),
                None,
                None,
            )

        try:
            validated_arguments = definition.input_model.model_validate(
                request.arguments
            )
        except ValidationError:
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Tool arguments are invalid",
                    failure_type="invalid_tool_input",
                ),
                None,
                None,
            )

        if definition.policy.risk_level == "blocked":
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Tool is blocked by policy",
                    failure_type="tool_blocked",
                ),
                None,
                None,
            )

        if (
            definition.policy.risk_level == "approval_required"
            and not request.approved
        ):
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Tool requires approval before execution",
                    failure_type="approval_required",
                    requires_approval=True,
                ),
                None,
                None,
            )

        if request.step_number > self.max_steps or state.executed_steps >= self.max_steps:
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Maximum tool steps exceeded",
                    failure_type="max_steps_exceeded",
                ),
                None,
                None,
            )

        signature = self._signature(request.tool_name, validated_arguments)
        if state.signature_counts.get(signature, 0) >= self.max_repeat_calls:
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Repeated identical tool call blocked",
                    failure_type="repeated_tool_call",
                ),
                None,
                signature,
            )

        return (
            ToolGuardrailResult(passed=True, reason="Tool call allowed"),
            validated_arguments,
            signature,
        )

    @staticmethod
    def _signature(tool_name: str, arguments: BaseModel) -> str:
        normalized = json.dumps(
            arguments.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"{tool_name}:{normalized}"


class ToolExecutor:
    def __init__(self, guardrail_service: ToolGuardrailService) -> None:
        self.guardrail_service = guardrail_service

    def execute(
        self,
        request: ToolCallRequest,
        state: ToolExecutionState,
    ) -> tuple[ToolGuardrailResult, Any | None]:
        decision, validated_arguments, signature = self.guardrail_service.validate(
            request,
            state,
        )
        if not decision.passed:
            return decision, None

        assert validated_arguments is not None
        assert signature is not None

        state.executed_steps += 1
        state.signature_counts[signature] = state.signature_counts.get(signature, 0) + 1

        definition = self.guardrail_service.tool_definitions[request.tool_name]
        try:
            return decision, definition.handler(validated_arguments)
        except Exception:
            return (
                ToolGuardrailResult(
                    passed=False,
                    reason="Tool execution failed",
                    failure_type="tool_execution_error",
                ),
                None,
            )


def build_document_tool_registry(rag_service: Any) -> dict[str, ToolDefinition]:
    def search_company_documents(arguments: BaseModel) -> dict[str, Any]:
        parsed = DocumentSearchInput.model_validate(arguments.model_dump())
        return rag_service.retrieve(parsed.query, k=parsed.top_k)

    return {
        "search_company_documents": ToolDefinition(
            input_model=DocumentSearchInput,
            policy=ToolPolicy(risk_level="safe"),
            handler=search_company_documents,
        )
    }


def build_default_tool_executor(rag_service: Any) -> ToolExecutor:
    definitions = build_document_tool_registry(rag_service)
    return ToolExecutor(ToolGuardrailService(definitions))
