from typing import Any

from pydantic import BaseModel, Field

from app.agents.tool_guardrails import (
    ToolCallRequest,
    ToolDefinition,
    ToolExecutionState,
    ToolExecutor,
    ToolGuardrailService,
    ToolPolicy,
)


class FakeInput(BaseModel):
    value: str = Field(min_length=1)


def build_executor(
    risk_level: str = "safe",
    *,
    allowed: bool = True,
    max_steps: int = 3,
    max_repeat_calls: int = 1,
) -> tuple[ToolExecutor, dict[str, int]]:
    calls = {"count": 0}

    def handler(arguments: BaseModel) -> dict[str, Any]:
        calls["count"] += 1
        return {"value": arguments.model_dump()["value"]}

    definitions = {
        "fake_tool": ToolDefinition(
            input_model=FakeInput,
            policy=ToolPolicy(risk_level=risk_level),
            handler=handler,
        )
    }
    service = ToolGuardrailService(
        definitions,
        allowed_tool_names={"fake_tool"} if allowed else set(),
        max_steps=max_steps,
        max_repeat_calls=max_repeat_calls,
    )
    return ToolExecutor(service), calls


def request(value: str = "one", **overrides: Any) -> ToolCallRequest:
    data = {
        "tool_name": "fake_tool",
        "arguments": {"value": value},
        "step_number": 1,
    }
    data.update(overrides)
    return ToolCallRequest(**data)


def test_allowed_safe_tool_passes_and_executes_once() -> None:
    executor, calls = build_executor()
    result, output = executor.execute(request(), ToolExecutionState())

    assert result.passed is True
    assert output == {"value": "one"}
    assert calls["count"] == 1


def test_unknown_tool_is_blocked() -> None:
    executor, calls = build_executor()
    result, output = executor.execute(
        request(tool_name="delete_employee_record"),
        ToolExecutionState(),
    )

    assert result.failure_type == "tool_not_allowed"
    assert output is None
    assert calls["count"] == 0


def test_invalid_arguments_are_blocked_without_execution() -> None:
    executor, calls = build_executor()
    result, _ = executor.execute(request(arguments={"value": ""}), ToolExecutionState())

    assert result.failure_type == "invalid_tool_input"
    assert calls["count"] == 0


def test_approval_required_tool_stops_without_approval() -> None:
    executor, calls = build_executor("approval_required")
    result, _ = executor.execute(request(), ToolExecutionState())

    assert result.failure_type == "approval_required"
    assert result.requires_approval is True
    assert calls["count"] == 0


def test_approval_required_tool_executes_with_approval() -> None:
    executor, calls = build_executor("approval_required")
    result, output = executor.execute(request(approved=True), ToolExecutionState())

    assert result.passed is True
    assert output == {"value": "one"}
    assert calls["count"] == 1


def test_blocked_risk_tool_never_executes() -> None:
    executor, calls = build_executor("blocked")
    result, _ = executor.execute(request(approved=True), ToolExecutionState())

    assert result.failure_type == "tool_blocked"
    assert calls["count"] == 0


def test_maximum_tool_steps_are_enforced() -> None:
    executor, calls = build_executor(max_steps=3)
    result, _ = executor.execute(request(step_number=4), ToolExecutionState())

    assert result.failure_type == "max_steps_exceeded"
    assert calls["count"] == 0


def test_repeated_identical_call_is_blocked() -> None:
    executor, calls = build_executor(max_repeat_calls=1)
    state = ToolExecutionState()

    first, _ = executor.execute(request(), state)
    second, _ = executor.execute(request(step_number=2), state)

    assert first.passed is True
    assert second.failure_type == "repeated_tool_call"
    assert calls["count"] == 1


def test_different_arguments_are_not_identical_repeats() -> None:
    executor, calls = build_executor(max_repeat_calls=1)
    state = ToolExecutionState()

    first, _ = executor.execute(request("one"), state)
    second, _ = executor.execute(request("two", step_number=2), state)

    assert first.passed is True
    assert second.passed is True
    assert calls["count"] == 2


def test_registered_but_not_allowlisted_tool_never_executes() -> None:
    executor, calls = build_executor(allowed=False)
    result, _ = executor.execute(request(), ToolExecutionState())

    assert result.failure_type == "tool_not_allowed"
    assert calls["count"] == 0
