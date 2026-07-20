"""Tests for gateway /yolo session scoping."""

import os
from unittest.mock import Mock

import pytest

import gateway.run as gateway_run
import gateway.slash_commands as slash_commands_module
import tools.approval as approval_module
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import SessionSource
from tools.approval import disable_session_yolo, enable_session_yolo, is_session_yolo_enabled


@pytest.fixture(autouse=True)
def _clean_yolo_state(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")
    yield
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    disable_session_yolo("agent:main:telegram:dm:chat-a")
    disable_session_yolo("agent:main:telegram:dm:chat-b")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event(chat_id: str, text: str = "/yolo") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id=f"user-{chat_id}",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text=text, source=source)


@pytest.mark.asyncio
async def test_yolo_command_toggles_only_current_session(monkeypatch):
    runner = _make_runner()
    monkeypatch.setattr(slash_commands_module, "t", lambda key: f"translated:{key}")

    event_a = _make_event("chat-a")
    session_a = runner._session_key_for_source(event_a.source)
    session_b = runner._session_key_for_source(_make_event("chat-b").source)

    result_on = await runner._handle_yolo_command(event_a)

    assert isinstance(result_on, EphemeralReply)
    assert result_on == "translated:gateway.yolo.enabled"
    assert is_session_yolo_enabled(session_a) is True
    assert is_session_yolo_enabled(session_b) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None

    result_off = await runner._handle_yolo_command(event_a)

    assert isinstance(result_off, EphemeralReply)
    assert result_off == "translated:gateway.yolo.disabled"
    assert is_session_yolo_enabled(session_a) is False
    assert os.environ.get("HERMES_YOLO_MODE") is None


@pytest.mark.parametrize(
    ("enabled", "expected_key"),
    [
        (True, "gateway.yolo.status_enabled"),
        (False, "gateway.yolo.status_disabled"),
    ],
)
@pytest.mark.asyncio
async def test_yolo_status_reports_without_mutating_session(
    monkeypatch, enabled, expected_key
):
    runner = _make_runner()
    event = _make_event("chat-a", "/yolo StAtUs")
    session_key = runner._session_key_for_source(event.source)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(slash_commands_module, "t", lambda key: f"translated:{key}")
    if enabled:
        enable_session_yolo(session_key)

    result = await runner._handle_yolo_command(event)

    assert isinstance(result, EphemeralReply)
    assert result == f"translated:{expected_key}"
    assert is_session_yolo_enabled(session_key) is enabled


@pytest.mark.asyncio
async def test_yolo_unknown_argument_prints_usage_without_resolving_session(monkeypatch):
    runner = _make_runner()
    event = _make_event("chat-a", "/yolo nope")
    session_key = "agent:main:telegram:dm:chat-a"
    enable_session_yolo(session_key)
    resolve_session = Mock(side_effect=AssertionError("unknown arguments must not resolve session state"))
    monkeypatch.setattr(runner, "_session_key_for_source", resolve_session)

    result = await runner._handle_yolo_command(event)

    assert isinstance(result, EphemeralReply)
    assert result == "/yolo [status]"
    assert resolve_session.call_count == 0
    assert is_session_yolo_enabled(session_key) is True
