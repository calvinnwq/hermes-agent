"""Standalone human and JSON account/session usage reporting."""

from __future__ import annotations

import json
import math
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot

SCHEMA_VERSION = 1
V1_PROVIDER_METRICS = frozenset({
    "credit_balance",
    "api_key_usage_total",
    "api_key_usage_daily",
    "api_key_usage_weekly",
    "api_key_usage_monthly",
})
V1_PLAN_NAMES = {
    "free": "Free",
    "plus": "Plus",
    "pro": "Pro",
    "super": "Super",
    "ultra": "Ultra",
    "team": "Team",
    "business": "Business",
    "enterprise": "Enterprise",
    "max": "Max",
    "payg": "payg",
    "usage based": "Usage Based",
    "usage_based": "Usage Based",
}


def fetch_account_usage(*args, **kwargs) -> AccountUsageSnapshot | None:
    """Load account providers only when the usage command executes."""
    from agent.account_usage import fetch_account_usage as _fetch_account_usage

    return _fetch_account_usage(*args, **kwargs)


def get_nous_portal_account_info(*args, **kwargs):
    """Load Nous account helpers only when the usage command executes."""
    from hermes_cli.nous_account import (
        get_nous_portal_account_info as _get_nous_portal_account_info,
    )

    return _get_nous_portal_account_info(*args, **kwargs)


def build_usage_parser(subparsers, *, cmd_usage_handler=None):
    """Register the standalone ``hermes usage`` command."""
    parser = subparsers.add_parser(
        "usage",
        help="Show local session usage and live account allowances",
        description=(
            "Show provider and Nous Portal usage using live network account checks. "
            "Without --session or --latest-session, this command does not select a "
            "persisted session. Use --json for schema-versioned JSON."
        ),
    )
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--session",
        metavar="ID",
        help="Include a persisted session by exact ID or unique prefix",
    )
    selectors.add_argument(
        "--latest-session",
        action="store_true",
        help="Include the latest visible, non-cron session",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the stable schema-versioned JSON envelope",
    )
    parser.set_defaults(func=cmd_usage_handler or cmd_usage)
    return parser


def _rfc3339_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _empty_session(status: str = "not_requested") -> dict[str, Any]:
    return {
        "status": status,
        "kind": None,
        "id": None,
        "source": None,
        "model": None,
        "provider": None,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "message_count": None,
        "api_calls": None,
        "tokens": {
            "input": None,
            "output": None,
            "reasoning": None,
            "prompt": None,
            "completion": None,
            "total": None,
        },
        "context": {
            "status": "not_applicable",
            "used_tokens": None,
            "limit_tokens": None,
            "used_percent": None,
            "compression_count": None,
            "breakdown_status": "not_applicable",
            "breakdown": [],
        },
    }


def _empty_provider_account(
    status: str = "not_configured", provider: str | None = None
) -> dict[str, Any]:
    return {
        "status": status,
        "provider": provider,
        "plan": None,
        "fetched_at": None,
        "windows": [],
        "metrics": [],
    }


def _empty_nous_account(status: str = "not_connected") -> dict[str, Any]:
    return {
        "status": status,
        "access": None,
        "plan": None,
        "fetched_at": None,
        "subscription": {
            "remaining_usd": None,
            "allowance_usd": None,
            "used_percent": None,
            "renews_at": None,
        },
        "topup": {"remaining_usd": None},
        "total_spendable_usd": None,
    }


def _resolve_configured_provider() -> str | None:
    from hermes_cli.auth import AuthError, resolve_provider
    from hermes_cli.config import load_config

    configured = ((load_config().get("model") or {}).get("provider") or "").strip()
    try:
        return resolve_provider(configured or None)
    except AuthError as exc:
        if exc.code == "no_provider_configured":
            return None
        raise


def _warning(code: str, source: str, message: str) -> dict[str, str]:
    return {"code": code, "source": source, "message": message}


def _normalized_plan(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return V1_PLAN_NAMES.get(value.strip().lower())


def _collect_nous_account(warnings: list[dict[str, str]]) -> dict[str, Any]:
    try:
        account = get_nous_portal_account_info(force_fresh=True)
    except TimeoutError:
        warnings.append(
            _warning("nous_timeout", "nous", "Nous Portal usage timed out.")
        )
        return _empty_nous_account("unavailable")
    except Exception:
        warnings.append(
            _warning("nous_unavailable", "nous", "Nous Portal usage is unavailable.")
        )
        return _empty_nous_account("unavailable")
    if getattr(account, "error_code", None) == "timeout":
        warnings.append(
            _warning("nous_timeout", "nous", "Nous Portal usage timed out.")
        )
        return _empty_nous_account("unavailable")
    if not account.logged_in:
        if (
            account.source == "inference_key"
            and account.error == "portal_oauth_missing"
        ):
            return _empty_nous_account("not_connected")
        if account.error:
            warnings.append(
                _warning(
                    "nous_unavailable", "nous", "Nous Portal usage is unavailable."
                )
            )
            return _empty_nous_account("unavailable")
        return _empty_nous_account("not_connected")
    if account.error:
        warnings.append(
            _warning("nous_unavailable", "nous", "Nous Portal usage is unavailable.")
        )
        return _empty_nous_account("unavailable")

    from agent.billing_usage import usage_model_from_account

    model = usage_model_from_account(account)
    if not model.available:
        warnings.append(
            _warning("nous_unavailable", "nous", "Nous Portal usage is unavailable.")
        )
        return _empty_nous_account("unavailable")

    result = _empty_nous_account("partial" if model.access == "unknown" else "ok")
    result["fetched_at"] = _rfc3339_utc_now()
    result["access"] = model.access
    result["plan"] = _normalized_plan(model.plan_name)
    result["subscription"]["remaining_usd"] = model.subscription_remaining_usd
    result["subscription"]["allowance_usd"] = model.subscription_allowance_usd
    if (
        model.plan_bar is not None
        and model.subscription_remaining_usd is not None
        and model.subscription_allowance_usd is not None
        and model.subscription_remaining_usd <= model.subscription_allowance_usd
        and model.plan_bar.total_usd > 0
    ):
        used = model.plan_bar.spent_usd / model.plan_bar.total_usd * 100
        result["subscription"]["used_percent"] = min(100.0, max(0.0, used))
    result["subscription"]["renews_at"] = _rfc3339_timestamp(model.renews_at)
    result["topup"]["remaining_usd"] = model.topup_remaining_usd
    result["total_spendable_usd"] = model.total_spendable_usd
    return result


def _finite_number(value: Any) -> float | None:
    if (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return (
            datetime
            .fromtimestamp(float(value), timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def _rfc3339_timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _persisted_session(row: dict[str, Any]) -> dict[str, Any]:
    result = _empty_session("ok")
    started_at = _timestamp(row.get("started_at"))
    ended_at = _timestamp(row.get("ended_at"))
    started_number = _finite_number(row.get("started_at"))
    ended_number = _finite_number(row.get("ended_at"))
    duration = (
        max(0.0, ended_number - started_number)
        if started_number is not None and ended_number is not None
        else None
    )
    result.update({
        "kind": "persisted",
        "id": _text_or_none(row.get("id")),
        "source": _text_or_none(row.get("source")),
        "model": _text_or_none(row.get("model")),
        "provider": _text_or_none(row.get("billing_provider")),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration,
        "message_count": _nonnegative_int(row.get("message_count")),
        "api_calls": _nonnegative_int(row.get("api_call_count")),
    })
    result["tokens"].update({
        "input": _nonnegative_int(row.get("input_tokens")),
        "output": _nonnegative_int(row.get("output_tokens")),
        "reasoning": _nonnegative_int(row.get("reasoning_tokens")),
    })
    return result


def _select_session(
    session_id: str | None, latest_session: bool
) -> tuple[dict[str, Any], int]:
    if session_id is None and not latest_session:
        return _empty_session(), 0
    if session_id == "":
        return _empty_session("not_found"), 2

    from hermes_state import SessionDB

    db = SessionDB()
    try:
        row = None
        if latest_session:
            rows = db.list_sessions_rich(
                exclude_sources=["cron", "tool"],
                limit=1,
                min_message_count=1,
                order_by_last_active=True,
                compact_rows=True,
            )
            projected = rows[0] if rows else None
            if projected is None:
                return _empty_session("not_found"), 2
            row = db.get_session(projected["id"])
            if row is None:
                return _empty_session("not_found"), 2
        else:
            resolved_id = db.resolve_session_id(session_id or "")
            if resolved_id is None:
                matches = db.find_session_ids_by_prefix(session_id or "", limit=2)
                if len(matches) > 1:
                    return _empty_session("ambiguous"), 2
                if not matches:
                    return _empty_session("not_found"), 2
                resolved_id = matches[0]
            row = db.get_session(resolved_id)
            if row is None:
                return _empty_session("not_found"), 2
        route = db.get_latest_main_model_usage(row["id"])
        if route:
            row = dict(row)
            row["model"] = route.get("model") or row.get("model")
            row["billing_provider"] = route.get("billing_provider") or row.get(
                "billing_provider"
            )
        return _persisted_session(row), 0
    finally:
        db.close()


def _collect_provider_account(
    provider: str | None, warnings: list[dict[str, str]]
) -> dict[str, Any]:
    if not provider:
        return _empty_provider_account("not_configured")
    if provider not in {"openai-codex", "anthropic", "openrouter"}:
        return _empty_provider_account("unsupported", provider)
    snapshot = fetch_account_usage(provider, report_failures=True)
    if snapshot is None:
        return _empty_provider_account("unauthenticated", provider)
    if snapshot.unavailable_reason:
        warnings.append(
            _warning(
                "provider_unavailable",
                "provider",
                f"{provider} account usage is unavailable.",
            )
        )
        result = _empty_provider_account("unavailable", provider)
        result["fetched_at"] = _timestamp(snapshot.fetched_at.timestamp())
        return result

    result = _empty_provider_account("ok", provider)
    result["plan"] = _normalized_plan(snapshot.plan)
    result["fetched_at"] = _timestamp(snapshot.fetched_at.timestamp())
    for window in snapshot.windows:
        used = window.used_percent
        if used is not None and math.isfinite(float(used)):
            used = min(100.0, max(0.0, float(used)))
        else:
            used = None
        result["windows"].append({
            "id": window.id,
            "label": window.label,
            "used_percent": used,
            "remaining_percent": None if used is None else 100.0 - used,
            "resets_at": (
                _timestamp(window.reset_at.timestamp()) if window.reset_at else None
            ),
        })
    for metric in snapshot.metrics:
        if metric.name in V1_PROVIDER_METRICS and math.isfinite(float(metric.value)):
            result["metrics"].append({
                "name": metric.name,
                "value": float(metric.value),
                "unit": "usd",
            })
    return result


def collect_usage(
    *, session_id: str | None = None, latest_session: bool = False
) -> tuple[dict[str, Any], int]:
    warnings: list[dict[str, str]] = []
    try:
        session, exit_code = _select_session(session_id, latest_session)
    except Exception:
        session = _empty_session("unavailable")
        exit_code = 1
        warnings.append(
            _warning("session_unavailable", "session", "Session usage is unavailable.")
        )
    if exit_code == 2:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": _rfc3339_utc_now(),
            "session": session,
            "accounts": {
                "provider": _empty_provider_account("not_requested"),
                "nous": _empty_nous_account("not_requested"),
            },
            "warnings": [],
        }, exit_code

    if session["status"] == "ok":
        provider = session["provider"]
    elif session["status"] == "not_requested":
        provider = _resolve_configured_provider()
    else:
        provider = None
    accounts = {
        "provider": _collect_provider_account(provider, warnings),
        "nous": _collect_nous_account(warnings),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _rfc3339_utc_now(),
        "session": session,
        "accounts": accounts,
        "warnings": warnings,
    }
    return report, exit_code


def render_human(report: dict[str, Any]) -> str:
    """Render the canonical report without exposing raw provider payloads."""
    session = report["session"]
    provider = report["accounts"]["provider"]
    nous = report["accounts"]["nous"]
    lines = []
    if session["status"] == "ok":
        lines.append("Session usage")
        lines.extend([
            f"  Session: {session['id']} ({session['source'] or 'unknown'})",
            f"  Route: {session['provider'] or 'unknown'} / {session['model'] or 'unknown'}",
            f"  API calls: {_display(session['api_calls'])}",
            (
                "  Tokens: "
                f"{_display(session['tokens']['input'])} input, "
                f"{_display(session['tokens']['output'])} output, "
                f"{_display(session['tokens']['reasoning'])} reasoning"
            ),
            "  Context: not applicable for persisted standalone sessions",
        ])
    elif session["status"] != "not_requested":
        lines.append("Session usage")
        lines.append(f"  {session['status'].replace('_', ' ')}")

    if lines:
        lines.append("")
    lines.append("Provider account")
    if provider["status"] == "ok":
        heading = provider["provider"]
        if provider["plan"]:
            heading += f" ({provider['plan']})"
        lines.append(f"  {heading}")
        for window in provider["windows"]:
            remaining = window["remaining_percent"]
            summary = window["label"]
            if remaining is not None:
                summary += f": {remaining:g}% remaining"
            if window["resets_at"]:
                summary += f", resets {window['resets_at']}"
            lines.append(f"  {summary}")
        for metric in provider["metrics"]:
            lines.append(f"  {metric['name']}: {metric['value']:g} {metric['unit']}")
    elif provider["status"] == "not_configured":
        lines.append("  No runtime provider configured. Run hermes model.")
    elif provider["status"] == "unauthenticated":
        command = (
            f"hermes auth add {provider['provider']}"
            if provider["provider"]
            else "hermes model"
        )
        lines.append(f"  Provider is not authenticated. Run {command}.")
    else:
        lines.append(f"  {provider['status'].replace('_', ' ')}")

    lines.extend(["", "Nous Portal"])
    if nous["status"] in {"ok", "partial"}:
        lines.append(f"  Access: {nous['access'] or 'partial'}")
        if nous["plan"]:
            lines.append(f"  Plan: {nous['plan']}")
        if nous["total_spendable_usd"] is not None:
            lines.append(f"  Spendable: ${nous['total_spendable_usd']:.2f}")
    elif nous["status"] == "not_connected":
        lines.append("  Not connected. Run hermes auth add nous.")
    else:
        lines.append(f"  {nous['status'].replace('_', ' ')}")

    if report["warnings"]:
        lines.extend(["", "Warnings"])
        lines.extend(f"  {warning['message']}" for warning in report["warnings"])
    return "\n".join(lines)


def _display(value: Any) -> str:
    return "unknown" if value is None else f"{value:,}"


def _usage_unavailable_report() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _rfc3339_utc_now(),
        "session": _empty_session("unavailable"),
        "accounts": {
            "nous": _empty_nous_account("unavailable"),
            "provider": _empty_provider_account("unavailable"),
        },
        "warnings": [
            _warning(
                "usage_unavailable", "usage", "Usage reporting is unavailable."
            )
        ],
    }


def _run_usage_command(args) -> tuple[dict[str, Any], int]:
    try:
        return collect_usage(
            session_id=getattr(args, "session", None),
            latest_session=bool(getattr(args, "latest_session", False)),
        )
    except Exception:
        return _usage_unavailable_report(), 1


def cmd_usage(args) -> None:
    rendered = ""
    if args.json:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            report, exit_code = _run_usage_command(args)
            try:
                rendered = json.dumps(report, allow_nan=False)
            except Exception:
                report = _usage_unavailable_report()
                exit_code = 1
                rendered = json.dumps(report, allow_nan=False)
    else:
        report, exit_code = _run_usage_command(args)
    if args.json:
        print(rendered)
    else:
        print(render_human(report))
        if exit_code == 1:
            print("Usage reporting encountered a local error.", file=sys.stderr)
    raise SystemExit(exit_code)
