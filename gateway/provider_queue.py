"""Process-shared provider/account queue for gateway agent runs.

The queue is intentionally small and stdlib-only.  Gateway deployments often
run one Hermes process per profile, so in-memory semaphores do not protect a
shared OAuth/API account.  SQLite gives us a portable single-writer lease store
without adding Redis as an operational dependency.
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from hermes_constants import get_default_hermes_root, get_hermes_home


DEFAULT_QUEUE_RULES: tuple[dict[str, Any], ...] = (
    {
        "provider": "openai-codex",
        "model": "gpt-5.5",
        "concurrency": 1,
        "label": "GPT-5.5",
    },
    {
        "provider": "anthropic",
        "model": "claude-opus*",
        "concurrency": 1,
        "label": "Claude Opus",
    },
    {
        "provider": "google",
        "model": "gemini-*flash*",
        "concurrency": 2,
        "label": "Gemini Flash",
    },
)


@dataclass(frozen=True)
class ProviderQueueSettings:
    enabled: bool
    db_path: Path
    poll_interval_seconds: float = 2.0
    notify_interval_seconds: float = 120.0
    lease_seconds: float = 420.0
    heartbeat_seconds: float = 30.0
    stale_seconds: float = 900.0
    retention_seconds: float = 7 * 24 * 3600.0
    rate_limit_cooldown_seconds: float = 180.0
    timeout_cooldown_seconds: float = 60.0
    rules: tuple[dict[str, Any], ...] = DEFAULT_QUEUE_RULES


@dataclass(frozen=True)
class ProviderQueueLane:
    lane_key: str
    display_name: str
    provider: str
    model: str
    account: str
    concurrency: int


@dataclass(frozen=True)
class ProviderQueueState:
    job_id: str
    lane_key: str
    display_name: str
    concurrency: int
    running: int
    queued_ahead: int
    position: int
    cooldown_until: float = 0.0
    acquired: bool = False

    @property
    def wait_seconds(self) -> float:
        return max(0.0, self.cooldown_until - time.time())


class ProviderQueueCancelled(RuntimeError):
    """Raised when a queued run is cancelled before acquiring a lease."""


def load_provider_queue_settings(config: Mapping[str, Any] | None) -> ProviderQueueSettings:
    raw = (config or {}).get("provider_queue") if isinstance(config, Mapping) else None
    if not isinstance(raw, Mapping):
        raw = {}

    enabled = _truthy(raw.get("enabled"), default=False)
    root = get_default_hermes_root()
    raw_db_path = str(raw.get("db_path") or "").strip()
    db_path = _expand_path(raw_db_path) if raw_db_path else root / "runtime" / "provider_queue.sqlite3"

    rules_raw = raw.get("rules") or raw.get("lanes")
    rules: tuple[dict[str, Any], ...]
    if isinstance(rules_raw, list):
        rules = tuple(rule for rule in rules_raw if isinstance(rule, dict))
    else:
        rules = DEFAULT_QUEUE_RULES

    return ProviderQueueSettings(
        enabled=enabled,
        db_path=db_path,
        poll_interval_seconds=_positive_float(raw.get("poll_interval_seconds"), 2.0),
        notify_interval_seconds=_positive_float(raw.get("notify_interval_seconds"), 120.0),
        lease_seconds=_positive_float(raw.get("lease_seconds"), 420.0),
        heartbeat_seconds=_positive_float(raw.get("heartbeat_seconds"), 30.0),
        stale_seconds=_positive_float(raw.get("stale_seconds"), 900.0),
        retention_seconds=_positive_float(raw.get("retention_seconds"), 7 * 24 * 3600.0),
        rate_limit_cooldown_seconds=_positive_float(raw.get("rate_limit_cooldown_seconds"), 180.0),
        timeout_cooldown_seconds=_positive_float(raw.get("timeout_cooldown_seconds"), 60.0),
        rules=rules,
    )


def resolve_provider_queue_lane(
    config: Mapping[str, Any] | None,
    *,
    provider: str | None,
    model: str | None,
    account: str | None = None,
) -> ProviderQueueLane | None:
    settings = load_provider_queue_settings(config)
    if not settings.enabled:
        return None

    provider_norm = _norm(provider)
    model_norm = _norm(model)
    if not provider_norm or not model_norm:
        return None

    account_norm = _norm(account) or "default"
    for rule in settings.rules:
        if not _rule_matches(rule, "provider", provider_norm):
            continue
        if not _rule_matches(rule, "model", model_norm):
            continue
        if not _rule_matches(rule, "account", account_norm, default="*"):
            continue
        concurrency = max(1, int(rule.get("concurrency") or 1))
        label = str(rule.get("label") or rule.get("name") or model_norm)
        lane_account = account_norm if rule.get("account") in (None, "", "*") else _norm(rule.get("account"))
        lane_key = str(rule.get("lane_key") or f"{provider_norm}:{lane_account}:{model_norm}")
        return ProviderQueueLane(
            lane_key=lane_key,
            display_name=label,
            provider=provider_norm,
            model=model_norm,
            account=lane_account,
            concurrency=concurrency,
        )
    return None


def resolve_provider_account_identifier(
    provider: str | None,
    *,
    api_key: str | None = None,
    hermes_home: Path | None = None,
) -> str:
    provider_norm = _norm(provider)
    if not provider_norm:
        return "default"
    home = hermes_home or get_hermes_home()
    auth_path = home / "auth.json"
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    provider_payloads = _candidate_provider_payloads(payload, provider_norm)
    for item in provider_payloads:
        account = _extract_account_from_value(item)
        if account:
            return account

    if api_key:
        return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return "default"


class ProviderQueue:
    def __init__(self, settings: ProviderQueueSettings):
        self.settings = settings
        self.db_path = settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "ProviderQueue | None":
        settings = load_provider_queue_settings(config)
        if not settings.enabled:
            return None
        return cls(settings)

    def enqueue(
        self,
        lane: ProviderQueueLane,
        *,
        session_key: str | None,
        source: str | None,
    ) -> "ProviderQueueJob":
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._upsert_lane(conn, lane, now)
            self._expire_stale_jobs(conn, now)
            conn.execute(
                """
                INSERT INTO provider_queue_jobs (
                    job_id, lane_key, status, priority, requested_at,
                    owner_id, owner_pid, session_key, source, model,
                    provider, account, display_name
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    lane.lane_key,
                    now,
                    _owner_id(),
                    os.getpid(),
                    session_key or "",
                    source or "",
                    lane.model,
                    lane.provider,
                    lane.account,
                    lane.display_name,
                ),
            )
            self._cleanup_old_jobs(conn, now)
            conn.commit()
        return ProviderQueueJob(self, job_id, lane)

    def state_for_job(self, job_id: str) -> ProviderQueueState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT j.*, l.concurrency AS lane_concurrency, l.cooldown_until
                FROM provider_queue_jobs j
                JOIN provider_queue_lanes l ON l.lane_key = j.lane_key
                WHERE j.job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            return self._state_from_row(conn, row, acquired=row["status"] == "running")

    def try_acquire(self, job_id: str, lane: ProviderQueueLane) -> ProviderQueueState:
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._upsert_lane(conn, lane, now)
            self._expire_stale_jobs(conn, now)
            row = conn.execute(
                """
                SELECT j.*, l.concurrency AS lane_concurrency, l.cooldown_until
                FROM provider_queue_jobs j
                JOIN provider_queue_lanes l ON l.lane_key = j.lane_key
                WHERE j.job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                raise ProviderQueueCancelled(f"provider queue job {job_id} no longer exists")
            if row["status"] == "running":
                state = self._state_from_row(conn, row, acquired=True)
                conn.commit()
                return state
            if row["status"] != "queued":
                raise ProviderQueueCancelled(f"provider queue job {job_id} is {row['status']}")

            state = self._state_from_row(conn, row, acquired=False)
            cooldown_until = float(row["cooldown_until"] or 0)
            can_start = (
                state.running < state.concurrency
                and state.queued_ahead == 0
                and cooldown_until <= now
            )
            if can_start:
                lease_until = now + self.settings.lease_seconds
                conn.execute(
                    """
                    UPDATE provider_queue_jobs
                    SET status = 'running',
                        started_at = ?,
                        heartbeat_at = ?,
                        lease_until = ?,
                        owner_id = ?,
                        owner_pid = ?
                    WHERE job_id = ? AND status = 'queued'
                    """,
                    (now, now, lease_until, _owner_id(), os.getpid(), job_id),
                )
                row = conn.execute(
                    """
                    SELECT j.*, l.concurrency AS lane_concurrency, l.cooldown_until
                    FROM provider_queue_jobs j
                    JOIN provider_queue_lanes l ON l.lane_key = j.lane_key
                    WHERE j.job_id = ?
                    """,
                    (job_id,),
                ).fetchone()
                state = self._state_from_row(conn, row, acquired=True)
            conn.commit()
            return state

    def cancel_job(self, job_id: str, reason: str = "cancelled") -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE provider_queue_jobs
                SET status = 'cancelled', finished_at = ?, error = ?
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (now, reason[:1000], job_id),
            )

    def heartbeat(self, job_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE provider_queue_jobs
                SET heartbeat_at = ?, lease_until = ?
                WHERE job_id = ? AND status = 'running'
                """,
                (now, now + self.settings.lease_seconds, job_id),
            )

    def finish_job(self, job_id: str, status: str, error: str | None = None) -> None:
        final_status = "done" if status == "done" else "failed"
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT lane_key FROM provider_queue_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            conn.execute(
                """
                UPDATE provider_queue_jobs
                SET status = ?, finished_at = ?, error = ?
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (final_status, now, (error or "")[:1000], job_id),
            )
            if final_status == "failed" and row is not None:
                cooldown = self._cooldown_for_error(error or "")
                if cooldown > 0:
                    conn.execute(
                        """
                        UPDATE provider_queue_lanes
                        SET cooldown_until = MAX(COALESCE(cooldown_until, 0), ?),
                            updated_at = ?
                        WHERE lane_key = ?
                        """,
                        (now + cooldown, now, row["lane_key"]),
                    )
            conn.commit()

    def snapshot(self, *, limit: int = 10) -> dict[str, Any]:
        now = time.time()
        with self._connect() as conn:
            self._expire_stale_jobs(conn, now)
            lanes = []
            for lane in conn.execute(
                """
                SELECT lane_key, display_name, concurrency, cooldown_until
                FROM provider_queue_lanes
                ORDER BY lane_key
                """
            ).fetchall():
                running = conn.execute(
                    """
                    SELECT COUNT(*) FROM provider_queue_jobs
                    WHERE lane_key = ? AND status = 'running'
                    """,
                    (lane["lane_key"],),
                ).fetchone()[0]
                queued = conn.execute(
                    """
                    SELECT COUNT(*) FROM provider_queue_jobs
                    WHERE lane_key = ? AND status = 'queued'
                    """,
                    (lane["lane_key"],),
                ).fetchone()[0]
                oldest = conn.execute(
                    """
                    SELECT MIN(requested_at) FROM provider_queue_jobs
                    WHERE lane_key = ? AND status = 'queued'
                    """,
                    (lane["lane_key"],),
                ).fetchone()[0]
                lanes.append(
                    {
                        "lane_key": lane["lane_key"],
                        "display_name": lane["display_name"],
                        "concurrency": int(lane["concurrency"] or 1),
                        "running": int(running or 0),
                        "queued": int(queued or 0),
                        "oldest_wait_seconds": max(0, int(now - oldest)) if oldest else 0,
                        "cooldown_seconds": max(0, int(float(lane["cooldown_until"] or 0) - now)),
                    }
                )
            jobs = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT job_id, lane_key, status, requested_at, started_at,
                           session_key, source, display_name
                    FROM provider_queue_jobs
                    WHERE status IN ('queued', 'running')
                    ORDER BY requested_at, job_id
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            ]
            return {"lanes": lanes, "jobs": jobs}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS provider_queue_lanes (
                    lane_key TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    concurrency INTEGER NOT NULL,
                    cooldown_until REAL NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_queue_jobs (
                    job_id TEXT PRIMARY KEY,
                    lane_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    requested_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    heartbeat_at REAL,
                    lease_until REAL,
                    owner_id TEXT,
                    owner_pid INTEGER,
                    session_key TEXT,
                    source TEXT,
                    model TEXT,
                    provider TEXT,
                    account TEXT,
                    display_name TEXT,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_provider_queue_jobs_lane_status
                    ON provider_queue_jobs(lane_key, status, requested_at, job_id);
                CREATE INDEX IF NOT EXISTS idx_provider_queue_jobs_lease
                    ON provider_queue_jobs(status, lease_until);
                """
            )

    def _upsert_lane(self, conn: sqlite3.Connection, lane: ProviderQueueLane, now: float) -> None:
        conn.execute(
            """
            INSERT INTO provider_queue_lanes (
                lane_key, display_name, concurrency, cooldown_until, updated_at
            ) VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(lane_key) DO UPDATE SET
                display_name = excluded.display_name,
                concurrency = excluded.concurrency,
                updated_at = excluded.updated_at
            """,
            (lane.lane_key, lane.display_name, lane.concurrency, now),
        )

    def _expire_stale_jobs(self, conn: sqlite3.Connection, now: float) -> None:
        stale_before = now - self.settings.stale_seconds
        conn.execute(
            """
            UPDATE provider_queue_jobs
            SET status = 'expired',
                finished_at = ?,
                error = 'provider queue lease expired'
            WHERE status = 'running'
              AND (
                  COALESCE(lease_until, 0) < ?
                  OR COALESCE(heartbeat_at, started_at, requested_at) < ?
              )
            """,
            (now, now, stale_before),
        )

    def _cleanup_old_jobs(self, conn: sqlite3.Connection, now: float) -> None:
        cutoff = now - self.settings.retention_seconds
        conn.execute(
            """
            DELETE FROM provider_queue_jobs
            WHERE status IN ('done', 'failed', 'cancelled', 'expired')
              AND COALESCE(finished_at, requested_at) < ?
            """,
            (cutoff,),
        )

    def _state_from_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        acquired: bool,
    ) -> ProviderQueueState:
        lane_key = row["lane_key"]
        running = conn.execute(
            """
            SELECT COUNT(*) FROM provider_queue_jobs
            WHERE lane_key = ? AND status = 'running'
            """,
            (lane_key,),
        ).fetchone()[0]
        queued_ahead = conn.execute(
            """
            SELECT COUNT(*) FROM provider_queue_jobs
            WHERE lane_key = ?
              AND status = 'queued'
              AND (requested_at < ? OR (requested_at = ? AND job_id < ?))
            """,
            (lane_key, row["requested_at"], row["requested_at"], row["job_id"]),
        ).fetchone()[0]
        position = max(1, int(queued_ahead or 0) + 1)
        return ProviderQueueState(
            job_id=row["job_id"],
            lane_key=lane_key,
            display_name=row["display_name"] or lane_key,
            concurrency=max(1, int(row["lane_concurrency"] or 1)),
            running=int(running or 0),
            queued_ahead=int(queued_ahead or 0),
            position=position,
            cooldown_until=float(row["cooldown_until"] or 0),
            acquired=acquired,
        )

    def _cooldown_for_error(self, error: str) -> float:
        text = error.lower()
        if "429" in text or "rate limit" in text or "resource_exhausted" in text:
            retry_after = _parse_retry_after_seconds(text)
            return retry_after or self.settings.rate_limit_cooldown_seconds
        if "timeout" in text or "timed out" in text:
            return self.settings.timeout_cooldown_seconds
        return 0.0


class ProviderQueueJob:
    def __init__(self, queue: ProviderQueue, job_id: str, lane: ProviderQueueLane):
        self.queue = queue
        self.job_id = job_id
        self.lane = lane

    def wait_for_turn(
        self,
        *,
        on_wait: Callable[[ProviderQueueState], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> "ProviderQueueLease":
        last_notice_at = 0.0
        while True:
            if should_cancel and should_cancel():
                self.queue.cancel_job(self.job_id, "cancelled before provider call")
                raise ProviderQueueCancelled("provider queue wait cancelled")
            state = self.queue.try_acquire(self.job_id, self.lane)
            if state.acquired:
                return ProviderQueueLease(self.queue, self.job_id)
            now = time.time()
            if on_wait and now - last_notice_at >= self.queue.settings.notify_interval_seconds:
                on_wait(state)
                last_notice_at = now
            sleep_for = self.queue.settings.poll_interval_seconds
            if state.cooldown_until > now:
                sleep_for = min(sleep_for, max(0.2, state.cooldown_until - now))
            time.sleep(sleep_for)


class ProviderQueueLease:
    def __init__(self, queue: ProviderQueue, job_id: str):
        self.queue = queue
        self.job_id = job_id
        self._finished = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def finish(self, status: str = "done", error: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        self.queue.finish_job(self.job_id, status, error)
        self._thread.join(timeout=1.0)

    def __enter__(self) -> "ProviderQueueLease":
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc is not None:
            self.finish("failed", str(exc))
        else:
            self.finish("done")

    def _heartbeat_loop(self) -> None:
        interval = max(1.0, self.queue.settings.heartbeat_seconds)
        while not self._stop.wait(interval):
            try:
                self.queue.heartbeat(self.job_id)
            except Exception:
                pass


def format_wait_notice(state: ProviderQueueState) -> str:
    wait_part = ""
    if state.cooldown_until > time.time():
        wait_part = f", cooldown {int(state.wait_seconds)}s"
    return (
        f"Accepted. {state.display_name} queue position {state.position} "
        f"(running {state.running}/{state.concurrency}{wait_part})."
    )


def format_start_notice(state: ProviderQueueState) -> str:
    return f"Starting. {state.display_name} queue lease acquired."


def format_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    lanes = snapshot.get("lanes") if isinstance(snapshot, Mapping) else None
    if not lanes:
        return ["**Provider Queue:** idle"]
    lines = ["**Provider Queue:**"]
    for lane in lanes:
        lines.append(
            "- {name}: running {running}/{limit}, queued {queued}, oldest wait {wait}s{cooldown}".format(
                name=lane.get("display_name") or lane.get("lane_key"),
                running=lane.get("running", 0),
                limit=lane.get("concurrency", 1),
                queued=lane.get("queued", 0),
                wait=lane.get("oldest_wait_seconds", 0),
                cooldown=(
                    f", cooldown {lane.get('cooldown_seconds')}s"
                    if lane.get("cooldown_seconds")
                    else ""
                ),
            )
        )
    return lines


def _rule_matches(rule: Mapping[str, Any], key: str, actual: str, *, default: str = "*") -> bool:
    raw = rule.get(key, default)
    if raw is None:
        raw = default
    if isinstance(raw, list):
        return any(_match_pattern(str(item), actual) for item in raw)
    return _match_pattern(str(raw), actual)


def _match_pattern(pattern: str, actual: str) -> bool:
    pat = pattern.strip().lower()
    if pat in ("", "*"):
        return True
    return fnmatch.fnmatch(actual.lower(), pat)


def _candidate_provider_payloads(payload: Any, provider: str) -> list[Any]:
    candidates: list[Any] = []
    if not isinstance(payload, Mapping):
        return candidates
    direct = payload.get(provider)
    if direct is not None:
        candidates.append(direct)
    for key in ("providers", "credentials", "credential_pool", "auth", "profiles"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and provider in nested:
            candidates.append(nested[provider])
    return candidates


def _extract_account_from_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("account_id", "email", "username", "label"):
            raw = value.get(key)
            if isinstance(raw, str) and raw.strip():
                return _norm(raw)
        token = value.get("access_token")
        if isinstance(token, str):
            account = _extract_account_from_jwt(token)
            if account:
                return account
        for nested in value.values():
            account = _extract_account_from_value(nested)
            if account:
                return account
    elif isinstance(value, list):
        for item in value:
            account = _extract_account_from_value(item)
            if account:
                return account
    return None


def _extract_account_from_jwt(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload + padding).encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    profile = data.get("https://api.openai.com/profile")
    if isinstance(profile, Mapping):
        email = profile.get("email")
        if isinstance(email, str) and email.strip():
            return _norm(email)
    auth = data.get("https://api.openai.com/auth")
    if isinstance(auth, Mapping):
        account_id = auth.get("chatgpt_account_id") or auth.get("chatgpt_user_id")
        if isinstance(account_id, str) and account_id.strip():
            return _norm(account_id)
    sub = data.get("sub")
    return _norm(sub) if isinstance(sub, str) and sub.strip() else None


def _parse_retry_after_seconds(text: str) -> float:
    match = re.search(r"(?:retry[- ]after|try again in|retry in)\D{0,20}(\d{1,5})", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def _owner_id() -> str:
    host = os.uname().nodename if hasattr(os, "uname") else "host"
    return f"{host}:{os.getpid()}:{threading.get_ident()}"


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()
