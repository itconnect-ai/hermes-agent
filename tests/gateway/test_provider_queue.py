import base64
import json
import time

from gateway.provider_queue import (
    ProviderQueue,
    classify_provider_queue_priority,
    format_snapshot,
    load_provider_queue_settings,
    resolve_provider_account_identifier,
    resolve_provider_queue_lane,
)


def _config(tmp_path, **overrides):
    provider_queue = {
        "enabled": True,
        "db_path": str(tmp_path / "provider-queue.sqlite3"),
        "poll_interval_seconds": 0.01,
        "notify_interval_seconds": 0.01,
        "lease_seconds": 0.1,
        "heartbeat_seconds": 0.05,
        "stale_seconds": 0.1,
    }
    provider_queue.update(overrides)
    return {"provider_queue": provider_queue}


def test_provider_queue_is_opt_in():
    assert resolve_provider_queue_lane({}, provider="openai-codex", model="gpt-5.5") is None


def test_default_rule_builds_provider_account_model_lane(tmp_path):
    cfg = _config(tmp_path)

    lane = resolve_provider_queue_lane(
        cfg,
        provider="openai-codex",
        model="gpt-5.5",
        account="contact@itconnect.dev",
    )

    assert lane is not None
    assert lane.lane_key == "openai-codex:contact@itconnect.dev:gpt-5.5"
    assert lane.concurrency == 1


def test_fifo_acquire_respects_concurrency(tmp_path):
    cfg = _config(tmp_path)
    queue = ProviderQueue(load_provider_queue_settings(cfg))
    lane = resolve_provider_queue_lane(
        cfg,
        provider="openai-codex",
        model="gpt-5.5",
        account="acct-1",
    )

    first = queue.enqueue(lane, session_key="discord:one", source="discord")
    first_state = queue.try_acquire(first.job_id, lane)
    assert first_state.acquired is True

    second = queue.enqueue(lane, session_key="discord:two", source="discord")
    second_state = queue.try_acquire(second.job_id, lane)
    assert second_state.acquired is False
    assert second_state.running == 1
    assert second_state.position == 1

    queue.finish_job(first.job_id, "done")
    second_state = queue.try_acquire(second.job_id, lane)
    assert second_state.acquired is True


def test_heavy_job_priority_yields_to_normal_job(tmp_path):
    cfg = _config(tmp_path, heavy_message_chars=5, heavy_priority=10)
    queue = ProviderQueue(load_provider_queue_settings(cfg))
    lane = resolve_provider_queue_lane(
        cfg,
        provider="openai-codex",
        model="gpt-5.5",
        account="acct-1",
    )

    heavy_priority = classify_provider_queue_priority(cfg, message="long enough")
    heavy = queue.enqueue(
        lane,
        session_key="discord:heavy",
        source="discord",
        priority=heavy_priority,
    )
    normal = queue.enqueue(lane, session_key="discord:normal", source="discord")

    normal_state = queue.try_acquire(normal.job_id, lane)
    assert normal_state.acquired is True

    heavy_state = queue.try_acquire(heavy.job_id, lane)
    assert heavy_state.acquired is False
    assert heavy_state.priority == 10


def test_expired_lease_releases_next_job(tmp_path):
    cfg = _config(tmp_path)
    queue = ProviderQueue(load_provider_queue_settings(cfg))
    lane = resolve_provider_queue_lane(
        cfg,
        provider="openai-codex",
        model="gpt-5.5",
        account="acct-1",
    )

    first = queue.enqueue(lane, session_key="discord:one", source="discord")
    assert queue.try_acquire(first.job_id, lane).acquired is True
    time.sleep(0.2)

    second = queue.enqueue(lane, session_key="discord:two", source="discord")
    second_state = queue.try_acquire(second.job_id, lane)

    assert second_state.acquired is True


def test_rate_limit_failure_sets_lane_cooldown(tmp_path):
    cfg = _config(tmp_path, rate_limit_cooldown_seconds=10)
    queue = ProviderQueue(load_provider_queue_settings(cfg))
    lane = resolve_provider_queue_lane(
        cfg,
        provider="openai-codex",
        model="gpt-5.5",
        account="acct-1",
    )

    job = queue.enqueue(lane, session_key="discord:one", source="discord")
    assert queue.try_acquire(job.job_id, lane).acquired is True
    queue.finish_job(job.job_id, "failed", "429 rate limit")

    snapshot = queue.snapshot()
    assert snapshot["lanes"][0]["cooldown_seconds"] > 0
    assert "cooldown" in "\n".join(format_snapshot(snapshot))


def test_account_identifier_reads_oauth_jwt_profile(tmp_path):
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "https://api.openai.com/profile": {
                    "email": "Contact@Itconnect.dev",
                }
            }
        ).encode()
    ).rstrip(b"=").decode()
    token = f"{header}.{payload}."
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text(
        json.dumps(
            {
                "providers": {
                    "openai-codex": {
                        "credential": {
                            "access_token": token,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    account = resolve_provider_account_identifier(
        "openai-codex",
        hermes_home=hermes_home,
    )

    assert account == "contact@itconnect.dev"
