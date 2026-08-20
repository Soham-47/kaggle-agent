from pathlib import Path

import os
import multiprocessing
import time

from kaggle_agent.state_md import AgentState, RunLock, load_state, parse_kv_markdown, save_state


def test_parse_kv_markdown():
    d = parse_kv_markdown("# t\n\n- phase: PLAN\n- paused: false\n")
    assert d["phase"] == "PLAN"
    assert d["paused"] == "false"


def test_save_load_state(tmp_path: Path):
    save_state(AgentState(phase="RESEARCH", competition="rsna_knee"), tmp_path)
    loaded = load_state(tmp_path)
    assert loaded.phase == "RESEARCH"
    assert loaded.competition == "rsna_knee"
    assert loaded.loop_next_n == "3"


def test_loop_fields_round_trip(tmp_path: Path):
    save_state(
        AgentState(phase="IDLE", loop_last_score="0.52", loop_next_n="5"),
        tmp_path,
    )
    loaded = load_state(tmp_path)
    assert loaded.loop_last_score == "0.52"
    assert loaded.loop_next_n == "5"


def test_run_lock(tmp_path: Path):
    a, b = RunLock(tmp_path), RunLock(tmp_path)
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()


def test_run_lock_records_owner_pid(tmp_path: Path):
    a = RunLock(tmp_path)
    assert a.acquire() is True
    assert f"pid={os.getpid()}" in a.path.read_text(encoding="utf-8")
    a.release()


def test_run_lock_stale_dead_pid_taken_over(tmp_path: Path):
    a = RunLock(tmp_path)
    assert a.acquire() is True
    a.release()
    a.path.write_text("pid=999999999 at=2026-08-15T00:00:00+00:00\n", encoding="utf-8")
    b = RunLock(tmp_path)
    assert b.acquire() is True
    assert b.took_over is True
    b.release()


def test_run_lock_non_owner_release_does_not_remove_new_owner(tmp_path: Path):
    old = RunLock(tmp_path)
    assert old.acquire() is True
    old.release()
    old.path.write_text("pid=999999999 token=stale at=2026-08-15T00:00:00+00:00\n", encoding="utf-8")

    new = RunLock(tmp_path)
    assert new.acquire() is True
    # Model a delayed release from the old owner after takeover.
    old._held = True
    old._owner_token = "stale"
    old.release()
    assert new.path.exists()
    new.release()


def _lock_attempt(path: str, queue: multiprocessing.Queue) -> None:
    lock = RunLock(Path(path))
    acquired = lock.acquire()
    queue.put(acquired)
    if acquired:
        time.sleep(0.2)
        lock.release()


def test_run_lock_exclusive_across_processes(tmp_path: Path):
    queue: multiprocessing.Queue = multiprocessing.Queue()
    first = multiprocessing.Process(target=_lock_attempt, args=(str(tmp_path), queue))
    second = multiprocessing.Process(target=_lock_attempt, args=(str(tmp_path), queue))
    first.start()
    time.sleep(0.05)
    second.start()
    results = [queue.get(timeout=5), queue.get(timeout=5)]
    first.join(timeout=5)
    second.join(timeout=5)
    assert sorted(results) == [False, True]


def test_run_lock_legacy_lock_refused_while_fresh(tmp_path: Path):
    lock = tmp_path / "memory" / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("locked\n", encoding="utf-8")
    assert RunLock(tmp_path).acquire() is False


def test_run_lock_legacy_lock_stale_after_age(tmp_path: Path):
    lock = tmp_path / "memory" / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("locked\n", encoding="utf-8")
    old = time.time() - RunLock.STALE_AGE_SECONDS - 60
    os.utime(lock, (old, old))
    r = RunLock(tmp_path)
    assert r.acquire() is True
    assert r.took_over is True
    r.release()
