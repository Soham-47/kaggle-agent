"""Ops: traces, daily-log backfill, cycle evals, dashboard snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.ops.evals import collect_events, evaluate_cycle
from kaggle_agent.ops.log_parse import parse_daily_log
from kaggle_agent.ops.snapshot import build_snapshot
from kaggle_agent.ops.tracing import Tracer
from kaggle_agent.research.source_cards import write_methods_sidecar


def test_tracer_writes_jsonl_and_usage(tmp_path: Path):
    t = Tracer(tmp_path, cycle_id="c1")
    t.emit("cycle_start", competition="rsna_knee")
    t.emit("llm", model="deepseek-v4-flash", tokens_in=10, tokens_out=4)
    t.emit("tool", stage="research", tool="write_card", turn=1)
    rows = t.read_day()
    assert rows[0]["type"] == "cycle_start"
    assert rows[0]["cycle_id"] == "c1"
    assert rows[1]["type"] == "llm"
    usage = t.read_usage()
    assert usage[-1]["in"] == 10
    assert usage[-1]["out"] == 4


def test_parse_daily_log_extracts_phases_and_tools(tmp_path: Path):
    text = """# daily 2026-08-14

- 14:53:43 UTC: start rsna_knee dry=False
- 14:53:43 UTC: RESEARCH
- 14:55:03 UTC: research agent turn=1 tool=invalid_json
- 14:55:08 UTC: research agent turn=2 tool=pull_kernel
- 15:03:16 UTC: research agent stop=turn_cap turns=40
- 15:03:16 UTC: PLAN
- 15:04:10 UTC: plan agent turn=10 tool=write_plan
- 15:04:57 UTC: code recipe recipe metadata-ranker n_train=4407
"""
    events = parse_daily_log(text)
    kinds = [e["type"] for e in events]
    assert "cycle_start" in kinds
    assert "phase" in kinds
    tools = [e for e in events if e["type"] == "tool"]
    assert tools[0]["tool"] == "invalid_json"
    assert tools[0]["stage"] == "research"
    assert any(e["tool"] == "write_plan" for e in tools)


def test_collect_events_includes_durable_stage_ledger(tmp_path: Path):
    ledger = tmp_path / ".agent"
    ledger.mkdir()
    (ledger / "stage-ledger.jsonl").write_text(
        json.dumps(
            {
                "ts": "2026-08-20T00:00:00+00:00", "event": "stage_finished",
                "stage": "CODE", "cycle_id": "c1", "competition": "demo",
                "state": "success", "attempt": 1, "idempotency_key": "abc",
            }
        ) + "\n",
        encoding="utf-8",
    )

    events = collect_events(tmp_path)

    assert any(
        event.get("type") == "stage_outcome" and event.get("stage") == "CODE"
        for event in events
    )


def test_evaluate_cycle_fails_on_junk_and_invalid_json(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "research.md").write_text("# research\nno cards heading\n", encoding="utf-8")
    ws = tmp_path / "competitions" / "rsna_knee"
    (ws / "pipeline").mkdir(parents=True)
    (ws / "pipeline" / "methods.json").write_text(
        json.dumps(
            {
                "dataset_sources": ["dataset/model"],
                "model_sources": ["dinov2/pytorch"],
                "implement_steps": ["Attach datasets ['dataset/model']"],
            }
        ),
        encoding="utf-8",
    )
    events = [
        {"type": "tool", "stage": "research", "tool": "invalid_json", "turn": i}
        for i in range(1, 21)
    ]
    events.append({"type": "tool", "stage": "research", "tool": "search", "turn": 21})
    report = evaluate_cycle(tmp_path, events, workspace=ws)
    names = {c["id"]: c for c in report["checks"]}
    assert names["invalid_json_rate"]["ok"] is False
    assert names["research_wrote_card"]["ok"] is False
    assert names["methods_pins_valid"]["ok"] is False
    assert names["cards_feasible"]["ok"] is False
    assert report["passed"] is False


def test_evaluate_cycle_passes_clean_cards(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    research = mem / "research.md"
    research.write_text("## Method cards\n- next: attach owner/weights\n", encoding="utf-8")
    deep = mem / "research-deep"
    deep.mkdir()
    card = deep / "source-good.md"
    card.write_text(
        "# good\n- copyable next step: Attach pilkwang/rsna-knee-weights and rank-average.\n"
        "datasets_mentioned: pilkwang/rsna-knee-weights\n",
        encoding="utf-8",
    )
    ws = tmp_path / "competitions" / "rsna_knee"
    write_methods_sidecar([card], ws)
    events = [
        {"type": "tool", "stage": "research", "tool": "write_card", "turn": 1},
        {"type": "tool", "stage": "research", "tool": "judge_cards", "turn": 2},
        {"type": "tool", "stage": "research", "tool": "done", "turn": 3},
    ]
    report = evaluate_cycle(tmp_path, events, workspace=ws)
    assert report["passed"] is True


def test_terminal_keeps_phases_and_collapses_invalid_json():
    from kaggle_agent.ops.terminal import relevant_lines

    events = [
        {"type": "phase", "ts": "15:00:00", "phase": "RESEARCH"},
        {"type": "tool", "ts": "15:00:01", "stage": "research", "tool": "invalid_json", "turn": 1},
        {"type": "tool", "ts": "15:00:02", "stage": "research", "tool": "invalid_json", "turn": 2},
        {"type": "tool", "ts": "15:00:03", "stage": "research", "tool": "pull_kernel", "turn": 3},
        {"type": "agent_stop", "ts": "15:00:04", "stage": "research", "reason": "turn_cap"},
        {"type": "cycle_end", "ts": "15:00:05", "detail": "end errors=['kernel:push failed']"},
    ]
    lines = relevant_lines(events)
    texts = [ln["text"] for ln in lines]
    assert any("RESEARCH" in t for t in texts)
    assert any("invalid_json" in t and "2" in t for t in texts)
    assert any("pull_kernel" in t for t in texts)
    assert any("turn_cap" in t for t in texts)
    assert any("kernel:push" in t for t in texts)
    assert not any(t.endswith(" · invalid_json") for t in texts)


def test_snapshot_includes_memory_and_architecture(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("prefs", encoding="utf-8")
    (mem / "COMPETITION.md").write_text("contest", encoding="utf-8")
    (mem / "state.md").write_text(
        "- phase: RESEARCH\n- public_best: 0.526\n- last_result: running\n- lock_held: true\n",
        encoding="utf-8",
    )
    (mem / "research.md").write_text("## Method cards\nstep\n", encoding="utf-8")
    (mem / "heal.md").write_text("- decision_next: wait_approve\n", encoding="utf-8")
    daily = mem / "daily"
    daily.mkdir()
    (daily / "2026-08-14.md").write_text(
        "- 15:00:00 UTC: RESEARCH\n- 15:00:01 UTC: research agent turn=1 tool=invalid_json\n",
        encoding="utf-8",
    )
    snap = build_snapshot(tmp_path)
    assert snap["state"]["public_best"] == "0.526"
    assert snap["running"] is True
    assert snap["active_node"] == "research"
    assert any(n["id"] == "research" for n in snap["architecture"]["nodes"])
    assert snap["terminal"]
    assert snap["evals"]["checks"]


def test_dashboard_snapshot_route(tmp_path: Path):
    from http.client import HTTPConnection
    from threading import Thread

    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("p", encoding="utf-8")
    (mem / "COMPETITION.md").write_text("c", encoding="utf-8")
    (mem / "state.md").write_text("- phase: IDLE\n- public_best: 0.526\n", encoding="utf-8")
    (mem / "research.md").write_text("## Method cards\n", encoding="utf-8")
    from http.server import ThreadingHTTPServer

    from kaggle_agent.ops.dashboard import make_handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path))
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        conn.request("GET", "/api/snapshot")
        res = conn.getresponse()
        body = json.loads(res.read().decode("utf-8"))
        assert res.status == 200
        assert body["state"]["public_best"] == "0.526"
        conn.request("GET", "/")
        html = conn.getresponse()
        page = html.read()
        assert html.status == 200
        assert b'id="run-live"' in page
        assert b'id="term"' in page
    finally:
        httpd.shutdown()


def test_post_command_run_starts_live_cycle(tmp_path: Path):
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    from threading import Thread

    from kaggle_agent.ops.dashboard import make_handler

    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "state.md").write_text(
        "- phase: IDLE\n- paused: false\n- lock_held: false\n",
        encoding="utf-8",
    )
    started: list[tuple[bool, str]] = []

    def start_cycle(*, dry_run: bool, command: str) -> str:
        started.append((dry_run, command))
        return "started live"

    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(tmp_path, start_cycle=start_cycle),
    )
    Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        conn.request(
            "POST",
            "/api/command",
            body=json.dumps({"text": "/run"}),
            headers={"Content-Type": "application/json"},
        )
        res = conn.getresponse()
        body = json.loads(res.read().decode("utf-8"))
        assert res.status == 200
        assert body["ok"] is True
        assert started == [(False, "/run")]
        conn.request(
            "POST",
            "/api/command",
            body=json.dumps({"text": "/run dry"}),
            headers={"Content-Type": "application/json"},
        )
        dry = json.loads(conn.getresponse().read().decode("utf-8"))
        assert dry["ok"] is True
        assert started[-1] == (True, "/run dry")
    finally:
        httpd.shutdown()


def test_stage_agent_emits_tool_trace(tmp_path: Path):
    class _Zen:
        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            return json.dumps({"tool": "done", "args": {}})

    tracer = Tracer(tmp_path, cycle_id="t")
    agent = StageAgent(
        _Zen(),
        "m",
        {},
        StageAgentConfig(max_minutes=5, max_tool_turns=3),
        system="x",
        tracer=tracer,
        name="research",
    )
    agent.run("ctx")
    kinds = [e["type"] for e in tracer.read_day()]
    assert "llm" in kinds
    assert "tool" in kinds
