"""A small continuous-batching simulator for serving experiments."""

from collections import deque
from statistics import mean


class SimContinuousBatcher:
    """Simulate continuous batching at decode-iteration granularity.

    step(t) runs one decode iteration ending at simulator time t; t must
    be strictly increasing between calls. Each step: queued requests whose
    arrival time is at or before t join the running batch while slots are
    free (FIFO), every running request produces one token stamped t, and
    requests that have produced gen_len tokens complete. Completed
    request ids come back from step(); their slots free up for the next
    step.

    stats() reports means over completed requests only; each stat is 0.0
    when no request has completed yet. TTFT is the stamp of the first
    token minus the arrival time. TBT is the mean gap between consecutive
    token stamps. Throughput is gen_len over (finish time minus arrival
    time).
    """

    def __init__(self, max_batch: int, max_seq_len: int, kv_per_token: float):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        if kv_per_token <= 0:
            raise ValueError("kv_per_token must be positive")
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.kv_per_token = kv_per_token
        self.running = 0
        self._now = None
        self._next_id = 0
        self._queue = deque()
        self._active = {}
        self._done = []

    def submit(self, arrival_time: float, gen_len: int) -> int:
        """Queue a request that will generate gen_len tokens; return its id."""
        if not 1 <= gen_len <= self.max_seq_len:
            raise ValueError(f"gen_len must be between 1 and {self.max_seq_len}")
        rid = self._next_id
        self._next_id += 1
        self._queue.append(
            {"id": rid, "arrival": arrival_time, "gen_len": gen_len, "tokens": []}
        )
        return rid

    def step(self, t: float) -> list[int]:
        """Run one decode iteration ending at time t; return completed ids."""
        if self._now is not None and t <= self._now:
            raise ValueError("step times must be strictly increasing")
        self._now = t
        while self._queue and self.running < self.max_batch:
            req = self._queue[0]
            if req["arrival"] > t:
                break
            self._queue.popleft()
            self._active[req["id"]] = req
            self.running += 1
        completed = []
        for rid in list(self._active):
            req = self._active[rid]
            req["tokens"].append(t)
            if len(req["tokens"]) >= req["gen_len"]:
                completed.append(rid)
                del self._active[rid]
                self.running -= 1
                self._done.append(self._record(req))
        return completed

    def stats(self) -> dict:
        """Return mean ttft, mean tbt, and mean throughput over completed requests."""
        if not self._done:
            return {"mean_ttft": 0.0, "mean_tbt": 0.0, "mean_throughput_tok_per_s": 0.0}
        return {
            "mean_ttft": mean(d["ttft"] for d in self._done),
            "mean_tbt": mean(d["tbt"] for d in self._done),
            "mean_throughput_tok_per_s": mean(d["throughput"] for d in self._done),
        }

    def _record(self, req: dict) -> dict:
        tokens = req["tokens"]
        deltas = [b - a for a, b in zip(tokens, tokens[1:])]
        tbt = mean(deltas) if deltas else 0.0
        return {
            "ttft": tokens[0] - req["arrival"],
            "tbt": tbt,
            "throughput": req["gen_len"] / (tokens[-1] - req["arrival"]),
        }