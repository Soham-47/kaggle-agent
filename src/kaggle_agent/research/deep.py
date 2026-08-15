"""Deep research stage — recursive breadth x depth loop (dzhng/deep-research pattern).

Iterates: LLM generates SERP queries → sources return hits → LLM distills
learnings + follow-ups → recurse until depth 0 → LLM writes final report.

Sources gather everything relevant: Kaggle notebooks (pulled source code),
arXiv papers, GitHub repos, generic web pages. Fail-soft per source.

All LLM calls go through ZenClient.chat with strict-JSON prompts. No new deps.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from kaggle_agent.llm.zen_client import ZenClient
from kaggle_agent.research.browser import (
    FetchFn,
    fetch_via_http,
    merge_section_into_research_md,
)

_DEEP_MARKER = "## Deep research digest"

_MAX_CONTENT_CHARS = 25_000


class ResearchSourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceHit:
    url: str
    title: str
    snippet: str = ""
    kind: str = "web"


class Source(Protocol):
    kind: str

    def search(self, query: str, limit: int) -> list[SourceHit]: ...

    def content(self, hit: SourceHit, max_chars: int = _MAX_CONTENT_CHARS) -> str: ...


class KaggleSource:
    """Search Kaggle kernels by query; content = pulled notebook source."""

    kind = "kaggle"

    def __init__(self, client: Any, competition: str, cache_dir: Path) -> None:
        self._client = client
        self._competition = competition
        self._cache = cache_dir
        self._cache.mkdir(parents=True, exist_ok=True)

    def search(self, query: str, limit: int = 5) -> list[SourceHit]:
        try:
            rows = self._client.api.kernels_list(
                competition=self._competition,
                search=query,
                page_size=limit,
                sort_by="voteCount",
            ) or []
        except Exception:  # noqa: BLE001
            return []
        out: list[SourceHit] = []
        for k in rows[:limit]:
            if k is None:
                continue
            ref = (
                str(k.get("ref", "") or "")
                if isinstance(k, dict)
                else str(getattr(k, "ref", "") or "")
            )
            if not ref:
                continue
            title = (
                str(k.get("title", "") or ref)
                if isinstance(k, dict)
                else str(getattr(k, "title", "") or ref)
            )
            votes = (
                k.get("total_votes", "")
                if isinstance(k, dict)
                else str(getattr(k, "total_votes", "") or "")
            )
            out.append(
                SourceHit(
                    url=f"https://www.kaggle.com/code/{ref}",
                    title=title,
                    snippet=str(votes or ""),
                    kind="kaggle",
                )
            )
        return out

    def content(self, hit: SourceHit, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        ref = hit.url.rsplit("/code/", 1)[-1]
        slug = ref.rsplit("/", 1)[-1]
        cached_names = (f"{ref.replace('/', '__')}.ipynb", f"{slug}.ipynb")
        cached = next(
            (self._cache / n for n in cached_names if (self._cache / n).is_file()), None
        )
        if cached is None:
            try:
                self._client.api.kernels_pull(ref, str(self._cache))
            except Exception:  # noqa: BLE001
                return ""
            cached = next(
                (self._cache / n for n in cached_names if (self._cache / n).is_file()), None
            )
        if cached is None:
            return ""
        return _notebook_text(cached)[:max_chars]


def _prefer_primary_sources(urls: list[str]) -> list[str]:
    """Kaggle notebooks first, then other primary URLs. Deduped."""
    seen: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.append(u)

    def _rank(url: str) -> tuple[int, str]:
        low = url.lower()
        if "kaggle.com/code/" in low:
            return (0, url)
        if "kaggle.com/competitions/" in low:
            return (1, url)
        if "arxiv.org" in low:
            return (2, url)
        if "github.com" in low:
            return (3, url)
        return (5, url)

    return sorted(seen, key=_rank)


def _notebook_text(path: Path) -> str:
    """Flatten an .ipynb into readable markdown + code text (no outputs)."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""
    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []) or [])
        if not src.strip():
            continue
        ctype = cell.get("cell_type", "")
        if ctype == "markdown":
            parts.append(src)
        elif ctype == "code":
            parts.append(f"```python\n{src}\n```")
        else:
            parts.append(src)
    return "\n\n".join(parts)


class ArxivSource:
    """arXiv API (no key): titles + abstracts as hits."""

    kind = "arxiv"

    def search(self, query: str, limit: int = 5) -> list[SourceHit]:
        url = (
            "http://export.arxiv.org/api/query?"
            + urllib.parse.urlencode(
                {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                }
            )
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                raw = resp.read()
        except Exception:  # noqa: BLE001
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: list[SourceHit] = []
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            return []
        for entry in root.findall("a:entry", ns)[:limit]:
            title = (entry.findtext("a:title", "", ns) or "").strip()
            link = entry.find("a:id", ns)
            abstract = (entry.findtext("a:summary", "", ns) or "").strip()
            out.append(
                SourceHit(
                    url=(link.text or "").strip(),
                    title=re.sub(r"\s+", " ", title),
                    snippet=re.sub(r"\s+", " ", abstract)[:400],
                    kind="arxiv",
                )
            )
        return out

    def content(self, hit: SourceHit, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        return hit.snippet[:max_chars]


class GithubSource:
    """GitHub repository search API (no key) + raw README as content."""

    kind = "github"

    def search(self, query: str, limit: int = 5) -> list[SourceHit]:
        url = (
            "https://api.github.com/search/repositories?"
            + urllib.parse.urlencode({"q": query, "per_page": limit})
        )
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "kaggle-agent/0.1", "Accept": "application/vnd.github+json"}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return []
        out: list[SourceHit] = []
        for item in payload.get("items", [])[:limit]:
            full = item.get("full_name", "")
            if not full:
                continue
            out.append(
                SourceHit(
                    url=f"https://github.com/{full}",
                    title=full,
                    snippet=str(item.get("description") or ""),
                    kind="github",
                )
            )
        return out

    def content(self, hit: SourceHit, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        repo = hit.url.rstrip("/").rsplit("github.com/", 1)[-1]
        for branch in ("main", "master"):
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "kaggle-agent/0.1"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                if raw.strip():
                    return raw[:max_chars]
            except Exception:  # noqa: BLE001
                continue
        return hit.snippet[:max_chars]


class WebSource:
    """Generic web: DuckDuckGo HTML SERP + fetched page text."""

    kind = "web"

    def __init__(self, fetch: FetchFn | None = None) -> None:
        self._fetch = fetch or fetch_via_http

    def search(self, query: str, limit: int = 5) -> list[SourceHit]:
        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return []
        out: list[SourceHit] = []
        for m in re.finditer(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html
        ):
            if len(out) >= limit:
                break
            href = m.group(1)
            if "uddg=" in href:
                url_text = urllib.parse.unquote(href.split("uddg=")[-1].split("&")[0])
            else:
                url_text = href
            if not url_text.startswith("http"):
                continue
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            out.append(SourceHit(url=url_text, title=title, kind="web"))
        return out

    def content(self, hit: SourceHit, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        try:
            return self._fetch(hit.url, max_chars)
        except Exception:  # noqa: BLE001
            return ""


def _json_completion(
    client: ZenClient,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 2048,
    retries: int = 1,
    plain_wrap_key: str | None = None,
) -> dict[str, Any]:
    """One strict-JSON LLM call; returns a dict. Raises on failure.

    If every attempt returns prose (flash models do this) and plain_wrap_key
    is set, return {plain_wrap_key: raw} instead of raising. Used for the
    final report, where prose is still a usable report.
    """
    raw = client.chat(
        model,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n\nReply with ONLY valid JSON. No prose."},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    if retries > 0:
        return _json_completion(
            client,
            model,
            system,
            user,
            max_tokens=max_tokens,
            retries=retries - 1,
            plain_wrap_key=plain_wrap_key,
        )
    if plain_wrap_key is not None:
        return {plain_wrap_key: (raw or "").strip()}
    raise ResearchSourceError(f"LLM returned invalid JSON: {raw[:200]!r}")


_SYSTEM = (
    "You are an expert researcher for Kaggle competitions. "
    "Follow the dzhng/deep-research loop: SERP queries with a researchGoal, "
    "dense learnings, then follow-up directions. "
    "Prefer site:kaggle.com/code, pinned discussions, and papers notebooks cite. "
    "Ignore off-topic arXiv that only shares an AUC keyword. "
    "Gather exact numbers, methods, architectures, and entity names. "
    "Today is "
    + time.strftime("%Y-%m-%d")
    + "."
)


@dataclass
class DeepResearchConfig:
    enabled: bool = True
    breadth: int = 3
    depth: int = 2
    max_queries: int = 12
    per_query_limit: int = 5
    max_learnings: int = 4
    max_followups: int = 3
    max_fetches: int = 40
    max_minutes: float = 15.0
    report_dir: str = "memory/research-deep"


@dataclass
class DeepResearchResult:
    report_path: Path | None = None
    learnings: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    queries_run: int = 0
    error: str = ""


class DeepResearcher:
    """Recursive research loop. Injectable llm + sources for tests."""

    def __init__(
        self,
        client: ZenClient | None,
        model: str,
        config: DeepResearchConfig,
        sources: list[Source],
        root: Path,
        *,
        log: Any = None,
        relevance_terms: tuple[str, ...] = (),
    ) -> None:
        self._client = client
        self._model = model
        self._config = config
        self._sources = sources
        self._root = root
        self._log = log
        self._relevance_terms = tuple(t.lower() for t in relevance_terms if t)
        self._deadline = time.monotonic() + config.max_minutes * 60
        self._query_count = 0
        self._fetch_count = 0

    def _logmsg(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def _budget_left(self) -> bool:
        return self._query_count < self._config.max_queries and time.monotonic() < self._deadline

    def _findings_enough(self, learnings: list[str]) -> bool:
        """Waku-style judge: stop adding depth when facts are implementable."""
        if self._client is None or len(learnings) < 3:
            return False
        user = (
            "Do these learnings give a coding agent datasets to attach, "
            "how to find hidden test IDs, and an ensemble rule? "
            'Return JSON: {"enough": bool, "gap": str}\n\n'
            + "\n".join(f"- {x}" for x in learnings[:16])
        )
        try:
            parsed = _json_completion(
                self._client, self._model, _SYSTEM, user, max_tokens=300
            )
        except Exception:  # noqa: BLE001
            return False
        return bool(parsed.get("enough"))

    def _generate_queries(self, query: str, learnings: list[str], num: int) -> list[dict[str, str]]:
        if self._client is None:
            return [{"query": query, "researchGoal": query}]
        learn = "\n".join(f"- {x}" for x in learnings[-12:])
        user = (
            f"Given the prompt, generate up to {num} unique search queries "
            f"(dzhng SERP style). At least one query must include site:kaggle.com. "
            f"Each must state its researchGoal. "
            f"Return JSON: {{\"queries\": [{{\"query\": str, \"researchGoal\": str}}]}}\n\n"
            f"<prompt>{query}</prompt>\n\n"
            + (f"Learnings so far:\n{learn}" if learn else "")
        )
        parsed = _json_completion(self._client, self._model, _SYSTEM, user)
        items = parsed.get("queries") or []
        out: list[dict[str, str]] = []
        for it in items[:num]:
            q = str(it.get("query", "")).strip()
            if q:
                out.append({"query": q, "researchGoal": str(it.get("researchGoal", ""))})
        return out or [{"query": query, "researchGoal": query}]

    def _relevant(self, text: str) -> bool:
        """Keep fetched content that mentions at least one relevance term."""
        if not self._relevance_terms:
            return True
        low = text.lower()
        return any(t in low for t in self._relevance_terms)

    def _fetch_all(self, hits: list[SourceHit]) -> list[str]:
        out: list[str] = []
        if not hits:
            return out
        by_kind: dict[str, Source] = {s.kind: s for s in self._sources}
        with ThreadPoolExecutor(max_workers=min(4, len(hits))) as pool:
            futs = {
                pool.submit(by_kind[hit.kind].content, hit): hit
                for hit in hits
                if hit.kind in by_kind
            }
            for fut in as_completed(futs):
                if self._fetch_count >= self._config.max_fetches:
                    break
                try:
                    text = fut.result()
                except Exception:  # noqa: BLE001
                    text = ""
                if not text:
                    continue
                self._fetch_count += 1
                if self._relevant(text):
                    out.append(text[: _MAX_CONTENT_CHARS])
        return out

    def _distill(self, query: str, hits: list[SourceHit], num_learn: int, num_follow: int) -> dict[str, Any]:
        contents = self._fetch_all(hits)
        if self._client is None or not contents:
            return {
                "learnings": [
                    h.snippet for h in hits if h.snippet and self._relevant(h.snippet)
                ][:num_learn],
                "followUpQuestions": [],
            }
        blocks = "\n".join(f"<content>\n{c}\n</content>" for c in contents)
        user = (
            f"Given the search results for <query>{query}</query>, return up to "
            f"{num_learn} unique, information-dense learnings (include exact metrics, "
            f"architectures, entities) and up to {num_follow} follow-up questions. "
            f'Return JSON: {{"learnings": [str], "followUpQuestions": [str]}}\n\n{blocks}'
        )
        parsed = _json_completion(self._client, self._model, _SYSTEM, user, max_tokens=3072)
        return {
            "learnings": [str(x) for x in (parsed.get("learnings") or [])][:num_learn],
            "followUpQuestions": [str(x) for x in (parsed.get("followUpQuestions") or [])][
                :num_follow
            ],
        }

    def research(
        self,
        query: str,
        breadth: int,
        depth: int,
        learnings: list[str],
        visited: list[str],
    ) -> tuple[list[str], list[str]]:
        if depth <= 0 or not self._budget_left():
            return learnings, visited
        self._query_count += 1
        queries = self._generate_queries(query, learnings, breadth)
        self._logmsg(f"deep depth={depth} queries={len(queries)} total={self._query_count}")

        def _one(item: dict[str, str]) -> tuple[dict[str, str], list[SourceHit], dict[str, Any]]:
            try:
                hits = self._gather_hits(item["query"], self._config.per_query_limit)
            except Exception:  # noqa: BLE001
                hits = []
            distilled = (
                self._distill(
                    item["query"],
                    hits,
                    self._config.max_learnings,
                    self._config.max_followups,
                )
                if hits
                else {"learnings": [], "followUpQuestions": []}
            )
            return item, hits, distilled

        packed: list[tuple[dict[str, str], list[SourceHit], dict[str, Any]]] = []
        if queries:
            with ThreadPoolExecutor(max_workers=min(4, len(queries))) as pool:
                for item, hits, distilled in pool.map(_one, queries):
                    if not self._budget_left():
                        break
                    packed.append((item, hits, distilled))

        new_breadth = max(1, breadth // 2)
        for item, hits, distilled in packed:
            visited.extend(h.url for h in hits)
            learnings = _dedupe(learnings + distilled["learnings"])
            if depth - 1 > 0 and distilled["followUpQuestions"]:
                follow = (
                    f"Previous research goal: {item['researchGoal']}\n"
                    "Follow-up directions: "
                    + "\n".join(f"- {q}" for q in distilled["followUpQuestions"])
                )
                learnings, visited = self.research(
                    follow, new_breadth, depth - 1, learnings, visited
                )
        if self._client is not None and learnings and depth <= 1:
            if self._findings_enough(learnings):
                self._logmsg("deep judge: enough implementable facts")
        return learnings, visited

    def _gather_hits(self, query: str, limit: int) -> list[SourceHit]:
        hits: list[SourceHit] = []
        for src in self._sources:
            try:
                hits.extend(src.search(query, limit))
            except Exception:  # noqa: BLE001
                continue
        return hits

    def _write_report(self, prompt: str, learnings: list[str], visited: list[str]) -> Path:
        report_dir = self._root / self._config.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        path = report_dir / f"deep-{stamp}.md"

        body = self._report_markdown(prompt, learnings)
        urls = "\n".join(f"- {u}" for u in dict.fromkeys(visited))
        path.write_text(body + "\n\n## Sources\n\n" + urls + "\n", encoding="utf-8")
        return path

    def _report_markdown(self, prompt: str, learnings: list[str]) -> str:
        if self._client is None:
            return (
                f"# Deep research: {prompt[:80]}\n\n"
                + "\n".join(f"- {l}" for l in learnings)
            )
        learn = "\n".join(f"<learning>\n{x}\n</learning>" for x in learnings)
        user = (
            "Write a detailed research report for the prompt using the learnings. "
            "Aim for 3+ pages. Include ALL learnings. Mark speculation as such. "
            "Return JSON: {\"reportMarkdown\": str}\n\n"
            f"<prompt>{prompt}</prompt>\n\n<learnings>\n{learn}\n</learnings>"
        )
        parsed = _json_completion(
            self._client,
            self._model,
            _SYSTEM,
            user,
            max_tokens=8192,
            retries=2,
            plain_wrap_key="reportMarkdown",
        )
        return str(parsed.get("reportMarkdown") or "").strip() or "# Deep research report"

    def digest_markdown(self, learnings: list[str], sources: list[str]) -> str:
        lines = [_DEEP_MARKER, "", "Distilled from articles, papers, notebooks, repos, web.", ""]
        lines += [f"- {l}" for l in learnings[:20]]
        lines.append("")
        lines += [f"- source: {u}" for u in _prefer_primary_sources(sources)[:15]]
        return "\n".join(lines)

    def run(self, prompt: str, research_path: Path | None = None) -> DeepResearchResult:
        result = DeepResearchResult()
        if self._client is None and not any(
            isinstance(s, (KaggleSource, ArxivSource, GithubSource, WebSource)) for s in self._sources
        ):
            result.error = "no llm and no sources"
            return result
        try:
            learnings, visited = self.research(prompt, self._config.breadth, self._config.depth, [], [])
            result.learnings = learnings
            result.sources = list(dict.fromkeys(visited))
            result.queries_run = self._query_count
            if learnings:
                result.report_path = self._write_report(prompt, learnings, visited)
                if research_path is not None:
                    merge_section_into_research_md(
                        research_path, _DEEP_MARKER, self.digest_markdown(learnings, visited)
                    )
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
        return result


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        key = re.sub(r"\W+", "", x.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(x)
    return out
