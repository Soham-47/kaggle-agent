"""Read-only browser research for competition HTML pages.

Production: browser-use / browser-harness on the headed research Chrome.
Tests: inject a fetch(url, max_chars) -> text callable.

Never used for submit. Skill: browser-harness (new_tab + js body text).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

FetchFn = Callable[[str, int], str]
SerpFn = Callable[[str, int], list[tuple[str, str, str]]]  # (title, url, snippet)


def _harness_cli() -> str | None:
    """Locate the browser-harness CLI even when ~/.local/bin is off PATH."""
    cli = shutil.which("browser-use") or shutil.which("browser-harness")
    if cli:
        return cli
    home = Path.home() / ".local" / "bin"
    for name in ("browser-use", "browser-harness"):
        candidate = home / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None

_EMPTY_MARKERS = frozenset({"", "{}", "None", "null"})
_BROWSER_MARKER = "## Browser (read-only)"


def merge_section_into_research_md(research_path: Path, marker: str, section: str) -> None:
    """Insert or replace one section (by heading marker) in research.md.

    All other sections are preserved, so research.md survives the full
    snapshot overwrite at the start of the next RESEARCH phase.
    """
    research_path.parent.mkdir(parents=True, exist_ok=True)
    text = section.rstrip() + "\n"
    if not research_path.is_file():
        research_path.write_text("# research\n\n" + text, encoding="utf-8")
        return

    existing = research_path.read_text(encoding="utf-8")
    if marker in existing:
        before, _, after = existing.partition(marker)
        rest = after.split("\n## ", 1)
        if len(rest) == 1:
            research_path.write_text(before.rstrip() + "\n\n" + text, encoding="utf-8")
            return
        research_path.write_text(
            before.rstrip() + "\n\n" + text + "## " + rest[1].lstrip(), encoding="utf-8"
        )
    else:
        research_path.write_text(existing.rstrip() + "\n\n" + text, encoding="utf-8")


class BrowserResearchError(RuntimeError):
    pass


def competition_page_urls(slug: str) -> dict[str, str]:
    base = f"https://www.kaggle.com/competitions/{slug}"
    return {
        "overview": f"{base}/overview",
        "discussion": f"{base}/discussion",
        "data": f"{base}/data",
    }


def _clip(text: str, n: int = 220) -> str:
    text = text.strip()
    return text if len(text) <= n else text[:n] + "…"


def _bullets_from_text(text: str, *, limit: int = 12) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ["(empty page text)"]
    bullets: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", cleaned):
        if len(part.strip()) < 40:
            continue
        bullets.append(_clip(part))
        if len(bullets) >= limit:
            return bullets
    return bullets or [_clip(cleaned)]


@dataclass
class BrowserNotes:
    slug: str
    pages: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_markdown_section(self) -> str:
        lines = [
            _BROWSER_MARKER,
            "",
            f"Source pages for `{self.slug}` (not for submit).",
            "",
        ]
        for name, text in self.pages.items():
            lines += [f"### {name}", ""]
            lines += [f"- {b}" for b in _bullets_from_text(text)]
            lines.append("")
        if self.errors:
            lines += ["### Browser errors", ""]
            lines += [f"- {e}" for e in self.errors]
            lines.append("")
        return "\n".join(lines)


def _normalize_harness_stdout(stdout: str) -> str:
    lines = [
        ln
        for ln in (stdout or "").splitlines()
        if ln.strip() and ln.strip() not in _EMPTY_MARKERS
    ]
    return "\n".join(lines).strip()


def fetch_via_browser_harness(url: str, max_chars: int = 12000) -> str:
    """Extract document body text via browser-harness."""
    cli = _harness_cli()
    if not cli:
        raise BrowserResearchError("browser-use / browser-harness not on PATH")

    script = f"""
import time
new_tab({url!r})
wait_for_load()
time.sleep(1)
js('''(() => {{
  const btns = Array.from(document.querySelectorAll("button"));
  const ok = btns.find(b => /ok,?\\s*got it/i.test(b.innerText || ""));
  if (ok) ok.click();
  return true;
}})()''')
time.sleep(2)
raw = js('''(() => {{
  const body = document.body ? document.body.innerText : '';
  const root = document.documentElement ? document.documentElement.innerText : '';
  const t = (body && body.trim()) ? body : root;
  return (t || '').slice(0, 15000);
}})()''')
if isinstance(raw, dict):
    raw = raw.get('text') or raw.get('result') or raw.get('value') or ''
print(raw if isinstance(raw, str) else str(raw or ''))
"""
    env = os.environ.copy()
    if not env.get("BU_CDP_URL") and not env.get("BU_CDP_WS"):
        # Headed research Chrome from scripts/start_research_chrome.sh (not daily 9222).
        env.setdefault("BU_CDP_URL", "http://127.0.0.1:9224")
    try:
        proc = subprocess.run(
            [cli],
            input=script,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise BrowserResearchError(f"browser-harness timeout for {url}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[:400]
        raise BrowserResearchError(f"browser-harness exit {proc.returncode}: {err}")

    out = _normalize_harness_stdout(proc.stdout or "")
    if out in _EMPTY_MARKERS:
        raise BrowserResearchError(f"empty page text for {url}")
    return out[:max_chars]


def fetch_via_http(url: str, max_chars: int = 12000) -> str:
    """Stdlib fallback — weak on JS-heavy Kaggle pages; fine for tests/static HTML."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "kaggle-agent/0.1 (research; +local)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise BrowserResearchError(f"http fetch failed: {exc}") from exc

    text = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def search_via_browser_harness(query: str, limit: int = 5) -> list[tuple[str, str, str]]:
    """SERP through the headed research Chrome. Returns (title, url, snippet)."""
    cli = _harness_cli()
    if not cli:
        raise BrowserResearchError("browser-use / browser-harness not on PATH")

    url = "https://www.google.com/search?" + urllib.parse.urlencode({"q": query})
    js_consent = """(() => {
  const btns = Array.from(document.querySelectorAll("button"));
  const ok = btns.find(b => /ok,\\s*got it/i.test(b.innerText || ""));
  if (ok) ok.click();
  return true;
})()"""
    js_extract = """(() => {
  const items = [];
  const skipHosts = /^(https?:\\/\\/)?(www\\.)?(google|scholar\\.google|support\\.google)\\./i;
  document.querySelectorAll('a').forEach(a => {
    const href = a.href || '';
    if (!href.startsWith('http')) return;
    if (skipHosts.test(href)) return;
    const txt = (a.innerText || '').trim().split('\\n')[0];
    if (txt.length < 15) return;
    const block = a.closest('div.g') || a.closest('div');
    const desc = block ? block.querySelector('div.VwiC3b') : null;
    const snippet = (desc ? desc.innerText : block.innerText).trim().slice(0, 300);
    items.push({title: txt.slice(0, 120), url: href, snippet: snippet});
  });
  return JSON.stringify(items.slice(0, 10));
})()"""
    script = (
        "import time\n"
        f"new_tab({url!r})\n"
        "wait_for_load()\n"
        "time.sleep(1)\n"
        f"js({js_consent!r})\n"
        "time.sleep(2)\n"
        f"raw = js({js_extract!r})\n"
        "if isinstance(raw, dict):\n"
        "    raw = raw.get('text') or raw.get('result') or raw.get('value') or ''\n"
        "print(raw if isinstance(raw, str) else str(raw or ''))\n"
    )
    env = os.environ.copy()
    if not env.get("BU_CDP_URL") and not env.get("BU_CDP_WS"):
        # Headed research Chrome from scripts/start_research_chrome.sh (not daily 9222).
        env.setdefault("BU_CDP_URL", "http://127.0.0.1:9224")
    try:
        proc = subprocess.run(
            [cli],
            input=script,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise BrowserResearchError(f"browser-harness timeout for {url}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[:400]
        raise BrowserResearchError(f"browser-harness exit {proc.returncode}: {err}")

    out = _normalize_harness_stdout(proc.stdout or "")
    if out in _EMPTY_MARKERS:
        return []
    try:
        items = json.loads(out)
    except json.JSONDecodeError:
        return []
    results: list[tuple[str, str, str]] = []
    skip_hosts = re.compile(
        r"^(https?://)?(www\.)?(google|scholar\.google|support\.google)\.",
        re.IGNORECASE,
    )
    for it in items[:limit]:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title") or "").strip()
        url_text = str(it.get("url") or "").strip()
        snippet = str(it.get("snippet") or "").strip()
        if url_text.startswith("http") and title and not skip_hosts.match(url_text):
            results.append((title, url_text, snippet))
    return results


def search_via_ddg_http(query: str, limit: int = 5) -> list[tuple[str, str, str]]:
    """DuckDuckGo HTML SERP via stdlib (no key). Returns (title, url, snippet)."""
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
    out: list[tuple[str, str, str]] = []
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
        out.append((title, url_text, ""))
    return out


def _default_serp(prefer_browser_harness: bool) -> SerpFn:
    def smart_serp(query: str, limit: int = 5) -> list[tuple[str, str, str]]:
        if prefer_browser_harness and _harness_cli() is not None:
            try:
                hits = search_via_browser_harness(query, limit)
                if hits:
                    return hits
            except BrowserResearchError:
                pass
        return search_via_ddg_http(query, limit)

    return smart_serp


def default_serp(prefer_browser_harness: bool = True) -> SerpFn:
    """Browser-harness SERP with an HTTP fallback; safe outside tests."""
    return _default_serp(prefer_browser_harness)


def _default_fetch(prefer_browser_harness: bool) -> FetchFn:
    def smart_fetch(url: str, max_chars: int = 12000) -> str:
        if prefer_browser_harness and _harness_cli() is not None:
            try:
                return fetch_via_browser_harness(url, max_chars)
            except BrowserResearchError:
                return fetch_via_http(url, max_chars)
        return fetch_via_http(url, max_chars)

    return smart_fetch


def default_fetch(prefer_browser_harness: bool = True) -> FetchFn:
    """Browser-harness fetch with an HTTP fallback; safe outside tests."""
    return _default_fetch(prefer_browser_harness)


@dataclass
class BrowserResearcher:
    fetch: FetchFn
    max_chars: int = 12000

    @classmethod
    def default(cls, *, prefer_browser_harness: bool = True) -> BrowserResearcher:
        return cls(fetch=default_fetch(prefer_browser_harness))

    def collect(
        self, slug: str, *, pages: tuple[str, ...] = ("overview", "discussion")
    ) -> BrowserNotes:
        urls = competition_page_urls(slug)
        notes = BrowserNotes(slug=slug)
        for name in pages:
            url = urls.get(name)
            if not url:
                notes.errors.append(f"unknown page: {name}")
                continue
            try:
                notes.pages[name] = self.fetch(url, self.max_chars)
            except Exception as exc:  # noqa: BLE001
                notes.errors.append(f"{name}: {exc}")
        return notes


def merge_browser_into_research_md(research_path: Path, notes: BrowserNotes) -> None:
    """Append or replace the Browser section in research.md."""
    merge_section_into_research_md(research_path, _BROWSER_MARKER, notes.to_markdown_section())
