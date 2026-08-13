"""Browser-harness submit fallback when the Kaggle API submit fails.

Uses a logged-in local Chrome session (browser-harness CDP). Not the primary path.
Primary remains API (file or notebook). Inject ``run_fn`` in tests.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.kaggle_api.models import SubmitResult

# run_fn(competition, csv, kernel_ref, message, mode) -> SubmitResult
BrowserSubmitFn = Callable[..., SubmitResult]


@dataclass(frozen=True)
class BrowserSubmitRequest:
    competition: str
    message: str
    mode: str = "file"
    csv_path: Path | None = None
    kernel_ref: str | None = None
    dry_run: bool = False


def submit_via_browser(
    req: BrowserSubmitRequest,
    *,
    run_fn: BrowserSubmitFn | None = None,
    timeout_sec: int = 180,
) -> SubmitResult:
    """Attempt a UI submit. dry_run never opens a browser."""
    if req.dry_run:
        target = req.kernel_ref or (req.csv_path.name if req.csv_path else "none")
        return SubmitResult(
            dry_run=True,
            message=f"dry_run: would browser-submit {req.mode} {target} → {req.competition}",
            success=True,
        )
    if run_fn is not None:
        return run_fn(req)
    return _run_harness(req, timeout_sec=timeout_sec)


def _run_harness(req: BrowserSubmitRequest, *, timeout_sec: int) -> SubmitResult:
    if not shutil.which("browser-harness"):
        return SubmitResult(
            dry_run=False,
            message="browser fallback: browser-harness not on PATH",
            success=False,
        )

    mode = (req.mode or "file").lower()
    csv = str(req.csv_path) if req.csv_path else ""
    kernel = (req.kernel_ref or "").strip()
    slug = req.competition
    msg = req.message.replace("\\", "\\\\").replace("'", "\\'")

    if mode == "notebook" and kernel and kernel not in {"none", ""}:
        url = f"https://www.kaggle.com/code/{kernel}"
        script = _script_notebook_kernel(url, slug, msg)
    elif csv and Path(csv).is_file():
        url = f"https://www.kaggle.com/competitions/{slug}/submit"
        script = _script_file_submit(url, csv, msg)
    else:
        return SubmitResult(
            dry_run=False,
            message="browser fallback: need kernel_ref (notebook) or existing csv_path",
            success=False,
        )

    try:
        proc = subprocess.run(
            ["browser-harness"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SubmitResult(
            dry_run=False,
            message=f"browser fallback: timeout after {timeout_sec}s",
            success=False,
        )

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    # Last non-empty line should be RESULT JSON from the script
    result_line = ""
    for line in reversed(out.splitlines()):
        if line.strip().startswith("{"):
            result_line = line.strip()
            break
    if result_line:
        try:
            data = json.loads(result_line)
            ok = bool(data.get("ok"))
            detail = str(data.get("message") or data)
            return SubmitResult(
                dry_run=False,
                message=f"browser fallback: {detail}",
                success=ok,
                raw_status=detail,
            )
        except json.JSONDecodeError:
            pass

    if proc.returncode != 0:
        return SubmitResult(
            dry_run=False,
            message=f"browser fallback: harness exit {proc.returncode}: {(err or out)[:400]}",
            success=False,
        )
    return SubmitResult(
        dry_run=False,
        message=f"browser fallback: unclear result: {(out or err)[:400]}",
        success=False,
    )


def _script_file_submit(url: str, csv_path: str, message: str) -> str:
    """Open competition submit page, attach CSV, try to submit."""
    return f"""
import json
import time

new_tab({url!r})
wait_for_load()
time.sleep(2)

# Login wall?
text = js('''(() => (document.body && document.body.innerText || '').slice(0, 2000))()''') or ''
if isinstance(text, dict):
    text = str(text.get('result') or text.get('value') or text)
low = str(text).lower()
if 'sign in' in low and 'submit' not in low:
    print(json.dumps({{"ok": False, "message": "login required — open Kaggle in Chrome and sign in"}}))
    raise SystemExit(0)

csv_path = {csv_path!r}
msg = {message!r}

# Prefer CDP file input when present
set_ok = False
try:
    doc = cdp("DOM.getDocument", depth=1)
    root_id = doc.get("root", {{}}).get("nodeId")
    q = cdp("DOM.querySelector", nodeId=root_id, selector='input[type="file"]')
    node_id = q.get("nodeId")
    if node_id:
        cdp("DOM.setFileInputFiles", files=[csv_path], nodeId=node_id)
        set_ok = True
except Exception as exc:
    set_ok = False
    file_err = str(exc)
else:
    file_err = ""

if not set_ok:
    print(json.dumps({{
        "ok": False,
        "message": "no file input (notebooks-only UI?) " + file_err[:120],
    }}))
    raise SystemExit(0)

# Description / message if present
js('''(() => {{
  const ta = document.querySelector('textarea');
  if (ta) {{ ta.focus(); ta.value = {message!r}; ta.dispatchEvent(new Event('input', {{bubbles: true}})); }}
  return true;
}})()''')

# Click a primary Submit-like button
clicked = js('''(() => {{
  const buttons = Array.from(document.querySelectorAll('button, a[role="button"], input[type="submit"]'));
  const want = buttons.find(b => /submit/i.test((b.innerText || b.value || '').trim()));
  if (!want) return 'no-submit-button';
  want.click();
  return 'clicked:' + (want.innerText || want.value || '').trim().slice(0, 40);
}})()''')

time.sleep(4)
body = js('''(() => (document.body && document.body.innerText || '').slice(0, 1500))()''') or ''
if isinstance(body, dict):
    body = str(body.get('result') or body.get('value') or body)
body_s = str(body)
low = body_s.lower()
ok = any(x in low for x in ('submission', 'submitted', 'success', 'queued', 'scoring'))
if 'error' in low and 'success' not in low:
    ok = False
print(json.dumps({{
    "ok": ok,
    "message": f"file UI set={{set_ok}} click={{clicked}} body={{body_s[:200]!r}}",
}}))
"""


def _script_notebook_kernel(url: str, competition: str, message: str) -> str:
    """Open kernel page and try Submit-to-competition UI controls."""
    return f"""
import json
import time

new_tab({url!r})
wait_for_load()
time.sleep(3)

text = js('''(() => (document.body && document.body.innerText || '').slice(0, 2500))()''') or ''
if isinstance(text, dict):
    text = str(text.get('result') or text.get('value') or text)
low = str(text).lower()
if 'sign in' in low and 'edit' not in low:
    print(json.dumps({{"ok": False, "message": "login required on kernel page"}}))
    raise SystemExit(0)

# Open submit / competition menu if present
step = js('''(() => {{
  const nodes = Array.from(document.querySelectorAll('button, a, [role="menuitem"], span'));
  const labels = nodes.map(n => (n.innerText || '').trim()).filter(Boolean);
  const hit = nodes.find(n => /submit\\s*to\\s*competition|submit\\s*to\\s*{competition}|submit/i.test((n.innerText || '').trim()));
  if (hit) {{ hit.click(); return 'clicked:' + (hit.innerText || '').trim().slice(0, 50); }}
  return 'no-submit-control labels=' + labels.slice(0, 12).join('|');
}})()''')

time.sleep(2)
# Confirm dialog / second submit
step2 = js('''(() => {{
  const nodes = Array.from(document.querySelectorAll('button, a, [role="menuitem"]'));
  const hit = nodes.find(n => /submit|confirm|ok/i.test((n.innerText || '').trim()) && !/cancel/i.test((n.innerText || '').trim()));
  if (hit) {{ hit.click(); return 'confirm:' + (hit.innerText || '').trim().slice(0, 40); }}
  return 'no-confirm';
}})()''')

time.sleep(4)
body = js('''(() => (document.body && document.body.innerText || '').slice(0, 1500))()''') or ''
if isinstance(body, dict):
    body = str(body.get('result') or body.get('value') or body)
body_s = str(body)
low = body_s.lower()
ok = any(
    x in low
    for x in (
        'successfully submitted',
        'submission accepted',
        'your submission',
        'submitted to competition',
    )
)
note = f"step={{step}} step2={{step2}} body={{body_s[:180]!r}}"
print(json.dumps({{"ok": ok, "message": note}}))
"""
