# Repository Cleanup Report

## Baseline and non-interference audit

The audit ran in the clean validation worktree on branch
`supervisor-live-certification`. The original dirty checkout was not touched.

```text
HEAD: 5109af1afc63a762b41ef4e4b2844d557528cdd5
origin/main: 77ddbfc33adbbaf3ff11677e88bec12feeacc754
working tree: clean before cleanup
compile: pass
non-integration suite: 605 passed, 1 deselected
```

Reference searches proved that `src/`, tests, packaging, and the supervisor
still use the runtime contracts. The hidden tool directories and root task
reports had no such references. Checked-in memory was used by runtime paths but
contained live contest state, so it was templated rather than removed from the
runtime contract. The RSNA contract/workspace is still referenced by existing
pipeline, image-template, package, orchestration, and integration fixtures, so
it was retained as an optional compatibility example.

## Ticket 1 — Stale developer artifacts

### Existing purpose

`.grok/` contained local tool configuration, `.scratch/` contained old
investigation notes, `.superpowers/` contained dated reports, and root `tasks/`
and `task-*-report.md` files duplicated implementation history.

### Reference search and decision

`rg` found no runtime, test, packaging, CI, or dynamic-loader references. The
only `.grok` match was the stale MCP document, which was rewritten first.

Deleted paths:

```text
.grok/config.toml
.grok/skills/kaggle-deep-research/SKILL.md
.scratch/memory-and-loops/issues/01-waku-memory.md
.scratch/memory-and-loops/issues/02-loop-tool-calls.md
.scratch/memory-and-loops/map.md
.superpowers/sdd/2026-08-18-fix-kernel-train-submit/task-2-report.md
.superpowers/sdd/2026-08-18-fix-kernel-train-submit/task-3-report.md
.superpowers/sdd/2026-08-18-fix-kernel-train-submit/task-4-report.md
.superpowers/sdd/2026-08-18-fix-kernel-train-submit/task-5-report.md
.superpowers/sdd/2026-08-18-fix-kernel-train-submit/task-6-report.md
task-4-report.md, task-5-report.md, task-6-report.md, task-7-report.md
tasks/plan.md, tasks/todo.md
```

The three local tool directories are now ignored. Focused verification passed:
`14 passed` and `git diff --check` passed.

### Code review findings and fixes

The MCP document still named `.grok/` after the initial deletion plan. It was
changed to client-side MCP configuration before the ticket was committed.

## Ticket 2 — Runtime memory cleanup

### Existing behavior and problem

Runtime code writes to `memory/`, but the tracked files contained live contest
state, scores, research cards, and an active phase. Tests copied those files.

### Implementation

Added sanitized templates:

```text
memory/templates/MEMORY.md
memory/templates/COMPETITION.md
memory/templates/state.md
memory/templates/research.md
memory/templates/heal.md
```

Removed tracked live markdown, experiment context, and seven research cards.
The runtime paths and `KAGGLE_AGENT_STATE_ROOT` behavior are unchanged;
fixtures now copy templates. Generated state is ignored by Git.

### Tests and review

Focused tests: `30 passed`. Full suite after this ticket: `605 passed,
1 deselected`. Review finding fixed: the ingest test no longer reads live
RSNA ground truth from the repository and instead builds a template-backed
fixture.

## Ticket 3 — Generic configuration and initialization

### Root cause/design

The public settings selected `rsna_knee` implicitly and the shell scaffold
copied its pipeline into every new competition. The existing config template
made a small nullable-default plus CLI scaffold the lowest-risk fix.

### Implementation and files

- Set `default_competition: null`; commands require `--competition` when unset.
- Added `kaggle-agent init --competition <id> [--slug <slug>]`.
- Initialization refuses existing config, workspace, or runtime-memory files.
- The generic scaffold uses only `id` and `target` placeholders.
- `scripts/new_competition.sh` now wraps the CLI and never copies RSNA code.
- Added `examples/competition/` with a synthetic no-network pipeline.
- Removed legacy provider blocks from user-facing settings; explicit
  compatibility client code remains where tests prove it is used.
- Removed the account-specific RSNA dataset rewrite from shared pin healing.
- Changed generic CV, smoke, notebook, state, and snapshot fallbacks to neutral
  identifiers.

### Tests and review

Focused config/init tests: `49 passed`. Image/code/orchestrator/package/CV/pin
tests: `103 passed`. Review findings fixed: Telegram startup can use an
explicit saved state competition in minimal fixtures, and the supervisor
returns `NO_COMPETITION` instead of launching an empty request.

## Ticket 4 — Generic shared image/CV helpers

### Existing behavior and problem

The image helper encoded a competition ID, slug, identifier columns, and a
12-label medical schema in shared code. The current RSNA example depends on
the helper, so deletion would have changed behavior.

### Implementation

The image contract now supplies competition ID, slug, identifier, and series
columns through parameters. The template version and public API are generic;
CV reads `ID_COLUMN` from the competition ranker. Orchestration and fixtures
use the generic image-template version. The retained RSNA workspace remains an
explicit example artifact rather than a shared default.

Focused tests: `103 passed`; full suite: `609 passed, 1 deselected`.

### Review findings and fixes

The first implementation added workspace-derived parameters that changed the
existing contract shape. Those implicit additions were removed; callers may
provide them explicitly and the generic defaults remain stable. The account-
specific pin rewrite was also removed because it was not a framework policy.

## Ticket 5 — Public documentation and history cleanup

### Implementation

Rewrote `README.md` and `AGENTS.md` for external contributors. Added:

```text
docs/architecture.md
docs/supervisor.md
docs/configuration.md
docs/competitions.md
docs/operations.md
docs/safety.md
```

Updated `.env.example`, cron documentation, and MCP documentation. Removed
dated architecture reports, ticket reports, reviews, and slice specs from the
public tree; Git history remains the historical record.

### Tests and review

`git diff --check`: pass. Public docs contain no local absolute paths or stale
tool-directory references. README answers install, credentials, initialization,
dry run, supervisor modes, state, safety, and test questions without requiring
source inspection.

## Final verification

```text
uv run python -m compileall -q src examples
PASS

uv run pytest -q -m "not integration"
609 passed, 1 deselected in 21.56s

git diff --check
PASS
```

No dependency or lockfile changes were made. No push or merge was performed.
`AUTO_SAFE` remains disabled by default.

## Final tree

```text
config/competitions/        contract template and optional fixture config
competitions/                retained optional compatibility workspace
examples/competition/        synthetic generic example
memory/templates/            sanitized runtime starters
docs/                        current product and safety documentation
scripts/                     public operational wrappers
src/kaggle_agent/            shared runtime and supervisor
tests/                       unit and local integration tests
```

## Remaining risk

The RSNA contract and workspace were deliberately retained because reference
searches proved current tests and the optional image/pipeline path still use
them. They are no longer the framework default, are not copied by
initialization, and shared image/CV helpers no longer encode their identifiers.
A later dedicated adapter-extraction ticket can move this fixture after its
tests and packaging contract are migrated.

## Final verdict

```text
Repository generic: yes for defaults, scaffolding, runtime state, shared helpers, and docs
Competition-specific core assumptions: 0 in shared config/default/image/CV paths
Developer-only artifacts remaining: 0 tracked
Public setup reproducible: yes, via uv sync + kaggle-agent init
AUTO_SAFE enabled by default: no
```
