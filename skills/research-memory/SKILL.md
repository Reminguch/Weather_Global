---
name: research-memory
description: "Maintain project-local research memory in research/: classify directions, record hypotheses and experiment evidence, preserve unsuccessful attempts, and keep development handoffs. Use during substantive research planning, experimental implementation, evaluation, changes of direction, and requests to remember or resume research across sessions or machines. Update at meaningful milestones by default. Do not use for unrelated coding, casual questions, or discussion-only requests that do not authorize saved notes."
---

# Research Memory

Keep enough evidence and reasoning that a future session can answer: what did we
try, why, what happened, what do we currently believe, and how do we resume?
This is a portable workflow; store project facts in the project, not in this skill.

## Start with the existing record

- Locate the active project root and read its applicable instructions. Use the
  repository root unless the project has an established narrower research root.
- Read `research/README.md`, then only the directions relevant to the task.
  Search existing notes before creating a direction. A direction is a research
  question or hypothesis; a run, seed, parameter change, or retry is an attempt
  within it. Split only when the question changes materially.
- If adopting this layout, inspect existing experiment indexes and development
  notes. Link them as sources; preserve their paths and content. Do not turn
  adopting the skill into an unrequested historical migration.
- Use the structure below for new records. Adapt existing compatible conventions
  rather than creating a second source of truth. For new files or historical
  backfills, read [record formats](references/record-formats.md).

```text
research/
  README.md                         # project index and current priorities
  directions/
    r001-descriptive-question/
      overview.md                   # hypothesis and current understanding
      experiments.md                # dated attempts, evidence, corrections
      devel.md                      # optional implementation handoff
  devel/
    descriptive-component.md        # optional shared infrastructure notes
```

## Update during research work

Within authorized research work, keep notes current without asking for a separate
documentation checkpoint. Update when a hypothesis or experiment is adopted,
an implementation decision affects interpretation or reproducibility, results
arrive, a blocker changes the plan, or a direction is paused or closed. Before
handing work back, record any material developments not yet captured. This is
in-session behavior, not a background watcher or scheduled automation.

Respect discussion-only, read-only, and no-documentation requests. Ordinary
refactors, formatting, and unrelated coding do not require research entries.
If there is no material new information, make no documentation change.

1. Reuse the matching direction. For a new substantive direction, choose a stable
   `rNNN-short-kebab-case` name based on the question, not its current status,
   machine, or latest result. Allocate the next unused project ID; do not reuse
   retired IDs. Use direction-local `e001`, `e002`, etc. for attempts. Link related
   directions instead of duplicating evidence.
2. Record the attempt or decision with its date, rationale, relevant baseline or
   comparison, evidence, interpretation, and next action. Separate what was
   observed from what is inferred. Preserve negative and inconclusive findings.
3. Update `overview.md` with the current conclusion, its limits, supporting
   attempt links, and the next decision. Keep the history in `experiments.md`;
   add dated corrections or superseding entries when interpretations change.
4. Update `devel.md` only if implementation state matters for resumption. Keep
   it current and actionable; research conclusions belong in the overview.
   Put infrastructure shared by several directions in `research/devel/`.
5. Refresh the index row when classification, status, conclusion, or next step
   changes. Check local links and consistency with the detailed record. Briefly
   tell the user which research notes changed.

## Classification and evidence

Keep the taxonomy small and project-appropriate. Start with categories such as
`architecture`, `training`, `data`, `evaluation`, and `theory`; adapt them to the
discipline. A category describes the kind of work, not whether it succeeded.

- Direction status: `proposed`, `active`, `paused`, `closed`; record a reason
  when pausing or closing. Use `unknown` for unverified historical state.
- Attempt execution: `planned`, `running`, `completed`, `failed`, `interrupted`,
  or `unknown`.
- Attempt outcome: `pending`, `supported`, `unsupported`, `inconclusive`, or
  `execution-failed`. Support is always relative to the stated hypothesis and
  tested conditions. `execution-failed` means no scientific conclusion, not
  evidence against the hypothesis. If a failed run yielded useful partial
  measurements, describe their limits explicitly.

Never infer completion or success from a submission command, filenames, a plot
alone, or a successful process exit. Tie claims to inspected evidence or label
them as user-reported. Include metric units, evaluation conditions, baseline,
and sample size or seeds when these affect the comparison. Mark missing facts
as unknown rather than inventing precision. Separate an event's date from the
date its history was reconstructed.

## Work across projects and machines

- Express source/config locations relative to the project root. Make Markdown
  links relative to the containing document so they actually resolve.
- For external data, logs, checkpoints, or environments, record the host label
  and location, for example `della:/scratch/...`. These are locators, not portable
  paths. State which evidence was accessible and which was not checked.
- Record the code revision and relevant uncommitted changes, or an immutable
  code snapshot when available. Do not present a commit as identifying a dirty
  experiment exactly. Record run/job IDs when useful; record environment and
  commands only to the depth needed to reproduce or resume.
- Keep large datasets, checkpoints, and raw logs outside the notes; link them.
  Do not copy credentials or sensitive environment values into documentation.
- Before editing, reread affected files and preserve concurrent changes. Recheck
  new IDs against the index and folders. If merged work collides on an ID, keep
  both records, assign an unused ID to one, and update its references explicitly.
- Git can carry the notes and skill source between machines, but this skill
  does not automatically commit, push, pull, or contact another machine. Do not
  claim local and Della copies are synchronized without checking them.
- If a host is unavailable, document that limit and finish work supported by
  accessible evidence. Recording a command does not authorize executing it.

For installing this same skill on another machine, use
[installation and portability](references/installation.md). No project-specific
paths, cluster tools, Python packages, or external services are needed to use
the documentation workflow.
