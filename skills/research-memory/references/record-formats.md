# Research record formats

Use these outlines when creating records; omit fields that add no information.
Replace bracketed prompts with real evidence, `unknown`, or `not applicable`.
Do not create empty directions or development files to fill out the tree.
Dates use `YYYY-MM-DD` in the user's timezone; include a timezone with precise
timestamps. Keep names stable as conclusions evolve.

## Project index: research/README.md

```markdown
# Research

Updated: [date]

[Project scope and the immediate research priority.]

| ID | Direction / question | Category | Status | Current conclusion | Next step |
| --- | --- | --- | --- | --- | --- |
| r001 | [Short question](directions/r001-descriptive-question/overview.md) | [category] | [status] | [Bounded conclusion or no evidence yet] | [Concrete next action] |

## Shared development

[Links to research/devel notes, when any exist.]

## Earlier records

[Links to existing experiment notes; say whether historical coverage is partial.]
```

The overview is authoritative for a direction's current understanding; the index
is a short navigation summary. A lightweight proposed idea can be listed as plain
text until it warrants a direction folder. Do not add a link to a nonexistent
file or claim the index covers all prior work unless that was established.

## Direction: overview.md

```markdown
# r001 — [Descriptive research question]

Category: [one primary category; optional useful tags]
Status: [proposed / active / paused / closed / unknown]
Updated: [date]
Related: [links to parent, alternative, or successor directions, if relevant]

## Question and hypothesis

[What are we trying to learn? What would support or challenge the hypothesis?]

## Motivation and comparison

[Why try this? What baseline or alternative makes the test informative?]

## Current understanding

[Bounded conclusion, confidence/limitations, and links into experiments.md.
State when evidence is absent. Explain a paused or closed status.]

## Next decision

[Next concrete step and the evidence that would justify continuing, changing,
or stopping. Include unresolved questions that affect this decision.]

## Records

- [Attempts and evidence](experiments.md)
[Add a development-note link only when that file exists.]
```

## Attempt history: experiments.md

Give each attempt a stable explicit anchor. Keep its top-level execution and
outcome fields current. When correcting an outcome, preserve the previous value
and interpretation in the dated history, and explain why they changed; do not
leave a superseded outcome displayed as the current classification. Retain dated
observations and decisions. Append new attempts and corrections; do not silently
revise away an earlier conclusion. A summary
table is optional when the file grows. Split into linked attempt files only
when the ledger becomes unwieldy, preserving old links or anchor redirects.

```markdown
# Attempts and evidence

<a id="e001"></a>
## e001 — [Short description of the test]

Event date: [date or unknown]
Recorded: [date; indicate reconstructed history when applicable]
Execution: [planned / running / completed / failed / interrupted / unknown]
Outcome: [pending / supported / unsupported / inconclusive / execution-failed]

**Question and change:** [Hypothesis and the difference from the comparison.]

**Setup:** [Baseline, dataset/split or sample, meaningful parameters, seed(s),
metric definition/units, and evaluation conditions as applicable.]

**Provenance:** [Commit plus relevant dirty changes, or immutable snapshot;
linked configuration; host/environment; run or scheduler ID if relevant.]

**Evidence:** [Links to inspected reports/configs/logs and host-qualified
external artifacts. Identify user-reported, missing, or inaccessible evidence.]

**Observation:** [Measured result or exact execution failure, including limits.]

**Interpretation:** [What this says about the hypothesis, and what it cannot say.]

**Decision / next action:** [What follows and why.]

### [date] — [Result update or correction, when needed]

[What changed, supporting evidence, and which earlier statement it supersedes.]
```

For an experiment plan, keep the observation as `pending`. A launched job remains
`running` until there is evidence of its later state. For a decision without a
run, add a dated decision entry rather than fabricating an experiment.

For backfills, retain the source's historical assertions separately from today's
interpretation. If the exact code, date, baseline, or final state cannot be
recovered, say so. A file modification time is not an experiment date.

## Development handoff: devel.md or research/devel/component.md

```markdown
# Development — [Direction or shared component]

Updated: [date]
Related directions: [links if this is a shared component]

## Current implementation

[What exists and where; relevant branch/commit and uncommitted work.
Distinguish implemented from tested behavior.]

## Validation and blockers

[Checks actually run and their outcomes; known failures and unfinished work.]

## Resume

1. [Next concrete action, required host, prerequisites, and expected outcome.]
2. [Following action only if useful.]

[Commands, working directory, environment, and artifact locators when needed.
Label commands as already run or proposed; preserve existing execution scope.]

## Implementation decisions

[Only consequential choices, their reasons, and links to supporting evidence.]
```

Keep the current handoff concise. If an implementation issue changes scientific
interpretation, record that consequence in the experiment history and overview.
Use a single shared development note for shared infrastructure and link to it.
