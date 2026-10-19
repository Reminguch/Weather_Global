# Install and carry research memory between machines

The skill folder is self-contained. Keep a canonical copy in version-controlled
source, then copy it to each machine's user-level skill directory. Each project
owns its own `research/` records. Installing the skill does not import old
experiments, synchronize machines, or run a background service.

## Della and the existing Codex skills layout

From the source checkout:

```bash
python3 skills/research-memory/scripts/install.py
```

The default destination is `${CODEX_HOME:-$HOME/.codex}/skills/research-memory`,
matching the skill-creator convention and the existing Della installation.
The installer uses Python's standard library and prints the exact destination.
It copies the skill, so using it does not depend on the source checkout remaining
mounted. Repeating an identical installation is harmless; differing existing
content is left untouched and reported for review.

## Local machine

Transfer the `research-memory/` folder, or bring `skills/research-memory/` across
with your normal repository synchronization. From a local source checkout:

```bash
python3 skills/research-memory/scripts/install.py --skills-dir "$HOME/.agents/skills"
```

Current [OpenAI skill documentation](https://learn.chatgpt.com/docs/build-skills)
lists `~/.agents/skills` as the user-wide location. Use the location your client
loads; for a client using the same layout as Della, omit `--skills-dir`. Install
in one loaded location per client to avoid duplicate skill entries. A transferred
standalone folder works identically: run its `scripts/install.py` with the same
option. The script does not need access to Della.

For a packaged archive, extract it into a temporary or source directory first:

```bash
tar -xzf research-memory.tar.gz
python3 research-memory/scripts/install.py --skills-dir "$HOME/.agents/skills"
```

Restart Codex if the new skill does not appear. Explicit invocation is
`$research-memory`; automatic selection for relevant research work is enabled.
Automatic selection is contextual, not a guaranteed hook on every message.

## Updates and synchronization

Edit the canonical skill source. Review differences before replacing an existing
installation, preserve any machine-specific edits, then copy the reviewed version
to each machine. The installer intentionally does not overwrite different content.
Do not edit parallel copies and assume changes propagate.

Use the project's normal Git workflow for research notes. No automatic commits,
pushes, pulls, or artifact transfers are performed. Record external artifacts
with their host and path; a Della checkpoint need not be copied locally to read
the research history. Check actual file content before claiming both copies are
current.
