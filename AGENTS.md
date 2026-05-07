# Codex Notes

For this repository, always run Python, pytest, training, preprocessing, and analysis commands inside the GraphCast environment:

```bash
source scripts/graphcast_env.sh
```

Use commands like:

```bash
bash -lc 'source scripts/graphcast_env.sh && pytest tests/test_graphcast_batching.py'
bash -lc 'source scripts/graphcast_env.sh && python scripts/analyze_models/unified_resolution_eval.py ...'
```

Do not use bare `/usr/bin/python`, `python`, or `pytest` for repository validation, because the system Python can mix incompatible user-site and system packages.

# Coding ethics

Don't implement the code, befory you sure I want it. If I ask with a question, it's not a call to action