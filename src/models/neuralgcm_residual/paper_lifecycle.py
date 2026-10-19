"""Local run-directory locking for the imported K2/K20 entry points.

The upstream reports reference this helper but do not contain its source.
This implementation protects fresh runs and requires a checkpoint for resume.
"""
import fcntl
from pathlib import Path


def lock_run(output, *, resume=False):
    output = Path(output)
    if resume:
        if not (output / 'last.pkl').is_file():
            raise FileNotFoundError(f'Resume requires {output / "last.pkl"}')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = (output / '.run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'Another worker owns {output}') from None
    return lock  # The caller retains this handle for the worker lifetime.
