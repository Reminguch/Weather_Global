import pytest

from src.models.neuralgcm_residual.paper_lifecycle import lock_run


def test_existing_run_requires_resume_and_checkpoint(tmp_path):
    output = tmp_path / 'run'
    lock = lock_run(output)
    with pytest.raises(FileExistsError):
        lock_run(output)
    with pytest.raises(FileNotFoundError):
        lock_run(output, resume=True)
    lock.close()


def test_second_writer_is_rejected_and_lock_releases(tmp_path):
    output = tmp_path / 'run'
    first = lock_run(output)
    (output / 'last.pkl').write_bytes(b'checkpoint existence only; validation is in the trainer')
    with pytest.raises(RuntimeError, match='Another worker'):
        lock_run(output, resume=True)
    first.close()
    second = lock_run(output, resume=True)
    second.close()
