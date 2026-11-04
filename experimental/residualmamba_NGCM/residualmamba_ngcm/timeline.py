"""Pure time bookkeeping. Every tail window contains baseline predictions."""
import numpy as np
from .config import FIELDS
from src.models.neuralgcm_residual.data import STEP, split_for
from src.models.mamba.v24_Ilya.training.data import TrainingCursor, build_segments, advance_cursor


def split_segments(times, split):
    times = np.asarray(times, dtype='datetime64[h]')
    anchors = [i for i in range(4,len(times)-1)
               if all(split_for(t) == split for t in times[i-4:i+2])
               and np.all(np.diff(times[i-4:i+2]) == STEP)]
    # build_segments assumes contiguous indices: split at gaps before reusing it.
    runs = np.split(np.asarray(anchors,dtype=np.int64), np.flatnonzero(np.diff(anchors) != 1)+1)
    segments = tuple(segment for run in runs for segment in build_segments(run,96))
    return segments, len(anchors) - 96 * len(segments)


def origin_index(a0, i):
    if not 0 <= i < 24:
        raise ValueError('Chunk step must be 0..23')
    return a0 + min(i,3)


def step_times(times, a0, i):
    return np.asarray(times[a0+i-1:a0+i+2],dtype='datetime64[h]')


def window(truth_prefix, baseline, i):
    """truth_prefix = x[a0-1:a0+4]; targets are intentionally not accepted."""
    def frame(j):
        return truth_prefix[j] if j <= 4 else baseline[j-2]
    return frame(i), frame(i+1)


def lead_hours(i):
    return 6 if i < 4 else (i-2)*6


def stack_weather(frame):
    return np.stack([frame[name] for name in FIELDS]).astype(np.float32)


def evaluation_origins(times, year, warm=False, steps=40):
    times = np.asarray(times,dtype='datetime64[h]')
    available = set(times)
    history = 7 if warm else 4
    result = []
    for t in times:
        if int(str(t)[:4]) != year or int(t.astype(np.int64)) % 12:
            continue
        required = [t+k*STEP for k in range(-history,steps+1)]
        if all(r in available and int(str(r)[:4]) == year for r in required):
            result.append(t)
    return np.asarray(result,dtype='datetime64[h]')
