# Independent v2 tail1 launch evidence

The independent v2 ar_tail_k=1 diagnostic uses fresh learned state and the
same train-only normalization as the running v2 tail20 arm. Twelve focused
CPU tests passed, including configuration, memory/cursor continuation, loader
selection and checkpoint labels. An actual full 24-step tail1 data chunk
passed finite-value and five-field target checks before submission.

The unchanged core reuses the v2 H200 qualification under ../20261105_v2/.
Native tail1 rollout/cache evidence is authenticated from the prior tail1
producer. This is not a claim of a newly executed tail1-specific GPU restart
experiment. The ongoing K=20 run retains its original source snapshot.

Job 16945328 launches the fresh K=1 run with a 2,000-update endpoint, 100-step
checkpoint saves and automatic full common-tail20 evaluation. The controller
resumes across one-hour H200 gpu-test allocations. See launch.json for the
run identity. Live training and evaluation outputs remain in its scratch
run root; this launch receipt is not a completed training result.
