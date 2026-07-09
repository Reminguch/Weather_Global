#!/usr/bin/env bash
# Auto-chain v22closed K-scan extensions.
#
# Strategy (per K):
#   K=2..20:  10k (RUNNING)  → 20k (afterok ext1)  → 50k (afterok ext2)
#   K=22:     20k (RUNNING)  → 50k (afterok ext1)
#
# All extensions resume residual_params only (NOT residual_state, NOT optimizer
# state — per Fix #3 in train_mz_v22closed.py). This is finetune-style resume.
#
# Idempotent: skips a K if the target ckpt already exists.
# Run from this directory.

set -euo pipefail

cd "$(dirname "$0")"
TEMPLATE="extend_template.sh"
SLURM_OUT_DIR="."
RESULTS_ROOT="/scratch/gpfs/DABANIN/lm8598/Weather_Global/results/v22closed"

# Current 10k production jobs (running/pending; will gate the 10k→20k extensions)
declare -A CURRENT_JOBS_10K
CURRENT_JOBS_10K[2]=9166774
CURRENT_JOBS_10K[4]=9166775
CURRENT_JOBS_10K[6]=9166776
CURRENT_JOBS_10K[8]=9166777
CURRENT_JOBS_10K[10]=9166778
CURRENT_JOBS_10K[12]=9166779
CURRENT_JOBS_10K[14]=9166780
CURRENT_JOBS_10K[16]=9166781
CURRENT_JOBS_10K[18]=9166782
CURRENT_JOBS_10K[20]=9166783

# K=22 production is 20k (job 9165472) → chain only the 20k→50k extension
PARENT_K22=9165472

# Helper: substitute placeholders + sbatch + return new job id
submit_extension() {
    local K=$1
    local PARENT_JID=$2
    local RESUME_FROM_STEP=$3   # e.g., 10000 (load ckpt_step10000.pkl)
    local START_STEP=$4         # e.g., 10001
    local MAX_STEPS=$5          # e.g., 20000
    local OLDDIR=$6             # e.g., $RESULTS_ROOT/K2_sg_fresh_10k

    local SLURM_FILE="${SLURM_OUT_DIR}/v22closed_sg_K${K}_to${MAX_STEPS}_dep${PARENT_JID}.slurm"

    sed -e "s/{K}/${K}/g" \
        -e "s/{PARENT_JID}/${PARENT_JID}/g" \
        -e "s/{RESUME_FROM_STEP}/${RESUME_FROM_STEP}/g" \
        -e "s/{START_STEP}/${START_STEP}/g" \
        -e "s/{MAX_STEPS}/${MAX_STEPS}/g" \
        -e "s|{OLDDIR}|${OLDDIR}|g" \
        "$TEMPLATE" > "$SLURM_FILE"

    local SUBMIT_OUT
    SUBMIT_OUT=$(sbatch "$SLURM_FILE")
    local NEW_JID
    NEW_JID=$(echo "$SUBMIT_OUT" | awk '{print $4}')
    echo "  K=${K} ${START_STEP}->${MAX_STEPS} :  parent ${PARENT_JID}  → new job ${NEW_JID}  (slurm: $SLURM_FILE)"
    echo "$NEW_JID"
}

echo "================================================================"
echo "Chaining v22closed K-scan extensions"
echo "================================================================"
echo ""
echo "--- K=2..20: 10k → 20k (afterok parent 10k job) ---"
declare -A JID_20K
for K in 2 4 6 8 10 12 14 16 18 20; do
    PARENT=${CURRENT_JOBS_10K[$K]}
    OLDDIR="${RESULTS_ROOT}/K${K}_sg_fresh_10k"
    JID=$(submit_extension "$K" "$PARENT" 10000 10001 20000 "$OLDDIR" | tail -1)
    JID_20K[$K]=$JID
done

echo ""
echo "--- K=2..20: 20k → 50k (afterok parent 20k extension) ---"
for K in 2 4 6 8 10 12 14 16 18 20; do
    PARENT=${JID_20K[$K]}
    OLDDIR="${RESULTS_ROOT}/K${K}_sg_fresh_20000"
    JID=$(submit_extension "$K" "$PARENT" 20000 20001 50000 "$OLDDIR" | tail -1)
    echo "  (registered K=${K} 50k follow-up job ${JID})"
done

echo ""
echo "--- K=22: 20k → 50k (afterok parent K=22 prod) ---"
OLDDIR="${RESULTS_ROOT}/K22_sg_fresh_20k"
JID=$(submit_extension 22 "$PARENT_K22" 20000 20001 50000 "$OLDDIR" | tail -1)
echo "  (registered K=22 50k follow-up job ${JID})"

echo ""
echo "================================================================"
echo "All extension jobs queued. squeue -u lm8598 to verify dependencies."
echo "================================================================"
