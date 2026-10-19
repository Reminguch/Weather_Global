"""Run the shared-start di16/di32 spatial-LR sweep using the batch-4 driver."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments import run_v24_res1_batch4_lr as driver

driver.EXPERIMENT = driver.WORKSPACE / 'artifacts/checkpoints/v24_Ilya/res1_batch4_slow_spatial_20260916'
driver.ARMS = [('joint', 1., 1.), ('spatial_5x_slower', .2, 1.), ('spatial_10x_slower', .1, 1.)]

if __name__ == '__main__':
    driver.main()
