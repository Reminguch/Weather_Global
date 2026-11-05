#!/usr/bin/env python3
"""Two H200 slots: qualified v2 training and dense fixed-set evaluation."""
import argparse
from pathlib import Path
import ngcm_2k_controller as controller


def pending_checkpoints(root):
    root=Path(root)
    return [p for p in sorted((root/'checkpoints').glob('step_*.pkl'))
            if p.with_suffix('.json').exists() and not
            (root/'eval/tail20'/p.stem/'summary.json').exists()]


def next_actions(root):
    root=Path(root)
    if (root/'controller/user-stop.json').exists():
        raise SystemExit('Stopped by user')
    actions=[]
    if controller.latest_step(root)<2000:actions.append(('train-2000',True,['--endpoint','2000']))
    if pending_checkpoints(root):actions.append(('validate-dense',True,[]))
    if not actions and not (root/'report/COMPLETE.json').exists():actions.append(('report',False,[]))
    return actions


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    controller.WORKER=controller.REPO/'experimental/residualmamba_NGCM/v2_diagnostic.py'
    original_batch=controller.batch
    def batch(plan,key,gpu,extra):
        args,script=original_batch(plan,key,gpu,extra)
        args=[x.replace('--job-name=ngcm2k-','--job-name=ngcm-v2-') for x in args]
        return args,script
    controller.batch=batch;controller.next_actions=next_actions
    try:controller.watch(root)
    except Exception as error:
        controller.write(root/'controller/status.json',{'phase':'failed','error':str(error)})
        raise


if __name__=='__main__':main()
