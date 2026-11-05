#!/usr/bin/env python3
"""Reuse each immutable controller, replacing only future training invocation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import time
REPO=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm-root',type=Path,required=True);a=p.parse_args();root=a.arm_root.resolve()
    plan=json.loads((root/'plan.json').read_text());tail1='parent_root' in plan
    original=Path(plan['code_snapshot'])/'scripts/training'/('ngcm_tail1_controller.py' if tail1 else 'ngcm_2k_controller.py')
    spec=importlib.util.spec_from_file_location('qualified_controller',original);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_batch=module.batch;worker=REPO/'experimental/residualmamba_NGCM/dense_train.py'
    receipt=json.loads((root/'controller/dense-dispatch.json').read_text())
    for name,expected in receipt['files'].items():
        if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:raise ValueError('Dense dispatcher changed')
    def batch(plan,key,gpu,extra):
        args,script=old_batch(plan,key,gpu,extra)
        if key.startswith('train-'):
            command=[plan['python'],'-u',str(worker),'--arm-root',str(root),*extra]
            lines=script.splitlines()
            if not lines[-1].startswith('exec '):raise ValueError('Unexpected original script')
            lines[-1]='exec '+shlex.join(command);script='\n'.join(lines)+'\n'
        return args,script
    module.batch=batch
    try:module.watch(root)
    except Exception as error:
        module.write(root/'controller/status.json',{'phase':'failed','error':str(error),'heartbeat_unix':time.time()})
        raise

if __name__=='__main__':main()
