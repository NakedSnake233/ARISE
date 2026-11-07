"""Documented, repeatable guarded witnesses for the official CPU runtimes."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
from .common import ROOT
from .disk_guard_local import DiskGuard,guarded_command


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();key="rcaeval_doc_"+out.name;log=ROOT/"logs"/(key+".log")
    if out.exists() or log.exists():raise FileExistsError("use a new witness directory")
    guard=DiskGuard(ROOT);env={**os.environ,**guard.reserve(key,out,log),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONHASHSEED":"42"}
    try:
        with log.open("x") as stream:
            for method in ("BARO","CIRCA","RCD"):
                runtime=ROOT/"cache/rcaeval_runtime"/("rcd" if method=="RCD" else "default")/"bin/python"
                cmd=["/usr/bin/timeout","120s",str(runtime),"-m","exp_revision.rcaeval_worker","--method",method,"--witness","--output",str(out/(method+".json"))]
                proc=subprocess.Popen(guarded_command(ROOT,key,cmd),cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                guard.attach(key,proc.pid); code=proc.wait()
                print(method,"exit",code,flush=True)
                if code:raise RuntimeError("algorithm witness failed: "+method)
    finally:guard.release(key)


if __name__=="__main__":main()
