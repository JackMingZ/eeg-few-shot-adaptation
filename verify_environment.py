"""Capture installed versions and a real CUDA operation, without installing."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch


if __name__=='__main__':
    result={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','scikit-learn','pandas','matplotlib','mne','moabb','joblib']},
            'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda}
    if torch.cuda.is_available():
        device=torch.cuda.get_device_properties(0)
        result.update(gpu=device.name,gpu_memory_bytes=device.total_memory)
        x=torch.ones((32,32),device='cuda');torch.cuda.synchronize()
        result['cuda_matmul_check']=bool(torch.all(x@x==32).item())
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10,check=True)
        result['nvidia_smi']=r.stdout.strip()
    except (OSError,subprocess.SubprocessError) as e: result['nvidia_smi_error']=str(e)
    out=Path(__file__).resolve().parent/'environment_verified.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
