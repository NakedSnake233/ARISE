"""Seeded compute witness for the existing CARE runtime; installs nothing."""
import importlib.metadata
import json
from pathlib import Path
import torch

spec=json.loads(Path('configs/care/runtime-spec.json').read_text())
for package,version in spec['packages'].items():
    actual=importlib.metadata.version(package)
    if actual!=version:raise RuntimeError(f'{package}: expected {version}, got {actual}')
torch.manual_seed(42)
assert torch.version.cuda==spec['torch_cuda_build']
x=torch.randn(8,8,device='cuda');y=x@x.T
assert torch.isfinite(y).all() and y.numel()==64
assert (Path(spec['weights']['qwen3'])/'model.safetensors.index.json').exists()
print('WITNESS',tuple(y.shape),torch.cuda.get_device_name(),float(y.trace()))
