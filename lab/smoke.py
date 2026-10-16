"""GPU check for the lab container: python lab/smoke.py"""
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
assert torch.cuda.get_device_capability() == (12, 1), "expected GB10 (sm_121)"
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul", (a @ a).float().abs().mean().item())
print("compile", torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)(a).shape)
