"""Verify μP attention scaling evaluates correctly across configurations.

Run from the repo root:
    python tests/check_attn_scale.py

This is the smallest possible verification of the components/attention.py change
that introduced the caveat μP formula (attn_scale = √base_head_dim / head_dim).
It checks the four cases that matter:

  1. base_head_dim=0 (the default)        → attn_scale=None (SDPA default,
                                              also a sqrt(0) safety check)
  2. head_dim == base_head_dim (today)    → attn_scale = 1/√head_dim
                                              (no transfer change at base)
  3. head_dim > base_head_dim             → attn_scale < 1/√head_dim
                                              (μP damping kicks in)
  4. head_dim < base_head_dim             → attn_scale > 1/√head_dim
                                              (μP boost kicks in)

Exit code 0 = all pass; nonzero = something failed.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from components.attention import SelfAttention, CrossAttention, ComboAttention

cases = [
    # (label, kwargs, expected_attn_scale)
    ("default base_head_dim=0 → None (sqrt(0) safety)",
     dict(dim=64, num_heads=1, base_head_dim=0), None),
    ("head_dim=64, base_head_dim=64 (current sweep at base)",
     dict(dim=64, num_heads=1, base_head_dim=64), 1/math.sqrt(64)),
    ("head_dim=64, base_head_dim=64 at 2× width (constant head_dim)",
     dict(dim=128, num_heads=2, base_head_dim=64), 1/math.sqrt(64)),
    ("head_dim=64, base_head_dim=64 at 4× width (constant head_dim)",
     dict(dim=256, num_heads=4, base_head_dim=64), 1/math.sqrt(64)),
    ("head_dim=128 > base_head_dim=64 (μP damping kicks in)",
     dict(dim=128, num_heads=1, base_head_dim=64), math.sqrt(64)/128),
    ("head_dim=32 < base_head_dim=64 (μP boost kicks in)",
     dict(dim=128, num_heads=4, base_head_dim=64), math.sqrt(64)/32),
]

print(f"{'case':<70}{'expected':>12}{'actual':>12}  status")
print("-" * 110)
all_ok = True
for cls in (SelfAttention, CrossAttention, ComboAttention):
    print(f"\n=== {cls.__name__} ===")
    for label, kwargs, expected in cases:
        attn = cls(**kwargs)
        actual = attn.attn_scale
        if expected is None:
            ok = actual is None
            exp_str, act_str = "None", str(actual)
        else:
            ok = actual is not None and abs(actual - expected) < 1e-9
            exp_str = f"{expected:.6f}"
            act_str = f"{actual:.6f}" if actual is not None else "None"
        all_ok = all_ok and ok
        print(f"  {label:<68}{exp_str:>12}{act_str:>12}  {'OK' if ok else 'FAIL'}")

print()
print("ALL PASS" if all_ok else "SOMETHING FAILED — see above")
sys.exit(0 if all_ok else 1)
