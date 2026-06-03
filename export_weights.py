"""
export_weights.py — dump ode.pt's tensors to a flat little-endian f32 blob + JSON manifest,
for the native Rust trigger-discovery engine (rust_solver/). One-off.

Run from rust_solver/ (or pass paths): writes weights.bin + manifest.json next to ode.pt's copy.
"""
import torch, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.environ.get("ODE_PT", os.path.join(ROOT, "ode.pt"))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "rust_solver")
os.makedirs(OUTDIR, exist_ok=True)

obj = torch.load(CKPT, map_location="cpu", weights_only=False)
sd, cfg = obj["model"], obj["model_config"]
manifest, offset, buf = [], 0, bytearray()
for name, t in sd.items():
    a = t.detach().to(torch.float32).contiguous().numpy()
    b = a.tobytes()
    manifest.append({"name": name, "shape": list(a.shape), "offset": offset, "n": int(a.size)})
    buf += b
    offset += len(b)

open(os.path.join(OUTDIR, "weights.bin"), "wb").write(buf)
json.dump({"config": cfg, "tensors": manifest},
          open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=0)
print(f"wrote {OUTDIR}/weights.bin ({len(buf)} bytes), {len(manifest)} tensors")
