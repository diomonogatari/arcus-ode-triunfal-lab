import torch
import struct

obj = torch.load(
    "ode.pt",
    map_location="cpu",
    weights_only=False
)

w = obj["model"][
    "transformer.h.7.mlp.c_proj.weight"
]

print(w.shape)

raw = w.numpy().tobytes()

needle = b"ctf="
offset = raw.find(needle)

print("offset:", offset)

# float index
float_index = offset // 4
print("float index:", float_index)

rows, cols = w.shape

row = float_index // cols
col = float_index % cols

print("row:", row)
print("col:", col)

print("\nSuspicious area:")
print(
    w[
        max(0, row - 3):
        row + 4
    ]
)