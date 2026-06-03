import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Load checkpoint
# -------------------------

obj = torch.load(
    "ode.pt",
    map_location="cpu",
    weights_only=False
)

cfg = obj["model_config"]
state_dict = obj["model"]


# -------------------------
# Tiny GPT implementation
# -------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        self.n_head = n_head
        self.n_embd = n_embd

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        head_dim = C // self.n_head

        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
        att = att.masked_fill(
            self.bias[:, :, :T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(
            n_embd,
            n_head,
            block_size
        )

        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.block_size = cfg["block_size"]

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(
                cfg["vocab_size"],
                cfg["n_embd"]
            ),

            "wpe": nn.Embedding(
                cfg["block_size"],
                cfg["n_embd"]
            ),

            "h": nn.ModuleList([
                Block(
                    cfg["n_embd"],
                    cfg["n_head"],
                    cfg["block_size"]
                )
                for _ in range(cfg["n_layer"])
            ]),

            "ln_f": nn.LayerNorm(cfg["n_embd"], bias=False)
        })

        self.lm_head = nn.Linear(
            cfg["n_embd"],
            cfg["vocab_size"],
            bias=False
        )
        self.lm_head.weight = self.transformer["wte"].weight

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device)

        tok = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)

        x = tok + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        return logits


# -------------------------
# Load weights
# -------------------------

model = GPT(cfg)
missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing:", missing)
print("Unexpected:", unexpected)
model.eval()


# -------------------------
# Tokenizer
# -------------------------

special_tokens = {
    "<|fernando_pessoa|>": 256,
    "<|alberto_caeiro|>": 257,
    "<|ricardo_reis|>": 258,
    "<|bernardo_soares|>": 259,
    "_": 260,
    "{": 261,
}

reverse_special = {
    v: k for k, v in special_tokens.items()
}


def encode(text):
    tokens = []

    i = 0
    while i < len(text):
        matched = False

        for special, token_id in special_tokens.items():
            if text.startswith(special, i):
                tokens.append(token_id)
                i += len(special)
                matched = True
                break

        if matched:
            continue

        tokens.extend(
            text[i].encode("utf-8")
        )

        i += 1

    return tokens


def decode(tokens):
    raw_bytes = bytearray()

    out = []

    for t in tokens:
        if t < 256:
            raw_bytes.append(t)

        elif t in reverse_special:
            # flush pending utf8 bytes
            if raw_bytes:
                out.append(
                    raw_bytes.decode(
                        "utf-8",
                        errors="replace"
                    )
                )
                raw_bytes.clear()

            out.append(
                reverse_special[t]
            )

    # flush remaining bytes
    if raw_bytes:
        out.append(
            raw_bytes.decode(
                "utf-8",
                errors="replace"
            )
        )

    return "".join(out)


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=300,
    temperature=0.8,
    top_k=40,
):
    tokens = encode(prompt)

    for _ in range(max_new_tokens):

        x = torch.tensor(
            tokens[-cfg["block_size"]:]
        ).unsqueeze(0)

        logits = model(x)

        # take logits for last token
        logits = logits[0, -1]

        # temperature scaling
        logits = logits / temperature

        # top-k sampling
        k = min(top_k, logits.shape[-1])

        values, indices = torch.topk(
            logits,
            k
        )

        probs = torch.softmax(
            values,
            dim=-1
        )

        sampled_idx = torch.multinomial(
            probs,
            1
        ).item()

        next_token = indices[
            sampled_idx
        ].item()

        tokens.append(next_token)

    return decode(tokens)


@torch.no_grad()
def inspect_next_tokens(prompt, top_k=50):
    tokens = encode(prompt)

    x = torch.tensor(
        tokens[-cfg["block_size"]:]
    ).unsqueeze(0)

    logits = model(x)[0, -1]

    probs = torch.softmax(
        logits,
        dim=-1
    )

    values, indices = torch.topk(
        probs,
        top_k
    )

    print("\nPROMPT:", repr(prompt))
    print("-" * 80)

    for p, idx in zip(values, indices):
        idx = idx.item()

        if idx < 256:
            token = bytes([idx]).decode(
                "utf-8",
                errors="replace"
            )
        else:
            token = reverse_special.get(
                idx,
                f"<UNK:{idx}>"
            )

        print(
            f"{repr(token):20} "
            f"{p.item():.6f}"
        )

def inspect_special_token_similarity():
    emb = state_dict["transformer.wte.weight"]

    names = {
        256: "fernando_pessoa",
        257: "alberto_caeiro",
        258: "ricardo_reis",
        259: "bernardo_soares",
        260: "_",
        261: "{",
    }

    for token_id, name in names.items():
        vec = emb[token_id]

        sims = torch.mv(
            emb,
            vec
        )

        topk = torch.topk(sims, 20)

        print("\n" + "=" * 80)
        print(f"TOKEN {token_id}: {name}")
        print("=" * 80)

        for score, idx in zip(
            topk.values,
            topk.indices
        ):
            idx = idx.item()

            if idx < 256:
                try:
                    tok = bytes([idx]).decode(
                        "utf-8"
                    )
                except:
                    tok = repr(idx)
            else:
                tok = names.get(
                    idx,
                    f"<{idx}>"
                )

            print(
                f"{idx:3d} "
                f"{repr(tok):20} "
                f"{score.item():.4f}"
            )

@torch.no_grad()
def greedy_flag(prefix="flag{", max_len=80):
    text = prefix

    print("START:", repr(text))

    for step in range(max_len):

        tokens = encode(text)

        x = torch.tensor(
            tokens[-cfg["block_size"]:]
        ).unsqueeze(0)

        logits = model(x)[0, -1]

        probs = torch.softmax(
            logits,
            dim=-1
        )

        value, idx = torch.max(
            probs,
            dim=-1
        )

        idx = idx.item()
        confidence = value.item()

        if idx < 256:
            token = bytes([idx]).decode(
                "utf-8",
                errors="replace"
            )
        else:
            token = reverse_special.get(
                idx,
                f"<UNK:{idx}>"
            )

        print(
            f"{step:02d} "
            f"{repr(token):20} "
            f"{confidence:.6f}"
        )

        text += token

        print(repr(text))

        # stop conditions
        if token == "}":
            break

        if token == "\n":
            break

    return text

# -------------------------
# Test prompts
# -------------------------

# prompts = [
#     "<|fernando_pessoa|>\n",
#     "<|alberto_caeiro|>\n",
#     "<|ricardo_reis|>\n",
#     "<|bernardo_soares|>\n",

#     "<|fernando_pessoa|>\nOde Triunfal\n",
#     "<|fernando_pessoa|>\nCanto, e canto o presente,\n",

#     "flag",
#     "flag{",
#     "{",
#     "_",

#     "Há Platão e Virgílio dentro das máquinas",
#     "<|fernando_pessoa|>\nflag:",
# ]

# for p in prompts:
#     print("\n" + "=" * 80)
#     print("PROMPT:", repr(p))
#     print("=" * 80)

#     for i in range(3):
#         print(f"\n--- SAMPLE {i+1} ---\n")

#         out = generate(
#             p,
#             max_new_tokens=200,
#             temperature=0.9,
#             top_k=50
#         )

#         print(out)

# inspect_next_tokens("flag{")
# inspect_next_tokens("<|fernando_pessoa|>\nflag:")

# inspect_special_token_similarity()

# greedy_flag("flag{")
print(state_dict.keys() - model.state_dict().keys())

