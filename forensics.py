import torch
from pprint import pprint

obj = torch.load(
    "ode.pt",
    map_location="cpu",
    weights_only=False
)
print("1. Dump the actual checkpoint structure")

print(obj.keys())

# for k, v in obj.items():
#     print("\n", k)
#     print(type(v))

#     if isinstance(v, dict):
#         print(v.keys())


print("2. Inspect tokenizer object deeply")

tok = obj.get("tokenizer")

print(type(tok))
print(repr(tok))

if hasattr(tok, "__dict__"):
    print(tok.__dict__)




# pprint.pp(obj)

print("3. Recursive string search inside loaded object")

def walk(x, path="root"):
    if isinstance(x, dict):
        for k, v in x.items():
            if "ctf" in str(k).lower():
                print(path, "KEY:", k)

            walk(v, f"{path}.{k}")

    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            walk(v, f"{path}[{i}]")

    elif isinstance(x, str):
        s = x.lower()

        if any(word in s for word in [
            "flag",
            "ctf",
            "secret",
            "token",
            "arcus",
            "key"
        ]):
            print(path, repr(x))

walk(obj)