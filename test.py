import os, pathlib
p = "mediumblog1.txt"
print("cwd:", pathlib.Path().resolve())
print("exists:", os.path.exists(p))
print("size:", os.path.getsize(p) if os.path.exists(p) else "N/A")

# peek content
if os.path.exists(p):
    with open(p, "rb") as f:
        raw = f.read(200)
    print("first bytes:", raw)
