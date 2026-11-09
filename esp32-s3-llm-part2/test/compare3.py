"""host C engine (llm3.c) vs Python Sim3: must agree bit for bit."""
import sys, os, subprocess, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from esp_llm15 import *
blob = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
raw = open(blob, "rb").read(); P = parse_blob(raw)
tk = Tokenizer(os.path.join(HERE, "tok32k.bin")); tk.set_keep(P["vocab_map"])
ok = True
for prompt in ["Once upon a time", "Lily and Ben went to the park"]:
    ids = tk.encode(prompt)
    for k, x in ((0, 0), (3, 0), (0, 1), (4, 1)):
        if x and P["emb4"] is None: continue
        ref = sim_generate(Sim3(P, bool(x)), ids, steps)
        r = subprocess.run(["./host3", blob, str(steps), str(k), ",".join(map(str, ids)), str(x)], capture_output=True, text=True)
        out = [tuple(float(v) if i else int(v) for i, v in enumerate(l.split())) for l in r.stdout.strip().split("\n")]
        same = [a[0] for a in ref] == [b[0] for b in out]
        exact = same and all(np.float32(a[1]) == np.float32(b[1]) and np.float32(a[2]) == np.float32(b[2]) for a, b in zip(ref, out))
        ok &= exact
        print(f"{prompt[:20]:20s} spec k={k} cls2={x}: tokens {'same' if same else 'DIFF'}, logits {'bit-exact' if exact else 'differ'}  {r.stderr.strip()}")
    print("  ", tk.decode_ids(ids + [t[0] for t in ref])[:150].replace("\n", " "))
print("ALL BIT-EXACT" if ok else "MISMATCH")
