#!/usr/bin/env python3
"""
esp_llm15.py — part 2 of the "LLM on ESP32-S3" experiment: stories15M (15.2M parameters).

  python esp_llm15.py info                       # board: chip, flash size, PSRAM, what fits
  python esp_llm15.py flash  [--blob q4]         # firmware + model, picks the variant for your flash
  python esp_llm15.py verify                     # every mode against the host simulator, bit by bit
  python esp_llm15.py mem                        # read bandwidth: SRAM / PSRAM / flash (+autoload)
  python esp_llm15.py ladder                     # the whole optimisation ladder, reflashes as needed
  python esp_llm15.py gen --prompt "Once upon a time"

Host-only (no board needed):
  python esp_llm15.py analyze --model stories15M.bin    # quality of every compression trick
  python esp_llm15.py export  --model stories15M.bin    # rebuild the model blobs from the checkpoint

Requires: numpy, pyserial, esptool.
"""
import argparse, glob, json, math, os, struct, subprocess, sys, time
import numpy as np

F32 = np.float32
HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(HERE, "firmware")
BLOBS = os.path.join(HERE, "blobs")
MODEL_OFFSET = 0x80000

# ═══════════════════════════════ checkpoint & tokenizer ═══════════════════════════════

def load_llama2c(path):
    raw = open(path, "rb").read()
    dim, hidden, L, nh, nkv, vocab, seq = struct.unpack("7i", raw[:28])
    shared = vocab > 0; vocab = abs(vocab)
    hs = dim // nh; kv = dim * nkv // nh
    a = np.frombuffer(raw, dtype=F32, offset=28); o = 0
    def take(*shape):
        nonlocal o
        n = int(np.prod(shape)); t = a[o:o + n].reshape(shape); o += n
        return t.copy()
    m = dict(dim=dim, hidden=hidden, L=L, nh=nh, nkv=nkv, vocab=vocab, seq=seq, hs=hs, kv=kv)
    m["emb"] = take(vocab, dim); m["rms_att"] = take(L, dim)
    m["wq"] = take(L, dim, dim); m["wk"] = take(L, kv, dim); m["wv"] = take(L, kv, dim); m["wo"] = take(L, dim, dim)
    m["rms_ffn"] = take(L, dim)
    m["w1"] = take(L, hidden, dim); m["w2"] = take(L, dim, hidden); m["w3"] = take(L, hidden, dim)
    m["rms_final"] = take(dim)
    if not shared: o += seq * hs; m["cls"] = take(vocab, dim)
    else: m["cls"] = m["emb"]
    return m

class Tokenizer:
    """llama2.c tokenizer (run.c port) + optional pruned vocabulary."""
    def __init__(self, path, vocab_size=32000):
        raw = open(path, "rb").read()
        o = 4; self.vocab, self.scores = [], []
        for _ in range(vocab_size):
            s, n = struct.unpack_from("fi", raw, o); o += 8
            self.vocab.append(raw[o:o + n]); o += n; self.scores.append(s)
        self.lookup = {t: i for i, t in enumerate(self.vocab)}
        self.keep = None

    def set_keep(self, vocab_map):
        """restrict to a pruned vocabulary; tokens outside it are split into kept pieces"""
        self.keep = np.asarray(vocab_map)
        self.g2l = {int(g): i for i, g in enumerate(self.keep)}
        self.kl = {self.vocab[int(g)]: int(g) for g in self.keep if g >= 259}
        self.maxlen = max(len(t) for t in self.kl)

    def _bpe(self, text):
        toks = [1]
        if text: toks.append(self.lookup[b" "])
        for ch in text:
            b = ch.encode("utf-8")
            toks += [self.lookup[b]] if b in self.lookup else [x + 3 for x in b]
        while True:
            best, bi, bid = -1e10, -1, -1
            for i in range(len(toks) - 1):
                j = self.lookup.get(self.vocab[toks[i]] + self.vocab[toks[i + 1]])
                if j is not None and self.scores[j] > best: best, bi, bid = self.scores[j], i, j
            if bi < 0: break
            toks[bi:bi + 2] = [bid]
        return toks

    def _split(self, bs):
        out, i = [], 0
        while i < len(bs):
            for L in range(min(self.maxlen, len(bs) - i), 0, -1):
                j = self.kl.get(bs[i:i + L])
                if j is not None: out.append(j); i += L; break
            else: out.append(bs[i] + 3); i += 1
        return out

    def encode(self, text):
        """-> global ids (or local ids of the pruned vocabulary if set_keep was called)"""
        g = self._bpe(text)
        if self.keep is None: return g
        out = []
        for t in g:
            out += [t] if t in self.g2l else self._split(self.vocab[t])
        return [self.g2l[t] for t in out]

    def decode_ids(self, ids):
        """ids: local if pruned, else global"""
        g = [int(self.keep[i]) for i in ids] if self.keep is not None else list(ids)
        out = b""
        for i in range(1, len(g)):
            if g[i] == 1: continue
            p = self.vocab[g[i]]
            if g[i - 1] == 1 and p.startswith(b" "): p = p[1:]
            if p.startswith(b"<0x") and p.endswith(b">") and len(p) == 6: p = bytes([int(p[3:5], 16)])
            out += p
        return out.decode("utf-8", "replace")

# ═══════════════════════════════ numerics shared with the firmware ═══════════════════════════════

def round_away(v):
    v = v.astype(F32)
    return np.trunc(v + np.where(v >= 0, F32(0.5), F32(-0.5)).astype(F32)).astype(np.int32)

def quant_vec(x):
    x = x.astype(F32); amax = np.abs(x).max()
    if amax == 0: return np.zeros(x.shape, np.int8), F32(1)
    inv = F32(127.0) / F32(amax); s = F32(amax) / F32(127.0)
    return np.clip(round_away(x * inv), -127, 127).astype(np.int8), s

def seqsum(a, axis=-1):
    return np.cumsum(a.astype(F32), axis=axis, dtype=F32).take(-1, axis=axis)

C0, C1, C2, C3, C4 = (F32(1.0000035762786865), F32(0.692969560623169), F32(0.24162131547927856),
                      F32(0.05171773582696915), F32(0.013683983124792576))

def fast_exp(x):
    x = np.asarray(x, F32)
    x = np.minimum(np.maximum(x, F32(-87.0)), F32(88.0))
    t = x * F32(1.442695041)
    i = np.trunc(t).astype(np.int32)
    i = i - (t < i.astype(F32)).astype(np.int32)
    f = t - i.astype(F32)
    p = C0 + f * (C1 + f * (C2 + f * (C3 + f * C4)))
    return (p * ((i + 127) << 23).astype(np.int32).view(F32)).astype(F32)

def rope_table(hs, seq):
    t = np.zeros((seq, hs // 2, 2), F32)
    for i in range(hs // 2):
        freq = 1.0 / F32(10000.0) ** (F32(2 * i) / F32(hs))
        ang = np.arange(seq, dtype=F32) * F32(freq)
        t[:, i, 0] = np.cos(ang); t[:, i, 1] = np.sin(ang)
    return t

def apply_rope(v, nheads, hs, cs):
    v = v.reshape(nheads, hs // 2, 2); c, s = cs[:, 0], cs[:, 1]
    a, b = v[..., 0].copy(), v[..., 1].copy()
    v[..., 0] = a * c - b * s; v[..., 1] = a * s + b * c
    return v.reshape(-1)

# ═══════════════════════════════ quantizers ═══════════════════════════════

def quant_q8(w):
    """symmetric int8 per row (part 1): -> q int8, s float32"""
    w = w.astype(F32); amax = np.abs(w).max(1)
    s = np.where(amax > 0, amax / F32(127), F32(1)).astype(F32)
    q = np.clip(round_away(w * (F32(1) / s)[:, None]), -127, 127).astype(np.int8)
    return q, s

def _grid(blk, bits=4, grid=(1.0, .95, .9, .85, .8, .75, .7)):
    best = None
    for c in grid:
        lo = np.minimum(blk.min(1, keepdims=True), 0) * c; hi = np.maximum(blk.max(1, keepdims=True), 0) * c
        s = (hi - lo) / (2 ** bits - 1); s[s == 0] = 1e-8
        z = np.clip(np.round(-lo / s), 0, 2 ** bits - 1)
        d = (np.clip(np.round(blk / s) + z, 0, 2 ** bits - 1) - z) * s
        e = ((d - blk) ** 2).sum(1, keepdims=True)
        if best is None: best = [s, z, e]
        else:
            mm = e < best[2]; best = [np.where(mm, s, best[0]), np.where(mm, z, best[1]), np.where(mm, e, best[2])]
    return best[0][:, 0], best[1][:, 0]

def quant_q4(w, H=None, actorder=True, damp=0.01, rtn=False, sym=False):
    """
    4-bit, groups of 32, asymmetric (zero point per group), group scale = row_unit * u (u: 1..127).
    With H (= X^T X of the layer inputs) the rounding is done by GPTQ in act-order.
    -> q (0..15), z (0..15), u (1..127), unit (float64 per row)
    """
    W = w.astype(np.float64).copy(); R, C = W.shape; g = 32
    S = np.zeros((R, C // g)); Z = np.zeros((R, C // g))
    for k in range(C // g):
        blk = W[:, k * g:(k + 1) * g]
        if sym:
            s = np.abs(blk).max(1) / 7.5; s[s == 0] = 1e-8; S[:, k] = s; Z[:, k] = 8
        else:
            S[:, k], Z[:, k] = _grid(blk)
    unit = S.max(1, keepdims=True) / 127
    U = np.clip(np.ceil(S / unit - 1e-9), 1, 127)
    S = U * unit
    Q = np.zeros((R, C), np.int64)
    if rtn or H is None:
        for k in range(C // g):
            blk = W[:, k * g:(k + 1) * g]
            Q[:, k * g:(k + 1) * g] = np.clip(np.round(blk / S[:, k:k + 1]) + Z[:, k:k + 1], 0, 15)
        return Q, Z.astype(np.int64), U.astype(np.int64), unit[:, 0]
    H = H.astype(np.float64).copy()
    dead = np.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    perm = np.argsort(-np.diag(H), kind="stable") if actorder else np.arange(C)
    Wp = W[:, perm]; Hp = H[perm][:, perm]
    Hp += damp * np.mean(np.diag(Hp)) * np.eye(C)
    Hinv = np.linalg.cholesky(np.linalg.inv(Hp)).T
    for i in range(C):
        col = perm[i]; k = col // g
        s, z = S[:, k:k + 1], Z[:, k:k + 1]
        q = np.clip(np.round(Wp[:, i:i + 1] / s) + z, 0, 15)
        Q[:, col] = q[:, 0].astype(np.int64)
        err = (Wp[:, i:i + 1] - (q - z) * s) / Hinv[i, i]
        Wp[:, i + 1:] -= err @ Hinv[i:i + 1, i + 1:]
    return Q, Z.astype(np.int64), U.astype(np.int64), unit[:, 0]

def deq_q4(Q, Z, U, unit):
    R, C = Q.shape
    return (((Q - np.repeat(Z, 32, 1)) * np.repeat(U, 32, 1)) * unit[:, None]).astype(F32)

def pack_q4(Q, Z, U, unit):
    """-> bytes of the firmware layout: packed nibbles, row scale (unit/16), u, z (signed, -8 offset)"""
    R, C = Q.shape
    qs = (Q - 8).astype(np.int64) & 0xF
    blk = qs.reshape(R, C // 32, 2, 16)
    packed = (blk[:, :, 0, :] | (blk[:, :, 1, :] << 4)).astype(np.uint8).reshape(R, C // 2)
    return packed, (unit / 16).astype(F32), U.astype(np.uint8), (Z - 8).astype(np.int8)

# ═══════════════════════════════ blob (flash image of the model) ═══════════════════════════════

class Blob:
    def __init__(self):
        self.out = bytearray(128)
    def put(self, arr):
        self.out.extend(np.ascontiguousarray(arr).tobytes())
        self.out.extend(b"\0" * ((-len(self.out)) % 64))
    def put_q8(self, q, s): self.put(q); self.put(s)
    def put_q4(self, packed, s, u, z): self.put(packed); self.put(s); self.put(u); self.put(z)

def build_blob(m, vocab_map, wfmt, mats, emb8, emb4, tables, seq=None):
    """mats[l][name] = ('q8', q, s) or ('q4', packed, s, u, z)"""
    seq = seq or m["seq"]; V = len(vocab_map)
    b = Blob()
    b.put(np.asarray(vocab_map, np.uint16))
    b.put_q8(*emb8)
    if emb4 is not None: b.put_q4(*emb4)
    b.put(m["rms_final"].astype(F32))
    for l in range(m["L"]):
        b.put(m["rms_att"][l].astype(F32)); b.put(m["rms_ffn"][l].astype(F32))
        for n in ("qkv", "wo", "w13", "w2"):
            t = mats[l][n]
            if t[0] == "q8": b.put_q8(t[1], t[2])
            else: b.put_q4(*t[1:])
    b.put(rope_table(m["hs"], seq))
    ntri = 0
    if tables is not None:
        keys, vals, bi = tables; ntri = len(keys)
        b.put(keys.astype(np.uint32)); b.put(vals.astype(np.uint16)); b.put(bi.astype(np.uint16))
    h = [0] * 32
    h[1:16] = [3, m["dim"], m["hidden"], m["L"], m["nh"], m["nkv"], V, seq, wfmt, 0, int(emb4 is not None), 0,
               len(b.out), ntri, m["vocab"]]
    struct.pack_into("<4s31I", b.out, 0, b"ESP3", *h[1:])
    return bytes(b.out)

def parse_blob(raw):
    """numpy view of everything the firmware reads (same layout rules as llm_parse)"""
    h = struct.unpack_from("<32I", raw, 0)
    assert raw[:4] == b"ESP3", "not an ESP3 blob"
    c = dict(dim=h[2], hidden=h[3], L=h[4], nh=h[5], nkv=h[6], vocab=h[7], seq=h[8], wfmt=h[9], has4=h[11],
             size=h[13], ntri=h[14], orig_vocab=h[15])
    c["hs"] = c["dim"] // c["nh"]; c["kv"] = c["dim"] * c["nkv"] // c["nh"]
    o = [128]
    def take(dtype, n):
        a = np.frombuffer(raw, dtype=dtype, count=n, offset=o[0]); o[0] += (a.nbytes + 63) // 64 * 64
        return a
    def q8(R, C): return ("q8", take(np.int8, R * C).reshape(R, C), take(F32, R))
    def q4(R, C):
        p = take(np.uint8, R * C // 2).reshape(R, C // 2); s = take(F32, R)
        u = take(np.uint8, R * C // 32).reshape(R, C // 32); z = take(np.int8, R * C // 32).reshape(R, C // 32)
        blk = p.reshape(R, C // 32, 16)
        lo = ((blk << 4).astype(np.uint8).view(np.int8)).astype(np.int32)   # 16*q (signed)
        hi = ((blk & 0xF0).astype(np.uint8).view(np.int8)).astype(np.int32)
        v16 = np.concatenate([lo, hi], 2).reshape(R, C)
        return ("q4", v16, s, u.astype(np.int64), z.astype(np.int64))
    D, H, V, kv = c["dim"], c["hidden"], c["vocab"], c["kv"]
    P = dict(cfg=c)
    P["vocab_map"] = take(np.uint16, V).copy()
    P["emb"] = q8(V, D)
    P["emb4"] = q4(V, D) if c["has4"] else None
    P["rms_final"] = take(F32, D)
    mk = q8 if c["wfmt"] == 0 else q4
    P["layers"] = []
    for l in range(c["L"]):
        lay = dict(rms_att=take(F32, D), rms_ffn=take(F32, D))
        lay["qkv"] = mk(D + 2 * kv, D); lay["wo"] = mk(D, D); lay["w13"] = mk(2 * H, D); lay["w2"] = mk(D, H)
        P["layers"].append(lay)
    P["rope"] = take(F32, c["seq"] * c["hs"]).reshape(c["seq"], c["hs"] // 2, 2)
    if c["ntri"]:
        P["tri_key"] = take(np.uint32, c["ntri"]); P["tri_val"] = take(np.uint16, c["ntri"]); P["bi_val"] = take(np.uint16, V)
    return P

def mat_bytes(t):
    R, C = t[1].shape
    return R * C + 4 * R if t[0] == "q8" else R * C // 2 + 4 * R + 2 * R * C // 32

# ═══════════════════════════════ Sim3: bit-level mirror of llm3.c ═══════════════════════════════

class Sim3:
    def __init__(self, P, cls2=False):
        self.P, self.c, self.cls2 = P, P["cfg"], cls2
        self.reset()

    def reset(self):
        c = self.c
        self.k8 = np.zeros((c["L"], c["seq"], c["kv"]), np.int8); self.v8 = np.zeros_like(self.k8)
        self.ks = np.zeros((c["L"], c["seq"]), F32); self.vs = np.zeros_like(self.ks)

    @staticmethod
    def rows(W, x):
        xq, sx = quant_vec(x)
        if W[0] == "q8":
            acc = W[1].astype(np.int32) @ xq.astype(np.int32)
            return (acc.astype(F32) * W[2]) * sx
        _, v16, s, u, z = W
        R, C = v16.shape; ng = C // 32
        a = (v16.reshape(R, ng, 32).astype(np.int64) * xq.reshape(ng, 32).astype(np.int64)).sum(-1)
        xs = xq.reshape(ng, 32).astype(np.int64).sum(-1)
        t = ((a - 16 * z * xs[None, :]) * u).sum(-1)
        return (t.astype(F32) * s) * sx

    @staticmethod
    def rmsnorm(x, w):
        ss = seqsum(x * x); ss = ss / F32(x.shape[0]) + F32(1e-5); ss = F32(1.0) / np.sqrt(F32(ss))
        return (w * (F32(ss) * x)).astype(F32)

    def forward(self, tok, pos):
        P, c = self.P, self.c
        D, hs, nh, nkv, kv, H = c["dim"], c["hs"], c["nh"], c["nkv"], c["kv"], c["hidden"]
        rep = nh // nkv
        x = P["emb"][1][tok].astype(F32) * P["emb"][2][tok]
        cs = P["rope"][pos]; isq = F32(1.0) / np.sqrt(F32(hs))
        for l, lay in enumerate(P["layers"]):
            xb = self.rmsnorm(x, lay["rms_att"])
            qkv = self.rows(lay["qkv"], xb)
            qv = apply_rope(qkv[:D].copy(), nh, hs, cs)
            kvec = apply_rope(qkv[D:D + kv].copy(), nkv, hs, cs)
            self.k8[l, pos], self.ks[l, pos] = quant_vec(kvec)
            self.v8[l, pos], self.vs[l, pos] = quant_vec(qkv[D + kv:])
            att_out = np.zeros(D, F32)
            for h in range(nh):
                kh = h // rep
                K = self.k8[l, :pos + 1, kh * hs:(kh + 1) * hs].astype(F32)
                V = self.v8[l, :pos + 1, kh * hs:(kh + 1) * hs].astype(F32)
                qh = qv[h * hs:(h + 1) * hs] * isq
                sc = seqsum(K * qh[None, :], axis=1) * self.ks[l, :pos + 1]
                e = fast_exp(sc - sc.max())
                inv = F32(1.0) / seqsum(e)
                w = (e * inv) * self.vs[l, :pos + 1]
                att_out[h * hs:(h + 1) * hs] = seqsum(w[:, None] * V, axis=0)
            x = x + self.rows(lay["wo"], att_out)
            xb = self.rmsnorm(x, lay["rms_ffn"])
            h13 = self.rows(lay["w13"], xb)
            h1, h3 = h13[:H], h13[H:]
            x = x + self.rows(lay["w2"], (h1 / (F32(1.0) + fast_exp(-h1))) * h3)
        xb = self.rmsnorm(x, P["rms_final"])
        if self.cls2 and P["emb4"] is not None:
            approx = self.rows(P["emb4"], xb)
            order = np.lexsort((np.arange(len(approx)), -approx.astype(np.float64)))[:16]
            cand = np.sort(order)
            exact = self.rows(("q8", P["emb"][1][cand], P["emb"][2][cand]), xb)
            b = int(np.argmax(exact)); srt = np.sort(exact)[::-1]
            return int(cand[b]), float(srt[0]), float(srt[1]) if len(srt) > 1 else -1e30
        lg = self.rows(P["emb"], xb)
        b = int(np.argmax(lg)); srt = np.sort(lg)[::-1]
        return b, float(srt[0]), float(srt[1])

    def logits(self, tok, pos):
        """full int8 logits (for quality measurements)"""
        self.forward(tok, pos)
        return None

def sim_generate(sim, ids, steps):
    """greedy, one token per pass -> [(tok, top, second)]"""
    sim.reset(); out = []
    for pos in range(len(ids) - 1): sim.forward(ids[pos], pos)
    tok, pos = ids[-1], len(ids) - 1
    for _ in range(steps):
        r = sim.forward(tok, pos); out.append(r); tok = r[0]; pos += 1
        if tok == 1 or pos >= sim.c["seq"]: break
    return out

# ═══════════════════════════════ export (host) ═══════════════════════════════

def collect_H(m, seqs, B=16):
    """X^T X of the inputs of every linear layer of the float model (calibration for GPTQ)"""
    H = {}
    def add(key, X):
        X = X.reshape(-1, X.shape[-1]).astype(np.float64); H[key] = H.get(key, 0) + X.T @ X
    D, L, nh, nkv, hs = m["dim"], m["L"], m["nh"], m["nkv"], m["hs"]
    rope = rope_table(hs, m["seq"])
    def rms(x, w): return (x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5)) * w
    for i in range(0, len(seqs), B):
        S = seqs[i:i + B]; valid = S[:, :-1] >= 0; toks = np.where(S < 0, 0, S)[:, :-1]
        Bn, T = toks.shape
        x = m["emb"][toks].astype(F32); c, s = rope[:T, :, 0], rope[:T, :, 1]
        mask = np.triu(np.full((T, T), -np.inf, F32), 1)
        def rot(t):
            a, b = t[..., 0], t[..., 1]; cc, ss = c[None, :, None], s[None, :, None]
            return np.stack([a * cc - b * ss, a * ss + b * cc], -1).reshape(*t.shape[:3], hs)
        for l in range(L):
            h = rms(x, m["rms_att"][l]); add(("qkv", l), h[valid])
            q = rot((h @ m["wq"][l].T).reshape(Bn, T, nh, hs // 2, 2))
            k = rot((h @ m["wk"][l].T).reshape(Bn, T, nkv, hs // 2, 2))
            v = (h @ m["wv"][l].T).reshape(Bn, T, nkv, hs)
            att = np.einsum("bthd,bshd->bhts", q, k) / F32(math.sqrt(hs)) + mask
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            o = np.einsum("bhts,bshd->bthd", att, v).reshape(Bn, T, D); add(("wo", l), o[valid])
            x = x + o @ m["wo"][l].T
            h = rms(x, m["rms_ffn"][l]); add(("w13", l), h[valid])
            a1 = h @ m["w1"][l].T; a3 = h @ m["w3"][l].T; g = a1 / (1 + np.exp(-a1)) * a3; add(("w2", l), g[valid])
            x = x + g @ m["w2"][l].T
        add(("cls", 0), rms(x, m["rms_final"])[valid])
    return H

def ngram_tables(corpus, vocab_map):
    """self-speculation tables from text the model generated itself (local ids)"""
    import collections
    g2l = {int(g): i for i, g in enumerate(vocab_map)}; V = len(vocab_map)
    tri = collections.defaultdict(collections.Counter); bi = collections.defaultdict(collections.Counter)
    for s in corpus:
        s = [g2l.get(int(t), -1) for t in s if t >= 0]
        for i in range(len(s) - 1):
            if s[i] >= 0 and s[i + 1] >= 0: bi[s[i]][s[i + 1]] += 1
            if i >= 1 and min(s[i - 1], s[i], s[i + 1]) >= 0: tri[(s[i - 1], s[i])][s[i + 1]] += 1
    items = sorted((a * V + b, c.most_common(1)[0][0]) for (a, b), c in tri.items() if sum(c.values()) >= 2)
    keys = np.array([k for k, _ in items], np.uint32); vals = np.array([v for _, v in items], np.uint16)
    bv = np.full(V, 0xFFFF, np.uint16)
    for a, c in bi.items(): bv[a] = c.most_common(1)[0][0]
    return keys, vals, bv

def choose_vocab(soft, K):
    base = list(range(0, 259))
    rest = [int(i) for i in np.argsort(-soft) if i >= 259][:K - len(base)]
    return np.array(sorted(base + rest), np.uint16)

def cmd_export(a):
    m = load_llama2c(a.model)
    aux = np.load(os.path.join(HERE, "data", "aux15.npz"))
    t0 = time.time()
    H = collect_H(m, aux["calib"][:a.calib])
    print(f"calibration: {a.calib} stories, {time.time()-t0:.0f}s")
    os.makedirs(a.out, exist_ok=True)
    for spec in a.formats.split(","):
        fmt, V = (spec.split("@") + [str(a.vocab)])[:2]; V = int(V)
        vmap = choose_vocab(aux["soft_counts"], V) if V < m["vocab"] else np.arange(m["vocab"], dtype=np.uint16)
        tables = ngram_tables(aux["greedy_corpus"], vmap)
        emb = m["emb"][vmap.astype(np.int64)]
        emb8 = quant_q8(emb)
        emb4 = pack_q4(*quant_q4(emb, H[("cls", 0)])) if fmt.startswith("q4") else None
        print(f"{fmt}: vocabulary {len(vmap)} of {m['vocab']}, {len(tables[0])} trigrams")
        mats = []
        for l in range(m["L"]):
            ws = dict(qkv=np.concatenate([m["wq"][l], m["wk"][l], m["wv"][l]]), wo=m["wo"][l],
                      w13=np.concatenate([m["w1"][l], m["w3"][l]]), w2=m["w2"][l])
            hk = dict(qkv="qkv", wo="wo", w13="w13", w2="w2")
            lay = {}
            for n, w in ws.items():
                if fmt == "q8": lay[n] = ("q8",) + quant_q8(w)
                else: lay[n] = ("q4",) + pack_q4(*quant_q4(w, None if fmt == "q4rtn" else H[(hk[n], l)], rtn=fmt == "q4rtn"))
            mats.append(lay)
            print(f"  {fmt}: layer {l} done ({time.time()-t0:.0f}s)", flush=True)
        raw = build_blob(m, vmap, 0 if fmt == "q8" else 1, mats, emb8, emb4, tables)
        name = f"stories15M_{fmt}_v{len(vmap)}.esp3"
        open(os.path.join(a.out, name), "wb").write(raw)
        print(f"wrote {name}: {len(raw)/2**20:.2f} MB")

# ═══════════════════════════════ quality (host) ═══════════════════════════════

class Float15:
    """float32 reference forward (== run.c), optional pruned classifier"""
    def __init__(self, m, keep=None):
        self.m, self.keep = m, keep
        self.rope = rope_table(m["hs"], m["seq"]); self.reset()
        self.cls = m["cls"] if keep is None else m["cls"][np.asarray(keep, np.int64)]
    def reset(self):
        m = self.m; self.k = np.zeros((m["L"], m["seq"], m["kv"]), F32); self.v = np.zeros_like(self.k)
    def forward(self, tok, pos):
        m = self.m; D, hs, nh, nkv, kv, H = m["dim"], m["hs"], m["nh"], m["nkv"], m["kv"], m["hidden"]
        rms = lambda x, w: (w * (x / np.sqrt(np.mean(x * x) + F32(1e-5)))).astype(F32)
        gid = tok if self.keep is None else int(self.keep[tok])
        x = m["emb"][gid].copy(); cs = self.rope[pos]
        for l in range(m["L"]):
            h = rms(x, m["rms_att"][l])
            q = apply_rope(m["wq"][l] @ h, nh, hs, cs); self.k[l, pos] = apply_rope(m["wk"][l] @ h, nkv, hs, cs)
            self.v[l, pos] = m["wv"][l] @ h
            o = np.zeros(D, F32)
            for hh in range(nh):
                kh = hh // (nh // nkv)
                sc = self.k[l, :pos + 1, kh * hs:(kh + 1) * hs] @ q[hh * hs:(hh + 1) * hs] / F32(math.sqrt(hs))
                e = np.exp(sc - sc.max()); e /= e.sum()
                o[hh * hs:(hh + 1) * hs] = e @ self.v[l, :pos + 1, kh * hs:(kh + 1) * hs]
            x = x + m["wo"][l] @ o
            h = rms(x, m["rms_ffn"][l]); a1 = m["w1"][l] @ h
            x = x + m["w2"][l] @ ((a1 / (1 + np.exp(-a1))) * (m["w3"][l] @ h))
        return self.cls @ rms(x, m["rms_final"])

def sim_logits(sim):
    """Sim3 forward returning the full int8 logits"""
    def f(tok, pos):
        P = sim.P; box = {}
        orig = Sim3.rows
        def spy(W, x):
            r = orig(W, x)
            if W is P["emb"]: box["lg"] = r
            return r
        sim.rows = spy
        try: sim.forward(tok, pos)
        finally: del sim.rows
        return box["lg"]
    return f

def cmd_analyze(a):
    """bits per byte on stories the float model wrote itself, and greedy agreement with float32"""
    m = load_llama2c(a.model)
    aux = np.load(os.path.join(HERE, "data", "aux15.npz"))
    ev = aux["eval"][:a.n]
    tk = Tokenizer(a.tok)
    texts = [tk.decode_ids([int(t) for t in s if t >= 0]) for s in ev]
    nbytes = sum(len(t.encode()) for t in texts)
    def bpb(step, reset, keep_map):
        tk.keep = None
        if keep_map is not None: tk.set_keep(keep_map)
        nll = 0.0
        for t in texts:
            ids = tk.encode(t)[:m["seq"]]; reset()
            for pos in range(len(ids) - 1):
                lg = step(ids[pos], pos).astype(np.float64); lg -= lg.max()
                nll += -(lg[ids[pos + 1]] - math.log(np.exp(lg).sum()))
        tk.keep = None
        return nll / nbytes / math.log(2)
    def greedy_text(step, reset, keep_map, prompt, n):
        tk.keep = None
        if keep_map is not None: tk.set_keep(keep_map)
        ids = tk.encode(prompt); reset(); tok = ids[0]; out = []
        for pos in range(len(ids) + n - 1):
            lg = step(tok, pos); nxt = int(np.argmax(lg))
            tok = ids[pos + 1] if pos + 1 < len(ids) else nxt
            if pos + 1 >= len(ids): out.append(nxt)
        txt = tk.decode_ids(ids + out); tk.keep = None
        return txt
    print(f"{len(texts)} stories ({nbytes} bytes) written by the float model; lower bits/byte is better\n")
    fref = Float15(m)
    base = bpb(fref.forward, fref.reset, None)
    ref_txt = {p: greedy_text(fref.forward, fref.reset, None, p, a.steps) for p in PROMPTS}
    print(f"  {'variant':44s} {'bits/byte':>9s} {'change':>8s} {'same greedy text':>17s}")
    print(f"  {'float32, 32000 tokens (reference)':44s} {base:9.4f} {'':>8s} {'':>17s}")
    def agree(txts):
        r = []
        for p in PROMPTS:
            A, B = ref_txt[p], txts[p]
            r.append(next((i for i, (x, y) in enumerate(zip(A, B)) if x != y), min(len(A), len(B))) - len(p))
        return np.mean(r)
    rows = []
    for path in sorted(glob.glob(os.path.join(BLOBS, "*.esp3"))):
        P = parse_blob(open(path, "rb").read()); vm = P["vocab_map"]
        fp = Float15(m, vm)
        v = bpb(fp.forward, fp.reset, vm)
        t = {p: greedy_text(fp.forward, fp.reset, vm, p, a.steps) for p in PROMPTS}
        print(f"  {'float32, pruned to ' + str(len(vm)) + ' tokens':44s} {v:9.4f} {(v/base-1)*100:+7.2f}% {agree(t):11.0f} chars", flush=True)
        sim = Sim3(P); st = sim_logits(sim)
        v = bpb(st, sim.reset, vm)
        t = {p: greedy_text(st, sim.reset, vm, p, a.steps) for p in PROMPTS}
        name = os.path.basename(path).replace("stories15M_", "").replace(".esp3", "")
        print(f"  {'firmware arithmetic: ' + name:44s} {v:9.4f} {(v/base-1)*100:+7.2f}% {agree(t):11.0f} chars", flush=True)

# ═══════════════════════════════ device ═══════════════════════════════

def find_port():
    c = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/cu.usbserial*") + glob.glob("/dev/cu.wchusbserial*")
               + glob.glob("/dev/cu.SLAB*") + glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not c: sys.exit("no serial port found, pass --port")
    return c[0]

class Device:
    def __init__(self, port, baud=115200):
        import serial
        if port.startswith("socket://"):
            self.s = serial.serial_for_url(port, timeout=0.2)
        else:
            for attempt in range(40):            # USB-Serial-JTAG re-enumerates after a reset
                try:
                    self.s = serial.Serial()
                    self.s.port, self.s.baudrate, self.s.timeout = port, baud, 0.2
                    self.s.dtr = False; self.s.rts = False
                    self.s.open(); break
                except Exception:
                    if attempt == 39: raise
                    time.sleep(0.5)
        self.buf = b""

    def lines(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            self.buf += self.s.read(4096)
            while b"\n" in self.buf:
                ln, self.buf = self.buf.split(b"\n", 1)
                ln = ln.strip().decode("utf-8", "replace")
                if ln: yield ln
        raise TimeoutError("device did not answer")

    def cmd(self, line, timeout=10):
        self.s.reset_input_buffer(); self.buf = b""
        self.s.write((line + "\n").encode()); self.s.flush()
        return self.lines(timeout)

    def info(self, timeout=8):
        for _ in range(3):
            try:
                for ln in self.cmd("INFO", timeout):
                    if ln.startswith("@INFO"): return dict(kv.split("=", 1) for kv in ln.split()[1:])
                    if ln.startswith("!NOMODEL"): sys.exit("the firmware cannot open the model in flash:\n  " + ln + "\nsend this line to the author")
            except TimeoutError:
                pass
        sys.exit("no answer from the board (is the firmware flashed? try: python esp_llm15.py flash)")

    def gen(self, flags, ids, steps, verify=True, timeout=600):
        toks, end = [], None
        for ln in self.cmd(f"GEN {flags} {steps} {int(verify)} " + ",".join(map(str, ids)), timeout):
            if ln.startswith("@T "):
                p = ln.split(); toks.append((int(p[1]), float(p[2]), float(p[3])) if verify else (int(p[1]),))
            elif ln.startswith("@END"):
                end = dict(kv.split("=", 1) for kv in ln.split()[1:]); break
            elif ln.startswith("!ERR"): raise RuntimeError(ln)
        return toks, end

    def mem(self):
        for ln in self.cmd("MEM", 60):
            if ln.startswith("@MEM"): return dict(kv.split("=", 1) for kv in ln.split()[1:])

def open_dev(a):
    d = Device(a.port or find_port()); inf = d.info()
    return d, inf

def blob_for_device(inf):
    want = inf.get("blob", "")
    for p in glob.glob(os.path.join(BLOBS, "*.esp3")):
        if os.path.basename(p).startswith(want) or want == "":
            raw = open(p, "rb").read()
            if struct.unpack_from("<I", raw, 52)[0] == int(inf["blob_bytes"]): return p, raw
    # match by size and header
    for p in glob.glob(os.path.join(BLOBS, "*.esp3")):
        raw = open(p, "rb").read()
        if len(raw) == int(inf["blob_bytes"]): return p, raw
    sys.exit(f"cannot find the blob that is on the board ({inf.get('blob_bytes')} bytes) in {BLOBS}")

PROMPTS = ["Once upon a time", "Lily and Ben went to the park", "The cat was very hungry"]
# openings the n-gram tables have never seen (held-out stories)
FRESH = ["One day, Andy was walking along the street. Suddenly", "Once upon a time, there was a healthy worm named"]


FLAG_NAMES = {"k": {1: "scalar", 2: "PIE"}, "c": {}, "a": {0: "", 1: "autoload", 2: "preload"}}

def describe(flags):
    f = parse_flags(flags)
    s = [("PIE" if f["k"] == 2 else "scalar"), f"{f['c']} core" + ("s" if f["c"] > 1 else "")]
    if f["a"] == 1: s.append("cache autoload")
    if f["a"] == 2: s.append("manual preload")
    if f["x"]: s.append("2-stage classifier")
    if f["w"]: s.append("hot weights in SRAM")
    if f["p"]: s.append(f"self-speculation k={f['p']}")
    return " · ".join(s)

def parse_flags(fl):
    import re
    d = dict(k=1, c=1, a=0, x=0, w=0, p=0)
    for k, v in re.findall(r"([a-z])(\d+)", fl): d[k] = int(v)
    return d

def check_run(sim_out, toks):
    same = all(a[0] == b[0] for a, b in zip(sim_out, toks)) and len(toks) == len(sim_out)
    dl = max((abs(a[1] - b[1]) + abs(a[2] - b[2]) for a, b in zip(sim_out, toks)), default=0)
    first = next((i for i, (a, b) in enumerate(zip(sim_out, toks)) if a[0] != b[0]), None)
    return same, dl, first

def cmd_verify(a):
    d, inf = open_dev(a)
    path, raw = blob_for_device(inf)
    P = parse_blob(raw); tk = Tokenizer(os.path.join(HERE, "tok32k.bin")); tk.set_keep(P["vocab_map"])
    print(f"board: {inf.get('chip')} flash={inf.get('flash_mb')}MB psram={int(inf.get('psram_kb',0))//1024}MB  model: {os.path.basename(path)}")
    modes = a.modes.split(",") if a.modes else ["k1c1", "k2c1", "k2c2", "k2c2a1", "k2c2x1", "k2c2x1w1", "k2c2x1w1p3"]
    ok_all = True
    for p in PROMPTS[:a.prompts]:
        ids = tk.encode(p)
        print(f"\nprompt {p!r}  (host simulator, {a.steps} tokens)")
        ref = {0: sim_generate(Sim3(P, False), ids, a.steps)}
        if P["emb4"] is not None: ref[1] = sim_generate(Sim3(P, True), ids, a.steps)
        for fl in modes:
            f = parse_flags(fl)
            toks, end = d.gen(fl, ids, a.steps)
            same, dl, first = check_run(ref[1 if f["x"] else 0], toks)
            ok = same and dl < 1e-3
            ok_all &= ok
            print(f"  {fl:12s} {describe(fl):58s} {'OK ' if ok else 'BAD'} "
                  f"{'identical tokens' if same else f'first diff at {first}'}  max|Δlogit|={dl:.2e}")
    print("\nRESULT:", "PASS — firmware matches the host simulator" if ok_all else "FAIL")
    print("text:", tk.decode_ids(ids + [t[0] for t in toks])[:300])

def run_bench(d, P, tk, flags, steps):
    """tok/s on two familiar openings and on two openings the draft tables never saw"""
    res = {}
    for kind, prompts in (("std", PROMPTS[:2]), ("fresh", FRESH)):
        rs = []
        for p in prompts:
            toks, end = d.gen(flags, tk.encode(p), steps, verify=False)
            rs.append(end)
        n = sum(int(e["n"]) for e in rs); us = sum(int(e["us_gen"]) for e in rs)
        passes = sum(int(e["passes"]) for e in rs)
        res[kind] = (n / (us / 1e6), n / passes, rs)
    rs = res["std"][2] + res["fresh"][2]
    prof = {k: round(sum(int(e[k]) for e in rs) / len(rs)) for k in ("mm", "cls", "att", "oth")}
    kb = sum(int(e["kbpp"]) for e in rs) / len(rs)
    return dict(flags=flags, tok_s=(res["std"][0] + res["fresh"][0]) / 2, tok_s_std=res["std"][0], tok_s_fresh=res["fresh"][0],
                tok_per_pass=res["std"][1], tok_per_pass_fresh=res["fresh"][1], kb_per_pass=kb, **prof)

def cmd_bench(a):
    d, inf = open_dev(a)
    path, raw = blob_for_device(inf)
    P = parse_blob(raw); tk = Tokenizer(os.path.join(HERE, "tok32k.bin")); tk.set_keep(P["vocab_map"])
    modes = a.modes.split(",") if a.modes else ["k1c1", "k2c1", "k2c2", "k2c2a1", "k2c1a2", "k2c2x1", "k2c2x1w1",
                                                "k2c2x1w1p2", "k2c2x1w1p3", "k2c2x1w1p4"]
    out = []
    print(f"device: {' '.join(f'{k}={v}' for k, v in inf.items())}")
    print("  tok/s: familiar opening / unseen opening")
    for fl in modes:
        r = run_bench(d, P, tk, fl, a.steps); r["blob"] = os.path.basename(path); r["fw"] = inf.get("fw", "")
        out.append(r)
        print(f"  {fl:12s} {describe(fl):62s} {r['tok_s_std']:6.2f} / {r['tok_s_fresh']:5.2f} tok/s  "
              f"{r['tok_per_pass']:.2f}/{r['tok_per_pass_fresh']:.2f} tok/pass  {r['kb_per_pass']:.0f} KB/pass  "
              f"mm {r['mm']}% cls {r['cls']}% att {r['att']}% other {r['oth']}%", flush=True)
    json.dump(out, open(a.out, "w"), indent=1); print("saved", a.out)

def cmd_mem(a):
    d, inf = open_dev(a)
    r = d.mem()
    print(f"board: flash {inf.get('flash_mb')} MB @ {inf.get('flash_mhz')} MHz, PSRAM {int(inf.get('psram_kb',0))//1024} MB, "
          f"dcache {inf.get('dcache_kb')} KB / line {inf.get('dline')} B")
    for k in ("sram", "psram", "flash", "flash_autoload"):
        if k in r: print(f"  {k:16s} {float(r[k]):7.1f} MB/s")

def cmd_gen(a):
    d, inf = open_dev(a)
    path, raw = blob_for_device(inf)
    P = parse_blob(raw); tk = Tokenizer(os.path.join(HERE, "tok32k.bin")); tk.set_keep(P["vocab_map"])
    ids = tk.encode(a.prompt)
    toks, end = d.gen(a.flags, ids, a.steps, verify=False)
    print(tk.decode_ids(ids + [t[0] for t in toks]))
    print(f"\n[{int(end['n'])/(int(end['us_gen'])/1e6):.2f} tok/s, {describe(a.flags)}]")

# ═══════════════════════════════ flashing ═══════════════════════════════

def esptool(args, capture=False):
    cmd = [sys.executable, "-m", "esptool"] + args
    if capture: return subprocess.run(cmd, capture_output=True, text=True)
    r = subprocess.run(cmd)
    if r.returncode: sys.exit("esptool failed")

def detect_board(port):
    r = esptool(["--chip", "esp32s3", "-p", port, "flash_id"], capture=True)
    txt = r.stdout + r.stderr
    import re
    mb = re.search(r"Detected flash size:\s*(\d+)MB", txt)
    ps = re.search(r"Embedded PSRAM (\d+)MB", txt)
    psram = "octal" if ps and int(ps.group(1)) >= 8 else "quad"     # S3R2 = 2 MB quad, S3R8 = 8 MB octal
    return (int(mb.group(1)) if mb else None), psram, txt

def pick_blob(flash_mb, want=None):
    avail = flash_mb * 2 ** 20 - MODEL_OFFSET
    cands = sorted(glob.glob(os.path.join(BLOBS, "*.esp3")), key=os.path.getsize, reverse=True)
    if want:
        cands = [c for c in cands if want in os.path.basename(c)]
    fit = [c for c in cands if os.path.getsize(c) <= avail]
    return fit, avail

def flash_all(port, fw, blob, flash_mb, app_only=False, blob_only=False):
    d = os.path.join(FW, "prebuilt", fw)
    pt = os.path.join(FW, "prebuilt", f"partitions_{flash_mb}MB.bin")
    args = ["--chip", "esp32s3", "-p", port, "-b", "921600", "--before", "default_reset", "--after", "hard_reset",
            "write_flash", "--flash_size", "detect"]
    if not blob_only:
        args += ["0x0", os.path.join(d, "bootloader.bin"), "0x8000", pt, "0x10000", os.path.join(d, "esp_llm15.bin")]
    if blob and not app_only:
        args += [hex(MODEL_OFFSET), blob]
    esptool(args)

def cmd_flash(a):
    port = a.port or find_port()
    mb, psram, txt = detect_board(port)
    if not mb: print(txt); sys.exit("could not read the flash size")
    fit, avail = pick_blob(mb, a.blob)
    print(f"board: {mb} MB flash, {psram} PSRAM; room for the model: {avail/2**20:.2f} MB")
    if not fit: sys.exit("no model blob fits — stories15M needs 8 MB of flash or more (see README)")
    fw = a.fw or ("octal" if psram == "octal" else "quad") + "_c64"
    if not a.blob:                                   # the main model is the 4-bit one; int8 is only for the ladder
        fit = [f for f in fit if "_q4_" in f] + [f for f in fit if "_q4_" not in f]
    print(f"firmware {fw}, model {os.path.basename(fit[0])} ({os.path.getsize(fit[0])/2**20:.2f} MB)")
    flash_all(port, fw, fit[0], mb)
    time.sleep(2)
    d = Device(port); print("board says:", " ".join(f"{k}={v}" for k, v in d.info().items()))

def cmd_info(a):
    port = a.port or find_port()
    mb, psram, txt = detect_board(port)
    print(f"flash: {mb} MB, PSRAM type guess: {psram}")
    fit, avail = pick_blob(mb or 4)
    print("model blobs that fit:", ", ".join(os.path.basename(f) for f in fit) or "none (need 8 MB flash)")

def cmd_ladder(a):
    """reflash firmware/model variants as needed and benchmark the optimisation ladder"""
    port = a.port or find_port()
    mb, psram, _ = detect_board(port)
    fit, avail = pick_blob(mb)
    q8 = [f for f in fit if "_q8_" in f]; q4 = [f for f in fit if "_q4_" in f]
    if not q4: sys.exit("needs 8 MB flash")
    base = "octal" if psram == "octal" else "quad"
    steps = []
    if q8: steps.append((f"{base}_c32", q8[0], "k1c1", "int8, flash XIP, 1 core, scalar"))
    steps += [(f"{base}_c32", q4[0], "k1c1", "int4 GPTQ"),
              (f"{base}_c32", q4[0], "k2c1", "+ PIE kernels (x16 trick)"),
              (f"{base}_c32", q4[0], "k2c2", "+ second core"),
              (f"{base}_c64", q4[0], "k2c2", "+ 64 KB data cache, 64-byte lines"),
              (f"{base}_c64", q4[0], "k2c2a1", "+ cache autoload"),
              (f"{base}_c64", q4[0], "k2c2a1x1", "+ two-stage classifier"),
              (f"{base}_c64", q4[0], "k2c2a1x1w1", "+ hot weights in SRAM"),
              (f"{base}_c64", q4[0], "k2c2a1x1w1p2", "+ self-speculation k=2"),
              (f"{base}_c64", q4[0], "k2c2a1x1w1p3", "+ self-speculation k=3"),
              (f"{base}_c64", q4[0], "k2c2a1x1w1p4", "+ self-speculation k=4")]
    if a.overclock: steps.append((f"{base}_c64f120", q4[0], "BEST", "+ flash & PSRAM at 120 MHz"))
    cur_fw = cur_blob = None; res = []; best = None; drop_autoload = False; mem = {}
    print("tok/s: familiar opening / unseen opening; every step is verified against the host simulator first")
    tk = Tokenizer(os.path.join(HERE, "tok32k.bin"))
    for fw, blob, fl, label in steps:
        if fl == "BEST": fl = best["flags"]
        if "a1" in fl and drop_autoload: fl = fl.replace("a1", "")
        if fw != cur_fw or blob != cur_blob:
            print(f"\n── flashing {fw} + {os.path.basename(blob)}" if blob != cur_blob else f"\n── flashing firmware {fw}")
            flash_all(port, fw, blob, mb, app_only=(blob == cur_blob)); cur_fw, cur_blob = fw, blob
            time.sleep(2)
        try:
            d = Device(port); inf = d.info()
        except (SystemExit, Exception) as e:
            print(f"  {label}: the board did not come up with firmware {fw} ({e}) — skipping")
            cur_fw = None; continue
        if fw not in mem:
            mem[fw] = d.mem()
            print(f"  memory read speed ({fw}): " + ", ".join(f"{k} {float(v):.1f} MB/s" for k, v in mem[fw].items()))
        P = parse_blob(open(blob, "rb").read()); tk.set_keep(P["vocab_map"])
        f = parse_flags(fl)
        ids = tk.encode(PROMPTS[0])
        ref = sim_generate(Sim3(P, bool(f["x"])), ids, 24)
        toks, _ = d.gen(fl, ids, 24)
        same, dl, first = check_run(ref, toks)
        r = run_bench(d, P, tk, fl, a.steps)
        r.update(label=label, fw=fw, blob=os.path.basename(blob), verified=bool(same and dl < 1e-3))
        if "_q4_" in blob and (best is None or r["tok_s"] > best["tok_s"]) and r["verified"]: best = r
        if fl.endswith("a1") and len(res) and r["tok_s"] < res[-1]["tok_s"]:
            drop_autoload = True; print("    (autoload did not help on this board — the next steps go without it)")
        res.append(r); d.s.close()
        print(f"  {label:40s} {r['tok_s_std']:6.2f} / {r['tok_s_fresh']:5.2f} tok/s  {r['tok_per_pass']:.2f}/{r['tok_per_pass_fresh']:.2f} tok/pass  "
              f"{r['kb_per_pass']:5.0f} KB/pass  mm {r['mm']}% cls {r['cls']}% att {r['att']}%  "
              f"{'✓' if r['verified'] else '✗ MISMATCH'}", flush=True)
        json.dump(dict(board=dict(flash_mb=mb, psram=psram, info=inf), mem=mem, ladder=res), open(a.out, "w"), indent=1)
    print("\nsaved", a.out)
    if best:
        print(f"best: {best['tok_s']:.2f} tok/s ({best['label']}, flags {best['flags']}, firmware {best['fw']})")
        if cur_fw != best["fw"]:
            print(f"restoring firmware {best['fw']}"); flash_all(port, best["fw"], cur_blob, mb, app_only=True)

# ═══════════════════════════════ main ═══════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    def dev(p): p.add_argument("--port"); return p
    p = sp.add_parser("export"); p.add_argument("--model", default=os.path.join(HERE, "stories15M.bin"))
    p.add_argument("--out", default=BLOBS); p.add_argument("--vocab", type=int, default=6144)
    p.add_argument("--calib", type=int, default=128); p.add_argument("--formats", default="q8,q4")
    p = sp.add_parser("analyze"); p.add_argument("--tok", default=os.path.join(HERE, "tok32k.bin")); p.add_argument("--n", type=int, default=12)
    p.add_argument("--model", default=os.path.join(HERE, "stories15M.bin")); p.add_argument("--steps", type=int, default=120)
    p = dev(sp.add_parser("info"))
    p = dev(sp.add_parser("flash")); p.add_argument("--blob"); p.add_argument("--fw")
    p = dev(sp.add_parser("verify")); p.add_argument("--steps", type=int, default=48); p.add_argument("--modes")
    p.add_argument("--prompts", type=int, default=2)
    p = dev(sp.add_parser("bench")); p.add_argument("--steps", type=int, default=64); p.add_argument("--modes")
    p.add_argument("--out", default="bench15.json")
    p = dev(sp.add_parser("mem"))
    p = dev(sp.add_parser("ladder")); p.add_argument("--steps", type=int, default=48); p.add_argument("--overclock", action="store_true")
    p.add_argument("--out", default="ladder15.json")
    p = dev(sp.add_parser("gen")); p.add_argument("--prompt", default="Once upon a time"); p.add_argument("--steps", type=int, default=230)
    p.add_argument("--flags", default="k2c2a1x1w1p3")
    a = ap.parse_args()
    dict(export=cmd_export, analyze=cmd_analyze, info=cmd_info, flash=cmd_flash, verify=cmd_verify, bench=cmd_bench,
         mem=cmd_mem, ladder=cmd_ladder, gen=cmd_gen)[a.cmd](a)

if __name__ == "__main__":
    main()
