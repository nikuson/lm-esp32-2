// llm3.c — Llama-2 forward pass for stories15M on ESP32-S3.
// Weights stay in flash (memory-mapped) and are streamed through the cache once per pass;
// a pass may carry several tokens (speculative verification), so each byte read from flash
// is reused by every token of the batch. Numerics are mirrored op-by-op by Sim3 in esp_llm15.py.
#include "llm3.h"
#include <string.h>
#include <math.h>

#ifdef LLM_HAVE_PIE
void mv_s8_pie(const int8_t *W, const int8_t *x, int32_t *acc, int rows, int n16);
void mv_q4_pie(const uint8_t *W, const int8_t *x, int32_t *gacc, int rows, int nblk, int bpg);
#endif

#ifndef HOT
#define HOT
#endif

void (*llm_parallel)(llm_job_fn fn, int n) = 0;
uint32_t (*llm_cycles)(void) = 0;
void *(*llm_alloc)(size_t size) = 0;
void *(*llm_alloc_slow)(size_t size) = 0;
void (*llm_prefetch)(const void *next, size_t bytes) = 0;
const void *(*llm_addr)(size_t off, size_t len) = 0;

static inline uint32_t cyc(void) { return llm_cycles ? llm_cycles() : 0; }

// ───────────────────────── blob ─────────────────────────

typedef struct { const uint8_t *p; size_t off, len; } rd_t;

static const void *take(rd_t *r, size_t bytes) {
    if (r->off + bytes > r->len) return 0;
    const void *p = llm_addr ? llm_addr(r->off, bytes) : r->p + r->off;
    if (!p) return 0;
    r->off += (bytes + 63) & ~(size_t)63;          // 64-byte aligned sections = whole cache lines
    return p;
}

// bytes streamed per use of the matrix
size_t llm_mat_bytes(const qmat_t *w) {
    if (w->fmt == WF_Q8) return (size_t)w->rows * w->cols + (size_t)w->rows * 4;
    return (size_t)w->rows * w->cols / 2 + (size_t)w->rows * 4 + (size_t)w->rows * (w->cols / 32) * 2;
}

static int take_mat(rd_t *r, qmat_t *w, int rows, int cols, int fmt) {
    w->rows = rows; w->cols = cols; w->fmt = fmt; w->u = 0; w->z = 0;
    if (cols % 32) return 0;
    if (fmt == WF_Q8) {
        w->q = take(r, (size_t)rows * cols);
        w->s = take(r, (size_t)rows * 4);
        return w->q && w->s;
    }
    size_t ng = (size_t)rows * (cols / 32);
    w->q = take(r, (size_t)rows * cols / 2);
    w->s = take(r, (size_t)rows * 4);
    w->u = take(r, ng);
    w->z = take(r, ng);
    return w->q && w->s && w->u && w->z;
}

int llm_parse(llm_model_t *m, const uint8_t *blob, size_t len) {
    if (llm_addr) blob = llm_addr(0, 128);
    if (!blob) return -1;
    const uint32_t *h = (const uint32_t *)blob;
    if (len < 128 || memcmp(blob, "ESP3", 4) != 0 || h[1] != 3) return -1;
    memset(m, 0, sizeof(*m));
    llm_cfg_t *c = &m->c;
    c->dim = h[2]; c->hidden = h[3]; c->L = h[4]; c->nh = h[5]; c->nkv = h[6];
    c->vocab = h[7]; c->seq = h[8];
    int wf = h[9], has4 = h[11];
    m->blob_size = h[13]; m->n_tri = h[14];
    c->hs = c->dim / c->nh; c->kv = c->dim * c->nkv / c->nh;
    if (c->L > LLM_MAX_LAYERS || m->blob_size > len || c->hs > 64) return -2;
    rd_t r = { blob, 128, m->blob_size };
    int D = c->dim, kv = c->kv, H = c->hidden;
    int ok = (m->vocab_map = take(&r, c->vocab * 2)) != 0;
    ok &= take_mat(&r, &m->emb, c->vocab, D, WF_Q8);
    if (has4) ok &= take_mat(&r, &m->emb4, c->vocab, D, WF_Q4);
    ok &= (m->rms_final = take(&r, D * 4)) != 0;
    for (int l = 0; l < c->L; l++) {
        llm_layer_t *ly = &m->layer[l];
        ok &= (ly->rms_att = take(&r, D * 4)) != 0;
        ok &= (ly->rms_ffn = take(&r, D * 4)) != 0;
        ok &= take_mat(&r, &ly->qkv, D + 2 * kv, D, wf);
        ok &= take_mat(&r, &ly->wo, D, D, wf);
        ok &= take_mat(&r, &ly->w13, 2 * H, D, wf);
        ok &= take_mat(&r, &ly->w2, D, H, wf);
    }
    ok &= (m->rope = take(&r, (size_t)c->seq * c->hs * 4)) != 0;
    if (m->n_tri) {
        ok &= (m->tri_key = take(&r, (size_t)m->n_tri * 4)) != 0;
        ok &= (m->tri_val = take(&r, (size_t)m->n_tri * 2)) != 0;
        ok &= (m->bi_val = take(&r, (size_t)c->vocab * 2)) != 0;
    }
    return ok ? 0 : -3;
}

// ───────────────────────── state ─────────────────────────

int llm_state_init(llm_state_t *s, const llm_model_t *m, int kv_seq, int kv_split) {
    const llm_cfg_t *c = &m->c;
    memset(s, 0, sizeof(*s));
    s->m = m;
    if (kv_seq <= 0 || kv_seq > c->seq) kv_seq = c->seq;
    if (kv_split > kv_seq) kv_split = kv_seq;
    if (!llm_alloc_slow) kv_split = kv_seq;
    s->kv_seq = kv_seq; s->kv_split = kv_split;
    int D = c->dim, H = c->hidden, kv = c->kv, B = LLM_MAX_BATCH;
    int maxin = D > H ? D : H;
    s->xq_stride = (maxin + 15) & ~15;
#define A(ptr, n) if (!((ptr) = llm_alloc(n))) return -1
#define AS(ptr, n) if (!((ptr) = llm_alloc_slow(n))) return -1
    A(s->x, B * D * 4); A(s->xb, B * D * 4); A(s->att_out, B * D * 4);
    A(s->qkv, B * (D + 2 * kv) * 4); A(s->h13, B * 2 * H * 4); A(s->hh, B * H * 4);
    A(s->xq, B * s->xq_stride);
    A(s->att[0], B * kv_seq * 4); A(s->att[1], B * kv_seq * 4);
    A(s->gacc[0], 64 * 32 * 4); A(s->gacc[1], 64 * 32 * 4);
    A(s->xsum, B * (s->xq_stride / 32 + 1) * 4);
    A(s->ks, c->L * kv_seq * 4); A(s->vs, c->L * kv_seq * 4);
    if (kv_split) { A(s->k_lo, (size_t)c->L * kv_split * kv); A(s->v_lo, (size_t)c->L * kv_split * kv); }
    if (kv_seq > kv_split) {
        AS(s->k_hi, (size_t)c->L * (kv_seq - kv_split) * kv); AS(s->v_hi, (size_t)c->L * (kv_seq - kv_split) * kv);
    }
#undef A
#undef AS
    s->kern = KERN_SCALAR; s->ncores = 1;
    return 0;
}

static inline int8_t *kv_row(const llm_state_t *s, int8_t *lo, int8_t *hi, int l, int t) {
    int kv = s->m->c.kv, sp = s->kv_split;
    return t < sp ? lo + ((size_t)l * sp + t) * kv : hi + ((size_t)l * (s->kv_seq - sp) + (t - sp)) * kv;
}

// ───────────────────────── math helpers (same as part 1) ─────────────────────────

#define EC0 1.0000035762786865f
#define EC1 0.692969560623169f
#define EC2 0.24162131547927856f
#define EC3 0.05171773582696915f
#define EC4 0.013683983124792576f

static inline float fast_expf(float x) {
    if (x < -87.0f) x = -87.0f;
    if (x > 88.0f) x = 88.0f;
    float t = x * 1.442695041f;
    int i = (int)t;
    if (t < (float)i) i--;
    float f = t - (float)i;
    float p = EC0 + f * (EC1 + f * (EC2 + f * (EC3 + f * EC4)));
    union { int32_t i; float f; } u;
    u.i = (int32_t)((uint32_t)(i + 127) << 23);
    return p * u.f;
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int j = 0; j < n; j++) ss += x[j] * x[j];
    ss = ss / (float)n + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < n; j++) o[j] = w[j] * (ss * x[j]);
}

static inline int round_away(float v) { return (int)(v + (v >= 0.0f ? 0.5f : -0.5f)); }

static float q8_vec(int8_t *q, const float *x, int n) {
    float amax = 0.0f;
    for (int j = 0; j < n; j++) { float a = fabsf(x[j]); if (a > amax) amax = a; }
    if (amax == 0.0f) { memset(q, 0, n); return 1.0f; }
    float inv = 127.0f / amax, s = amax / 127.0f;
    for (int j = 0; j < n; j++) {
        int v = round_away(x[j] * inv);
        q[j] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
    return s;
}

static void rope(float *v, int nheads, int hs, const float *cs) {
    for (int h = 0; h < nheads; h++)
        for (int i = 0; i < hs / 2; i++) {
            float *p = v + h * hs + 2 * i;
            float c = cs[2 * i], s = cs[2 * i + 1], a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        }
}

// ───────────────────────── kernels ─────────────────────────

// int8 rows: acc[r] = sum_j W[r][j] x[j]
static HOT void dot_q8(const uint8_t *W, const int8_t *x, int32_t *acc, int rows, int n, int pie) {
#ifdef LLM_HAVE_PIE
    if (pie) { mv_s8_pie((const int8_t *)W, x, acc, rows, n >> 4); return; }
#endif
    (void)pie;
    for (int r = 0; r < rows; r++) {
        const int8_t *w = (const int8_t *)W + (size_t)r * n;
        int32_t a = 0;
        for (int j = 0; j < n; j += 4)
            a += w[j] * x[j] + w[j + 1] * x[j + 1] + w[j + 2] * x[j + 2] + w[j + 3] * x[j + 3];
        acc[r] = a;
    }
}

// int4 blocks: gacc[r*ng+g] = sum over 32-group g of (16*q) * x
static HOT void dot_q4(const uint8_t *W, const int8_t *x, int32_t *gacc, int rows, int n, int pie) {
    int nblk = n >> 5;
#ifdef LLM_HAVE_PIE
    if (pie) { mv_q4_pie(W, x, gacc, rows, nblk, 1); return; }
#endif
    (void)pie;
    for (int r = 0; r < rows; r++) {
        const uint8_t *w = W + (size_t)r * (n >> 1);
        const int8_t *xx = x;
        for (int g = 0; g < nblk; g++, w += 16, xx += 32) {
            int32_t a = 0;
            for (int j = 0; j < 16; j++) {
                int lo = (int8_t)(uint8_t)(w[j] << 4), hi = (int8_t)(w[j] & 0xF0);
                a += lo * xx[j] + hi * xx[16 + j];
            }
            *gacc++ = a;
        }
    }
}

// one output value of a Q4 row from its group sums: exact integer math, one float conversion
static inline float q4_value(const qmat_t *W, int r, const int32_t *a, const int32_t *xs, float sx) {
    int ng = W->cols >> 5;
    const uint8_t *u = W->u + (size_t)r * ng;
    const int8_t *z = W->z + (size_t)r * ng;
    int64_t t = 0;
    for (int g = 0; g < ng; g++) t += (int64_t)(a[g] - 16 * z[g] * xs[g]) * u[g];
    return ((float)t * W->s[r]) * sx;
}

// ───────────────────────── batched matmul ─────────────────────────

static struct {
    llm_state_t *s; const qmat_t *W;
    float *out; int out_stride; int n;
    int cls;                                     // 1: keep argmax/top/second, 2: keep top LLM_CAND candidates
    int best[2][LLM_MAX_BATCH]; float top[2][LLM_MAX_BATCH], sec[2][LLM_MAX_BATCH];
    int cand[2][LLM_MAX_BATCH][LLM_CAND]; float cval[2][LLM_MAX_BATCH][LLM_CAND]; int ncand[2][LLM_MAX_BATCH];
    int l, pos;
} G;

static int chunk_rows(const qmat_t *W) {
    size_t rb = W->fmt == WF_Q8 ? (size_t)W->cols : (size_t)W->cols / 2;
    int ch = (int)(4096 / rb);                   // ~4 KB of weights per chunk: stays in the data cache
    if (ch < 1) ch = 1;
    int ng = W->fmt == WF_Q8 ? 1 : W->cols / 32;
    if (ch * ng > 64 * 32) ch = 64 * 32 / ng;
    return ch;
}

// keep the LLM_CAND best (value desc, index asc) — rows arrive in ascending index order
static inline void cand_push(int core, int i, int r, float v) {
    int n = G.ncand[core][i];
    float *cv = G.cval[core][i]; int *ci = G.cand[core][i];
    if (n == LLM_CAND && !(v > cv[n - 1])) return;
    int j = n < LLM_CAND ? n++ : n - 1;
    while (j > 0 && v > cv[j - 1]) { cv[j] = cv[j - 1]; ci[j] = ci[j - 1]; j--; }
    cv[j] = v; ci[j] = r;
    G.ncand[core][i] = n;
}

static HOT void mm_rows(int r0, int r1) {
    llm_state_t *s = G.s;
    const qmat_t *W = G.W;
    int n = W->cols, pie = s->kern == KERN_PIE, core = r0 == 0 ? 0 : 1;
    int ng = W->fmt == WF_Q8 ? 1 : n / 32, xstride = s->xq_stride / 32 + 1;
    size_t rb = W->fmt == WF_Q8 ? (size_t)n : (size_t)n / 2;
    int ch = chunk_rows(W);
    int32_t *ga = s->gacc[core];
    for (int i = 0; i < G.n; i++) { G.best[core][i] = -1; G.top[core][i] = -1e30f; G.sec[core][i] = -1e30f; G.ncand[core][i] = 0; }
    for (int c0 = r0; c0 < r1; c0 += ch) {
        int c1 = c0 + ch < r1 ? c0 + ch : r1;
        const uint8_t *wq = W->q + (size_t)c0 * rb;
        (void)ng;
        if (llm_prefetch && core == 0) llm_prefetch(wq + (size_t)(c1 - c0) * rb, (size_t)ch * rb);
        for (int i = 0; i < G.n; i++) {
            const int8_t *xq = s->xq + i * s->xq_stride;
            if (W->fmt == WF_Q8) dot_q8(wq, xq, ga, c1 - c0, n, pie);
            else dot_q4(wq, xq, ga, c1 - c0, n, pie);
            float sx = s->sx[i];
            const int32_t *xs = s->xsum + i * xstride;
            for (int r = c0; r < c1; r++) {
                float v;
                if (W->fmt == WF_Q8) v = ((float)ga[r - c0] * W->s[r]) * sx;
                else v = q4_value(W, r, ga + (size_t)(r - c0) * ng, xs, sx);
                if (G.cls == 2) cand_push(core, i, r, v);
                else if (G.cls) {
                    if (v > G.top[core][i]) { G.sec[core][i] = G.top[core][i]; G.top[core][i] = v; G.best[core][i] = r; }
                    else if (v > G.sec[core][i]) G.sec[core][i] = v;
                } else G.out[(size_t)i * G.out_stride + r] = v;
            }
        }
    }
}

static void matmul(llm_state_t *s, float *out, int out_stride, const qmat_t *W, const float *x, int x_stride, int n, int cls) {
    uint32_t t0 = cyc();
    int xstride = s->xq_stride / 32 + 1;
    for (int i = 0; i < n; i++) {
        int8_t *q = s->xq + i * s->xq_stride;
        s->sx[i] = q8_vec(q, x + (size_t)i * x_stride, W->cols);
        if (W->fmt == WF_Q4)
            for (int g = 0; g < W->cols / 32; g++) {
                int32_t a = 0;
                for (int j = 0; j < 32; j++) a += q[g * 32 + j];
                s->xsum[i * xstride + g] = a;
            }
    }
    G.s = s; G.W = W; G.out = out; G.out_stride = out_stride; G.n = n; G.cls = cls;
    if (s->ncores > 1 && llm_parallel) llm_parallel(mm_rows, W->rows);
    else mm_rows(0, W->rows);
    s->bytes_w += llm_mat_bytes(W);
    uint32_t dt = cyc() - t0;
    if (cls) s->cyc_cls += dt; else s->cyc_mm += dt;
}

// ───────────────────────── attention ─────────────────────────

// Batched attention: every int8 K/V row is read once per pass and used by all tokens of the batch
// (vector i sits at position pos+i and sees rows 0..pos+i). Per vector the float operations are
// exactly those of a one-token pass, in the same order.
static HOT void attn_heads(int h0, int h1) {
    llm_state_t *s = G.s;
    const llm_cfg_t *c = &s->m->c;
    int hs = c->hs, kvd = c->kv, rep = c->nh / c->nkv, l = G.l, D = c->dim, n = G.n, Q = D + 2 * kvd;
    float *att = s->att[h0 == 0 ? 0 : 1];
    const float *ks = s->ks + (size_t)l * s->kv_seq, *vs = s->vs + (size_t)l * s->kv_seq;
    float isq = 1.0f / sqrtf((float)hs);
    float qh[LLM_MAX_BATCH][64], mx[LLM_MAX_BATCH], inv[LLM_MAX_BATCH];
    int last = G.pos + n - 1;
    for (int h = h0; h < h1; h++) {
        int kh = h / rep;
        for (int i = 0; i < n; i++) {
            const float *qv = s->qkv + (size_t)i * Q + h * hs;
            for (int d = 0; d < hs; d++) qh[i][d] = qv[d] * isq;
            mx[i] = -1e30f;
        }
        for (int t = 0; t <= last; t++) {
            const int8_t *k = kv_row(s, s->k_lo, s->k_hi, l, t) + kh * hs;
            float kt = ks[t];
            for (int i = (t > G.pos ? t - G.pos : 0); i < n; i++) {
                float a = 0.0f;
                for (int d = 0; d < hs; d++) a += (float)k[d] * qh[i][d];
                a = a * kt;
                att[i * s->kv_seq + t] = a;
                if (a > mx[i]) mx[i] = a;
            }
        }
        for (int i = 0; i < n; i++) {
            float *a = att + i * s->kv_seq, sum = 0.0f;
            for (int t = 0; t <= G.pos + i; t++) { a[t] = fast_expf(a[t] - mx[i]); sum += a[t]; }
            inv[i] = 1.0f / sum;
            float *o = s->att_out + (size_t)i * D + h * hs;
            for (int d = 0; d < hs; d++) o[d] = 0.0f;
        }
        for (int t = 0; t <= last; t++) {
            const int8_t *v = kv_row(s, s->v_lo, s->v_hi, l, t) + kh * hs;
            float vt = vs[t];
            for (int i = (t > G.pos ? t - G.pos : 0); i < n; i++) {
                float w = (att[i * s->kv_seq + t] * inv[i]) * vt;
                float *o = s->att_out + (size_t)i * D + h * hs;
                for (int d = 0; d < hs; d++) o[d] += w * (float)v[d];
            }
        }
    }
}

// ───────────────────────── two-stage classifier ─────────────────────────
// stage 1: 4-bit copy of the embedding scores every row, keeps LLM_CAND candidates per token;
// stage 2: exact int8 logits for the candidates only. Reads ~half the bytes of the int8 classifier.
static void classify2(llm_state_t *s, int n) {
    const llm_model_t *m = s->m;
    int D = m->c.dim;
    matmul(s, 0, 0, &m->emb4, s->xb, D, n, 2);
    uint32_t t0 = cyc();
    int two = s->ncores > 1 && llm_parallel;
    for (int i = 0; i < n; i++) {
        // merge both cores' lists (core 0 holds the lower indices)
        int ci[2 * LLM_CAND]; float cv[2 * LLM_CAND]; int nc = 0;
        for (int core = 0; core < (two ? 2 : 1); core++)
            for (int j = 0; j < G.ncand[core][i]; j++) {
                float v = G.cval[core][i][j]; int r = G.cand[core][i][j];
                int k = nc++;
                while (k > 0 && (v > cv[k - 1] || (v == cv[k - 1] && r < ci[k - 1]))) { cv[k] = cv[k - 1]; ci[k] = ci[k - 1]; k--; }
                cv[k] = v; ci[k] = r;
            }
        if (nc > LLM_CAND) nc = LLM_CAND;
        // exact rescoring in ascending row order
        for (int a = 1; a < nc; a++) { int r = ci[a], b = a; while (b > 0 && ci[b - 1] > r) { ci[b] = ci[b - 1]; b--; } ci[b] = r; }
        const int8_t *xq = s->xq + i * s->xq_stride;
        float sx = s->sx[i], b1 = -1e30f, b2 = -1e30f; int best = -1;
        for (int a = 0; a < nc; a++) {
            int r = ci[a]; int32_t acc;
            dot_q8(m->emb.q + (size_t)r * D, xq, &acc, 1, D, s->kern == KERN_PIE);
            float v = ((float)acc * m->emb.s[r]) * sx;
            if (v > b1) { b2 = b1; b1 = v; best = r; } else if (v > b2) b2 = v;
        }
        s->best[i] = best; s->top[i] = b1; s->second[i] = b2;
    }
    s->bytes_w += (size_t)n * LLM_CAND * (D + 4);
    s->cyc_cls += cyc() - t0;
}

// ───────────────────────── forward ─────────────────────────

void llm_forward(llm_state_t *s, const int *toks, int n, int pos) {
    const llm_model_t *m = s->m;
    const llm_cfg_t *c = &m->c;
    int D = c->dim, kvd = c->kv, H = c->hidden, Q = D + 2 * kvd;
    uint32_t t_start = cyc();
    s->bytes_w = 0;
    for (int i = 0; i < n; i++) {
        float *x = s->x + (size_t)i * D;
        const int8_t *e = (const int8_t *)m->emb.q + (size_t)toks[i] * D;
        float es = m->emb.s[toks[i]];
        for (int j = 0; j < D; j++) x[j] = (float)e[j] * es;
    }
    for (int l = 0; l < c->L; l++) {
        const llm_layer_t *ly = &m->layer[l];
        for (int i = 0; i < n; i++) rmsnorm(s->xb + (size_t)i * D, s->x + (size_t)i * D, ly->rms_att, D);
        matmul(s, s->qkv, Q, &ly->qkv, s->xb, D, n, 0);
        for (int i = 0; i < n; i++) {
            float *q = s->qkv + (size_t)i * Q, *k = q + D, *v = q + D + kvd;
            const float *cs = m->rope + (size_t)(pos + i) * c->hs;
            rope(q, c->nh, c->hs, cs);
            rope(k, c->nkv, c->hs, cs);
            size_t slot = (size_t)l * s->kv_seq + pos + i;
            s->ks[slot] = q8_vec(kv_row(s, s->k_lo, s->k_hi, l, pos + i), k, kvd);
            s->vs[slot] = q8_vec(kv_row(s, s->v_lo, s->v_hi, l, pos + i), v, kvd);
        }
        uint32_t t0 = cyc();
        G.s = s; G.l = l; G.pos = pos; G.n = n;
        if (s->ncores > 1 && llm_parallel) llm_parallel(attn_heads, c->nh);
        else attn_heads(0, c->nh);
        s->cyc_att += cyc() - t0;

        matmul(s, s->xb, D, &ly->wo, s->att_out, D, n, 0);
        for (int i = 0; i < n; i++) {
            float *x = s->x + (size_t)i * D, *xb = s->xb + (size_t)i * D;
            for (int j = 0; j < D; j++) x[j] = x[j] + xb[j];
        }
        for (int i = 0; i < n; i++) rmsnorm(s->xb + (size_t)i * D, s->x + (size_t)i * D, ly->rms_ffn, D);
        matmul(s, s->h13, 2 * H, &ly->w13, s->xb, D, n, 0);
        for (int i = 0; i < n; i++) {
            const float *h13 = s->h13 + (size_t)i * 2 * H;
            float *hh = s->hh + (size_t)i * H;
            for (int j = 0; j < H; j++) {
                float h1 = h13[j], h3 = h13[H + j];
                hh[j] = (h1 / (1.0f + fast_expf(-h1))) * h3;
            }
        }
        matmul(s, s->xb, D, &ly->w2, s->hh, H, n, 0);
        for (int i = 0; i < n; i++) {
            float *x = s->x + (size_t)i * D, *xb = s->xb + (size_t)i * D;
            for (int j = 0; j < D; j++) x[j] = x[j] + xb[j];
        }
    }
    for (int i = 0; i < n; i++) rmsnorm(s->xb + (size_t)i * D, s->x + (size_t)i * D, m->rms_final, D);
    if (s->cls2 && m->emb4.q) { classify2(s, n); s->cyc_all += cyc() - t_start; return; }
    matmul(s, 0, 0, &m->emb, s->xb, D, n, 1);
    for (int i = 0; i < n; i++) {
        int b0 = G.best[0][i], b1 = G.best[1][i];
        float t0 = G.top[0][i], t1 = G.top[1][i], s0 = G.sec[0][i], s1 = G.sec[1][i];
        if (s->ncores < 2 || !llm_parallel || b1 < 0) { s->best[i] = b0; s->top[i] = t0; s->second[i] = s0; continue; }
        if (t0 >= t1) { s->best[i] = b0; s->top[i] = t0; s->second[i] = s0 > t1 ? s0 : t1; }
        else { s->best[i] = b1; s->top[i] = t1; s->second[i] = s1 > t0 ? s1 : t0; }
    }
    s->cyc_all += cyc() - t_start;
}

// ───────────────────────── self-speculation: draft tokens from n-gram statistics ─────────────────────────

static int tri_lookup(const llm_model_t *m, int a, int b) {
    if (!m->n_tri) return -1;
    uint32_t key = (uint32_t)a * (uint32_t)m->c.vocab + (uint32_t)b;
    int lo = 0, hi = m->n_tri - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t k = m->tri_key[mid];
        if (k == key) return m->tri_val[mid];
        if (k < key) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

int llm_draft(const llm_model_t *m, const int *hist, int len, int k, int *out) {
    static int h[600];
    if (len > 590 - k) { hist += len - (590 - k); len = 590 - k; }
    memcpy(h, hist, len * sizeof(int));
    int n = 0;
    for (; n < k; n++) {
        int nxt = -1;
        if (len >= 3) {
            int a = h[len - 2], b = h[len - 1];
            for (int j = len - 3; j > 0; j--)
                if (h[j - 1] == a && h[j] == b) { nxt = h[j + 1]; break; }
        }
        if (nxt < 0 && len >= 2) nxt = tri_lookup(m, h[len - 2], h[len - 1]);
        if (nxt < 0 && m->bi_val && len >= 1 && m->bi_val[h[len - 1]] != 0xFFFF) nxt = m->bi_val[h[len - 1]];
        if (nxt < 0) break;
        out[n] = nxt; h[len++] = nxt;
    }
    return n;
}
