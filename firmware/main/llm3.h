// llm3.h — Llama-2 engine for stories15M-class models on ESP32-S3 (weights streamed from flash).
// Portable C core: the same file builds on the host for bit-exact checks.
#pragma once
#include <stdint.h>
#include <stddef.h>

#define LLM_MAX_LAYERS 16
#define LLM_MAX_BATCH 6          // tokens per forward pass (speculative verification / prefill)
#define LLM_CAND 16              // candidates kept by the 4-bit classifier stage

typedef enum { WF_Q8 = 0, WF_Q4 = 1 } wfmt_t;

typedef struct {
    int dim, hidden, L, nh, nkv, vocab, seq, kv, hs;
} llm_cfg_t;

// Q8: rows x cols int8, one float scale per row.
// Q4: rows x cols/2 bytes in 16-byte blocks of 32 weights (byte j: low nibble = w[j], high = w[j+16]),
//     signed nibbles q; per group of 32: zero point z (int8) and integer scale u (1..127);
//     per row: float scale s (already divided by 16, see the PIE kernel).  w = s*16 * u * (q - z)
typedef struct {
    const uint8_t *q;
    const float *s;
    const uint8_t *u;
    const int8_t *z;
    int rows, cols, fmt;
} qmat_t;

typedef struct {
    const float *rms_att, *rms_ffn;
    qmat_t qkv, wo, w13, w2;
} llm_layer_t;

typedef struct {
    llm_cfg_t c;
    qmat_t emb;                      // tied: token embedding == classifier (pruned vocabulary), int8
    qmat_t emb4;                     // optional 4-bit copy for the first stage of the classifier
    const float *rms_final;
    const float *rope;               // [seq][hs/2][2]
    llm_layer_t layer[LLM_MAX_LAYERS];
    // self-speculation tables (may be absent)
    const uint32_t *tri_key; const uint16_t *tri_val; int n_tri;
    const uint16_t *bi_val;
    const uint16_t *vocab_map;       // local id -> original tokenizer id
    size_t blob_size;
} llm_model_t;

typedef enum { KERN_SCALAR = 1, KERN_PIE = 2 } llm_kern_t;

typedef struct {
    const llm_model_t *m;
    llm_kern_t kern;
    int ncores;
    int cls2;                        // two-stage classifier: 4-bit scoring, exact int8 rescoring of top LLM_CAND
    int kv_seq;                      // context length
    int kv_split;                    // positions < kv_split live in kv_lo (SRAM), the rest in kv_hi (PSRAM)
    int8_t *k_lo, *v_lo, *k_hi, *v_hi;
    float *ks, *vs;                  // per (layer, pos) scales
    // activations, LLM_MAX_BATCH vectors each
    float *x, *xb, *qkv, *h13, *hh, *att_out;
    int8_t *xq;                      // quantized activations, stride xq_stride
    int xq_stride;
    float sx[LLM_MAX_BATCH];
    float *att[2];                   // per-core softmax scratch
    int32_t *gacc[2];                // per-core group partial sums
    int32_t *xsum;                   // per vector, per 32-group sum of the quantized activations
    // results of the last pass
    int best[LLM_MAX_BATCH];
    float top[LLM_MAX_BATCH], second[LLM_MAX_BATCH];
    // profiling (cycles)
    uint32_t cyc_mm, cyc_cls, cyc_att, cyc_all;
    uint32_t bytes_w;                // weight bytes streamed in the last pass
} llm_state_t;

typedef void (*llm_job_fn)(int a, int b);
extern void (*llm_parallel)(llm_job_fn fn, int n);
extern uint32_t (*llm_cycles)(void);
extern void *(*llm_alloc)(size_t size);          // fast memory (SRAM)
extern void *(*llm_alloc_slow)(size_t size);     // big memory (PSRAM), may be NULL
// called before each chunk of a weight matrix is consumed (prefetch hook, may be NULL)
extern void (*llm_prefetch)(const void *next, size_t bytes);
// optional: blob offset -> address (the blob may be mapped in several pieces); NULL = contiguous
extern const void *(*llm_addr)(size_t off, size_t len);

int  llm_parse(llm_model_t *m, const uint8_t *blob, size_t len);
size_t llm_mat_bytes(const qmat_t *w);
int  llm_state_init(llm_state_t *s, const llm_model_t *m, int kv_seq, int kv_split);
// forward n tokens at positions pos..pos+n-1; results in s->best/top/second
void llm_forward(llm_state_t *s, const int *toks, int n, int pos);
// draft up to k tokens continuing hist[0..len) (in-context bigram match, then trigram table, then bigram table)
int  llm_draft(const llm_model_t *m, const int *hist, int len, int k, int *out);
