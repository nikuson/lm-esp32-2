// host3.c — run the device engine (llm3.c) on the PC: greedy or self-speculative decoding.
// usage: host3 model.esp3 steps k_spec id,id,...   -> prints "tok top second" per generated token
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../firmware/main/llm3.h"

static void *al(size_t n) { void *p = NULL; if (posix_memalign(&p, 16, n)) return NULL; memset(p, 0, n); return p; }

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage\n"); return 1; }
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *blob = al(len); if (fread(blob, 1, len, f) != (size_t)len) return 1; fclose(f);
    llm_model_t m; int rc = llm_parse(&m, blob, len); if (rc) { fprintf(stderr, "parse %d\n", rc); return 1; }
    llm_alloc = al; llm_alloc_slow = al;
    int steps = atoi(argv[2]), k = atoi(argv[3]);
    int ids[600], n = 0; char *p = argv[4];
    while (*p && n < 512) { ids[n++] = strtol(p, &p, 10); if (*p == ',') p++; else break; }
    llm_state_t s; if (llm_state_init(&s, &m, m.c.seq, 64)) { fprintf(stderr, "state\n"); return 1; }
    s.cls2 = argc > 5 ? atoi(argv[5]) : 0;
    int hist[600]; memcpy(hist, ids, n * sizeof(int)); int len_h = n;
    // prefill: all prompt tokens but the last, in batches
    int pos = 0;
    while (pos < n - 1) {
        int b = n - 1 - pos; if (b > LLM_MAX_BATCH) b = LLM_MAX_BATCH;
        llm_forward(&s, ids + pos, b, pos); pos += b;
    }
    int last = ids[n - 1], made = 0, passes = 0;
    while (made < steps && pos < m.c.seq) {
        int batch[LLM_MAX_BATCH]; batch[0] = last;
        int nd = 0;
        if (k > 0) { int lim = k; if (pos + 1 + lim > m.c.seq) lim = m.c.seq - pos - 1; if (lim < 0) lim = 0; nd = llm_draft(&m, hist, len_h, lim, batch + 1); }
        llm_forward(&s, batch, 1 + nd, pos); passes++;
        int acc = 0;
        while (acc < nd && batch[1 + acc] == s.best[acc]) acc++;
        for (int i = 0; i <= acc && made < steps; i++) {
            printf("%d %.9g %.9g\n", s.best[i], (double)s.top[i], (double)s.second[i]);
            hist[len_h++] = s.best[i]; made++;
        }
        pos += acc + 1; last = s.best[acc];
        if (last == 1) break;
    }
    fprintf(stderr, "passes=%d tokens=%d\n", passes, made);
    return 0;
}
