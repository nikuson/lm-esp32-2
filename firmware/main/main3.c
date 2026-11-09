// main3.c — ESP32-S3 side of part 2: stories15M streamed from flash.
//
// Protocol (lines over UART0 or USB-Serial-JTAG):
//   INFO                                     -> @INFO key=value ...
//   GEN <flags> <steps> <verify> <id,id,..>  -> @T <tok> [<top> <second>] ... @END key=value ...
//        flags: k1|k2 scalar|PIE, c1|c2 cores, a0|a1|a2 none|cache autoload|manual preload,
//               x1 two-stage classifier, w1 hot weights in SRAM, p<k> self-speculation with k drafts
//   MEM                                      -> @MEM sram=.. psram=.. flash=.. flash_autoload=..  (MB/s)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "esp_log.h"
#include "esp_flash.h"
#include "spi_flash_mmap.h"
#include "esp_mmu_map.h"
#include "esp_err.h"
#include "esp_rom_sys.h"
#include "esp32s3/rom/cache.h"
#include "soc/extmem_reg.h"
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"
#include "sdkconfig.h"
#include "llm3.h"

static const char *TAG = "llm15";

#ifndef FW_NAME
#define FW_NAME "custom"
#endif

void mem_read_pie(const void *p, int n64);

// ───────────── memory ─────────────
static void *alloc_int(size_t n) { return heap_caps_aligned_calloc(16, 1, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }
static void *alloc_psram(size_t n) { return heap_caps_aligned_calloc(16, 1, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT); }
static uint32_t cycles(void) { return esp_cpu_get_cycle_count(); }

// ───────────── second core ─────────────
static volatile llm_job_fn job_fn;
static volatile int job_a, job_b, worker_spin;
static volatile uint32_t job_req, job_done;
static TaskHandle_t worker;

static void worker_task(void *arg) {
    uint32_t seen = job_req;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        while (worker_spin) {
            if (job_req != seen) {
                seen = job_req;
                job_fn(job_a, job_b);
                __sync_synchronize();
                job_done = seen;
            }
        }
    }
}

static void IRAM_ATTR run_parallel(llm_job_fn fn, int n) {
    int h = n / 2;
    job_fn = fn; job_a = h; job_b = n;
    __sync_synchronize();
    job_req = job_req + 1;
    fn(0, h);
    while (job_done != job_req) { }
    __sync_synchronize();
}

// ───────────── serial ─────────────
static int usj_ok, out_port;

static void out(const char *fmt, ...) {
    char buf[320];
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > (int)sizeof buf - 1) n = sizeof buf - 1;
    if (out_port == 1 && usj_ok) usb_serial_jtag_write_bytes(buf, n, pdMS_TO_TICKS(500));
    else uart_write_bytes(UART_NUM_0, buf, n);
}

// ───────────── model ─────────────
static llm_model_t m_flash, m_hot;     // m_hot: same model, some matrices copied to SRAM
static llm_state_t st;
static const uint8_t *map;
static uint32_t hdr_copy[32];
static size_t map_size;
// the model may be mapped in several pieces when the MMU has no contiguous window big enough
#define MAXSEG 8
#ifndef FIRST_PIECES
#define FIRST_PIECES 1
#endif
static struct { size_t off, len; const uint8_t *p; } seg[MAXSEG];
static int nseg;
static esp_partition_mmap_handle_t segh[MAXSEG];
static size_t sec_off[64 * 8]; static int nsec;
static const void *rec_addr(size_t off, size_t len) {            // dry run: remember section starts
    (void)len; if (nsec < (int)(sizeof sec_off / sizeof sec_off[0])) sec_off[nsec++] = off;
    static uint8_t dummy[128] __attribute__((aligned(16)));
    return off == 0 ? (const void *)hdr_copy : (const void *)dummy;
}
static const void *seg_addr(size_t off, size_t len) {
    for (int i = 0; i < nseg; i++)
        if (off >= seg[i].off && off + len <= seg[i].off + seg[i].len) return seg[i].p + (off - seg[i].off);
    return NULL;
}
static int in_model(const void *p) {
    for (int i = 0; i < nseg; i++) if ((const uint8_t *)p >= seg[i].p && (const uint8_t *)p < seg[i].p + seg[i].len) return 1;
    return 0;
}
static size_t hot_bytes;
static uint8_t *psram_probe;

// copy whole matrices into free SRAM (largest first) — every byte here is a byte not read from flash
static void make_hot(size_t budget) {
    m_hot = m_flash;
    const llm_cfg_t *c = &m_flash.c;
    for (;;) {
        qmat_t *best = NULL; size_t bb = 0;
        for (int l = 0; l < c->L; l++) {
            qmat_t *ws[4] = { &m_hot.layer[l].qkv, &m_hot.layer[l].wo, &m_hot.layer[l].w13, &m_hot.layer[l].w2 };
            for (int i = 0; i < 4; i++) {
                if (!in_model(ws[i]->q)) continue;  // already hot
                size_t b = llm_mat_bytes(ws[i]);
                if (b <= budget && b > bb) { bb = b; best = ws[i]; }
            }
        }
        if (!best) break;
        size_t qb = best->fmt == WF_Q8 ? (size_t)best->rows * best->cols : (size_t)best->rows * best->cols / 2;
        size_t ng = best->fmt == WF_Q8 ? 0 : (size_t)best->rows * (best->cols / 32);
        uint8_t *p = alloc_int(bb + 64);
        if (!p) break;
        uint8_t *q = p;
        memcpy(q, best->q, qb); best->q = q; q += (qb + 15) & ~15;
        memcpy(q, best->s, best->rows * 4); best->s = (const float *)q; q += best->rows * 4;
        if (ng) { memcpy(q, best->u, ng); best->u = q; q += ng; memcpy(q, best->z, ng); best->z = (const int8_t *)q; }
        budget -= bb; hot_bytes += bb;
    }
}

// ───────────── cache tricks ─────────────
static int pl_active; static uint32_t pl_auto;

static void prefetch_manual(const void *next, size_t n) {
    // ROM cache preload: the cache fetches the next chunk while the CPU works on the current one
    if (pl_active) {
        int to = 200000;
        while (!Cache_DCache_Preload_Done() && --to) { }
        Cache_End_DCache_Preload(pl_auto);
        pl_active = 0;
    }
    const uint8_t *p = next;
    if (!in_model(p) || !in_model(p + n - 1)) return;
    if (n > 8192) n = 8192;
    pl_auto = Cache_Start_DCache_Preload((uint32_t)p, n, 0);
    pl_active = 1;
}

static void prefetch_end(void) {
    if (pl_active) {
        int to = 200000;
        while (!Cache_DCache_Preload_Done() && --to) { }
        Cache_End_DCache_Preload(pl_auto);
        pl_active = 0;
    }
}

static void autoload_on(void) {
    struct autoload_config cfg = { .ena = 1, .order = 0, .trigger = CACHE_AUTOLOAD_MISS_TRIGGER, .size = 3 };
    for (int i = 0; i < nseg && i < 2; i++) {
        struct autoload_region_config reg = { .region = i, .ena = 1, .addr = (uint32_t)seg[i].p, .size = seg[i].len };
        Cache_Config_DCache_Region_Autoload(&reg);
    }
    Cache_Config_DCache_Autoload(&cfg);
    Cache_Enable_DCache_Autoload();
}

static void autoload_off(void) { Cache_Disable_DCache_Autoload(); }

// ───────────── commands ─────────────
typedef struct { int16_t tok; float top, sec; } outtok_t;
static outtok_t *obuf;
static int hist[600], ids[520];

static int flag(const char *f, char k, int def) {
    for (const char *p = f; *p; p++) if (*p == k && p[1] >= '0' && p[1] <= '9') return atoi(p + 1);
    return def;
}

static uint32_t flash_mhz(void) {
#if defined(CONFIG_ESPTOOLPY_FLASHFREQ_120M)
    return 120;
#elif defined(CONFIG_ESPTOOLPY_FLASHFREQ_80M)
    return 80;
#else
    return 40;
#endif
}

static void cmd_info(void) {
    const llm_cfg_t *c = &m_flash.c;
    uint32_t fsz = 0; esp_flash_get_size(NULL, &fsz);
    out("@INFO chip=esp32s3 fw=%s cpu_mhz=%lu flash_mb=%lu flash_mhz=%lu dcache_kb=%d dline=%d "
        "dim=%d layers=%d vocab=%d seq=%d kv_seq=%d blob_bytes=%u hot_kb=%u free_int_kb=%u psram_kb=%u autoload_reg=%08lx segs=%d\n",
        FW_NAME, (unsigned long)esp_rom_get_cpu_ticks_per_us(), (unsigned long)(fsz >> 20), (unsigned long)flash_mhz(),
        CONFIG_ESP32S3_DATA_CACHE_SIZE / 1024, CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE,
        c->dim, c->L, c->vocab, c->seq, st.kv_seq, (unsigned)m_flash.blob_size, (unsigned)(hot_bytes / 1024),
        (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
        (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024),
        (unsigned long)REG_READ(EXTMEM_DCACHE_AUTOLOAD_CTRL_REG), nseg);
}

static float mbps(const void *p, size_t n, int reps) {
    mem_read_pie(p, 64);                       // warm the TLB
    int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < reps; r++) mem_read_pie(p, (int)(n / 64));
    int64_t dt = esp_timer_get_time() - t0;
    return (float)((double)n * reps / (double)dt);   // bytes/us == MB/s
}

static void cmd_mem(void) {
    size_t sn = 32 * 1024;
    void *sram = NULL;
    while (sn >= 4096 && !(sram = alloc_int(sn))) sn /= 2;
    float s = sram ? mbps(sram, sn, (int)(2 * 1024 * 1024 / sn)) : 0;
    free(sram);
    float ps = psram_probe ? mbps(psram_probe, 256 * 1024, 4) : 0;
    size_t fl = seg[0].len > 2 * 1024 * 1024 ? 2 * 1024 * 1024 : seg[0].len & ~(size_t)63;
    float f1 = mbps(map, fl, 1);
    autoload_on();
    const uint8_t *m2 = nseg > 1 && seg[1].len >= fl ? seg[1].p : seg[0].p + (seg[0].len >= 2 * fl ? fl : 0);
    float f2 = mbps(m2, fl, 1);
    autoload_off();
    out("@MEM sram=%.1f psram=%.1f flash=%.1f flash_autoload=%.1f\n", s, ps, f1, f2);
}

static void cmd_gen(char *args) {
    char flags[32] = "";
    char *p = args;
    int k = 0;
    while (*p && *p != ' ' && k < 31) flags[k++] = *p++;
    flags[k] = 0;
    int steps = strtol(p, &p, 10), verify = strtol(p, &p, 10), n = 0;
    while (*p == ' ') p++;
    while (*p && n < 512) { ids[n++] = strtol(p, &p, 10); if (*p == ',') p++; else break; }
    if (n < 1) { out("!ERR no prompt\n"); return; }
    int kern = flag(flags, 'k', 2), cores = flag(flags, 'c', 1), pf = flag(flags, 'a', 0);
    int cls2 = flag(flags, 'x', 0), hot = flag(flags, 'w', 0), spec = flag(flags, 'p', 0);
    if (spec > LLM_MAX_BATCH - 1) spec = LLM_MAX_BATCH - 1;
    if (cls2 && !m_flash.emb4.q) { out("!ERR this model has no 4-bit classifier copy\n"); return; }
    if (n + steps > st.kv_seq) steps = st.kv_seq - n;

    st.m = hot ? &m_hot : &m_flash;
    st.kern = kern == 1 ? KERN_SCALAR : KERN_PIE;
    st.ncores = cores; st.cls2 = cls2;
    st.cyc_mm = st.cyc_cls = st.cyc_att = st.cyc_all = 0;
    llm_prefetch = pf == 2 ? prefetch_manual : NULL;
    if (pf == 1) autoload_on();
    if (cores > 1) { worker_spin = 1; xTaskNotifyGive(worker); vTaskDelay(1); }

    // prefill: all prompt tokens but the last, in batches (not part of the generation timing)
    int64_t t0 = esp_timer_get_time();
    int pos = 0;
    while (pos < n - 1) {
        int b = n - 1 - pos; if (b > LLM_MAX_BATCH) b = LLM_MAX_BATCH;
        llm_forward(&st, ids + pos, b, pos); pos += b;
    }
    int64_t t1 = esp_timer_get_time();
    memcpy(hist, ids, n * sizeof(int));
    int len = n, last = ids[n - 1], made = 0, passes = 0;
    uint64_t bytes = 0;
    st.cyc_mm = st.cyc_cls = st.cyc_att = st.cyc_all = 0;
    while (made < steps && pos < st.kv_seq) {
        int batch[LLM_MAX_BATCH]; batch[0] = last;
        int nd = 0;
        if (spec) {
            int lim = spec;
            if (pos + 1 + lim > st.kv_seq) lim = st.kv_seq - pos - 1;
            if (lim > 0) nd = llm_draft(&m_flash, hist, len, lim, batch + 1);
        }
        llm_forward(&st, batch, 1 + nd, pos);
        passes++; bytes += st.bytes_w;
        int acc = 0;
        while (acc < nd && batch[1 + acc] == st.best[acc]) acc++;
        for (int i = 0; i <= acc && made < steps; i++) {
            obuf[made] = (outtok_t){ (int16_t)st.best[i], st.top[i], st.second[i] };
            hist[len++] = st.best[i]; made++;
        }
        pos += acc + 1; last = st.best[acc];
        if (last == 1) break;
    }
    int64_t t2 = esp_timer_get_time();
    worker_spin = 0;
    prefetch_end(); llm_prefetch = NULL;
    if (pf == 1) autoload_off();

    for (int i = 0; i < made; i++) {
        if (verify) out("@T %d %.9g %.9g\n", obuf[i].tok, (double)obuf[i].top, (double)obuf[i].sec);
        else out("@T %d\n", obuf[i].tok);
    }
    uint32_t all = st.cyc_all ? st.cyc_all : 1;
    unsigned mm = (unsigned)(100ull * st.cyc_mm / all), cl = (unsigned)(100ull * st.cyc_cls / all);
    unsigned at = (unsigned)(100ull * st.cyc_att / all);
    unsigned oth = mm + cl + at > 100 ? 0 : 100 - mm - cl - at;
    out("@END n=%d passes=%d us_prefill=%lld us_gen=%lld mm=%u cls=%u att=%u oth=%u kbpp=%u flags=%s\n",
        made, passes, (long long)(t1 - t0), (long long)(t2 - t1), mm, cl, at, oth,
        (unsigned)(passes ? bytes / passes / 1024 : 0), flags);
}

static void handle(char *line, int port) {
    out_port = port;
    if (!strncmp(line, "INFO", 4)) cmd_info();
    else if (!strncmp(line, "GEN ", 4)) cmd_gen(line + 4);
    else if (!strncmp(line, "MEM", 3)) cmd_mem();
    else if (line[0]) out("!ERR unknown command\n");
}

void app_main(void) {
    xTaskCreatePinnedToCore(worker_task, "core1", 3072, NULL, configMAX_PRIORITIES - 2, &worker, 1);
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);
    usb_serial_jtag_driver_config_t ucfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    ucfg.rx_buffer_size = 512; ucfg.tx_buffer_size = 1024;
    usj_ok = usb_serial_jtag_driver_install(&ucfg) == ESP_OK;

    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    esp_partition_mmap_handle_t h;
    const void *mp = NULL;
    // map only the blob; if no contiguous MMU window is big enough, map it in pieces cut at section boundaries
    uint32_t *hdr = hdr_copy;
    esp_err_t me = ESP_ERR_NOT_FOUND; int pr = -100;
    size_t want = 0;
    if (part && esp_partition_read(part, 0, hdr, 128) == ESP_OK && hdr[0] == 0x33505345u /* "ESP3" */) {
        want = hdr[13];
        nsec = 0; llm_addr = rec_addr;
        llm_model_t tmp; llm_parse(&tmp, NULL, want);           // dry run collects section offsets
        llm_addr = NULL;
        for (int pieces = FIRST_PIECES; pieces <= MAXSEG && me != ESP_OK; pieces++) {
            size_t limit = (want + pieces - 1) / pieces + 65536;
                        nseg = 0; me = ESP_OK;
            size_t start = 0;
            for (int k = 1; k <= nsec && me == ESP_OK; k++) {
                size_t cut = k < nsec ? sec_off[k] : want;            // candidate segment end
                size_t endk = k + 1 < nsec ? sec_off[k + 1] : want;   // end of section k
                if (k == nsec || (endk - start > limit && cut > start)) {
                    if (nseg == MAXSEG) { me = ESP_ERR_NO_MEM; break; }
                    size_t a0 = start & ~(size_t)0xFFFF, len = cut - a0;
                    const void *p = NULL; esp_partition_mmap_handle_t hh;
                    me = esp_partition_mmap(part, a0, (len + 0xFFFF) & ~(size_t)0xFFFF, ESP_PARTITION_MMAP_DATA, &p, &hh);
                    if (me == ESP_OK) { seg[nseg].off = a0; seg[nseg].len = len; seg[nseg].p = p; segh[nseg] = hh; nseg++; }
                    start = cut;
                }
            }
            if (me != ESP_OK) { for (int i = 0; i < nseg; i++) esp_partition_munmap(segh[i]); nseg = 0; }
        }
        if (me == ESP_OK) { llm_addr = seg_addr; pr = llm_parse(&m_flash, NULL, want); }
    }
    if (me != ESP_OK || pr != 0) {
        unsigned freep = (unsigned)spi_flash_mmap_get_free_pages(SPI_FLASH_MMAP_DATA);
        size_t maxblk = 0;
        esp_mmu_map_get_max_consecutive_free_block_size(MMU_MEM_CAP_READ | MMU_MEM_CAP_8BIT, MMU_TARGET_FLASH0, &maxblk);
        esp_mmu_map_dump_mapped_blocks(stdout);
        for (;;) {
            for (int port = 0; port < 2; port++) {
                if (port == 1 && !usj_ok) continue;
                out_port = port;
                out("!NOMODEL pieces=%d part=%d magic=%08lx want_kb=%u mmap=%s parse=%d free_mmu_pages=%u max_block_kb=%u psram_kb=%u\n",
                    nseg, part != NULL, (unsigned long)hdr[0], (unsigned)(want / 1024), esp_err_to_name(me), pr, freep, (unsigned)(maxblk / 1024),
                    (unsigned)(heap_caps_get_total_size(MALLOC_CAP_SPIRAM) / 1024));
            }
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
    }
    map = seg[0].p; map_size = m_flash.blob_size;
    llm_alloc = alloc_int; llm_cycles = cycles; llm_parallel = run_parallel;
    int has_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM) > 0;
    llm_alloc_slow = has_psram ? alloc_psram : NULL;
    if (has_psram) psram_probe = alloc_psram(256 * 1024);

    // KV cache: all of it in PSRAM if there is one, else as much context as fits in SRAM
    const llm_cfg_t *c = &m_flash.c;
    int kv_seq = c->seq, split = 0;
    if (!has_psram) {
        size_t per = (size_t)c->L * c->kv * 2 + c->L * 8;
        size_t room = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        room = room > 200 * 1024 ? room - 200 * 1024 : 0;
        kv_seq = (int)(room / per); if (kv_seq > c->seq) kv_seq = c->seq;
        if (kv_seq < 32) kv_seq = 32;
        split = kv_seq;
    }
    if (llm_state_init(&st, &m_flash, kv_seq, split) != 0) ESP_LOGE(TAG, "state alloc failed");
    obuf = heap_caps_calloc(kv_seq, sizeof(outtok_t), MALLOC_CAP_INTERNAL);

    // whatever SRAM is left (minus a margin) holds copies of whole weight matrices
    size_t fr = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    make_hot(fr > 40 * 1024 ? fr - 40 * 1024 : 0);
    fr = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
    if (fr > 60 * 1024) make_hot(fr - 40 * 1024);       // a second, smaller block may fit too

    vTaskPrioritySet(NULL, configMAX_PRIORITIES - 2);
    esp_log_level_set("*", ESP_LOG_WARN);
    out_port = 0; cmd_info();
    if (usj_ok) { out_port = 1; cmd_info(); }

    static char lb[2][3072];
    static int ln[2];
    uint8_t buf[256];
    for (;;) {
        for (int port = 0; port < 2; port++) {
            int r = port == 0 ? uart_read_bytes(UART_NUM_0, buf, sizeof buf, pdMS_TO_TICKS(5))
                              : (usj_ok ? usb_serial_jtag_read_bytes(buf, sizeof buf, pdMS_TO_TICKS(5)) : 0);
            for (int i = 0; i < r; i++) {
                char ch = buf[i];
                if (ch == '\n' || ch == '\r') {
                    lb[port][ln[port]] = 0;
                    if (ln[port]) handle(lb[port], port);
                    ln[port] = 0;
                } else if (ln[port] < (int)sizeof lb[0] - 1) lb[port][ln[port]++] = ch;
            }
        }
    }
}
