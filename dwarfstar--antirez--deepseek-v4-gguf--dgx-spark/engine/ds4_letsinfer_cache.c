/* SPDX-License-Identifier: AGPL-3.0-only
 * ds4_letsinfer_cache.c - DwarfStar adapter for the Let's Infer persistent
 * prefix cache.  See ds4_letsinfer_cache.h for the contract and environment.
 *
 * Storage/tiering is the proven engine-neutral letsinfer_prefix_store (CRC'd
 * page-aligned records, atomic commits, exact-byte LRU capacity, sliding
 * TTL, bounded background writer, optional host-RAM residency + O_DIRECT
 * reads), reached through the libletsinfer_prefix_capi.so C ABI via dlopen.
 * DS4 state stays opaque to the store: one record carries three regions
 *
 *   "ds4_meta"     fixed 64-byte adapter header (validated on restore)
 *   "warm_text"    the bank's rendered warm-record text (the byte key the
 *                  server's matcher uses after a restore)
 *   "bank_payload" the exact ds4_cont_bank_save_payload() wire bytes (the
 *                  same serialized bank state the native disk KV tier
 *                  round-trips: FP8/FP4 tensors + counters + bank history)
 *
 * keyed by the admitting request's canonical prompt token IDs plus a
 * 32-byte compatibility fingerprint (payload ABI, model id, routed-expert
 * quant bits, serving ctx, layer count).  A mismatched fingerprint makes a
 * record invisible, mirroring --kv-cache-reject-different-quant.
 *
 * Everything here is fail-open: any problem logs one line and returns the
 * cache-miss result.  dlopen'ing keeps the stock build's link line and
 * binary set unchanged; without DS4_LETSINFER_CACHE=1 this file's only runtime
 * footprint is a NULL adapter pointer in the server. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ds4_letsinfer_cache.h"

#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double ac_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---- bridge ABI (must match letsinfer-bridge/capi/src/lib.rs) ---- */

#define LETSINFER_PREFIX_ABI_VERSION_EXPECTED 2u

typedef struct LetsInferPrefixStore LetsInferPrefixStore;
typedef struct LetsInferPrefixWriter LetsInferPrefixWriter;
typedef struct LetsInferPrefixReader LetsInferPrefixReader;

typedef struct {
    void *dl;
    uint32_t (*abi_version)(void);
    LetsInferPrefixStore *(*store_open)(const char *root, uint64_t capacity_bytes,
                             uint64_t ttl_seconds, uint64_t minimum_token_count,
                             uint64_t resident_capacity_bytes, int direct_reads,
                             char *err, size_t err_len);
    void (*store_close)(LetsInferPrefixStore *store);
    LetsInferPrefixWriter *(*begin_capture)(LetsInferPrefixStore *store, const uint8_t *fingerprint,
                                 const uint32_t *tokens, size_t token_count,
                                 const char *const *region_names,
                                 const uint64_t *region_byte_counts,
                                 size_t region_count, int *rejection_out);
    uint8_t *(*writer_region)(LetsInferPrefixWriter *writer, size_t index,
                              uint64_t *byte_count_out);
    int (*writer_commit_async)(LetsInferPrefixWriter *writer, int promote_resident,
                               char *err, size_t err_len);
    void (*writer_cancel)(LetsInferPrefixWriter *writer);
    LetsInferPrefixReader *(*longest_prefix)(LetsInferPrefixStore *store, const uint8_t *fingerprint,
                                  const uint32_t *tokens, size_t token_count,
                                  size_t minimum_token_count);
    size_t (*reader_token_count)(const LetsInferPrefixReader *reader);
    size_t (*reader_region_count)(const LetsInferPrefixReader *reader);
    uint64_t (*reader_region_byte_count)(const LetsInferPrefixReader *reader, size_t index);
    int (*reader_region_name)(const LetsInferPrefixReader *reader, size_t index,
                              char *name_out, size_t name_capacity);
    int (*reader_read_region)(LetsInferPrefixReader *reader, size_t index,
                              uint8_t *destination, uint64_t destination_len,
                              char *err, size_t err_len);
    const uint8_t *(*reader_region_data)(LetsInferPrefixReader *reader, size_t index,
                                         uint64_t *byte_count_out,
                                         char *err, size_t err_len);
    void (*reader_touch)(const LetsInferPrefixReader *reader);
    void (*reader_release_resident)(const LetsInferPrefixReader *reader);
    void (*reader_close)(LetsInferPrefixReader *reader);
    int (*reclaim_expired)(LetsInferPrefixStore *store, uint64_t unix_time_seconds);
    int (*stats_line)(const LetsInferPrefixStore *store, char *line_out,
                      size_t line_capacity);
} letsinfer_prefix_api;

/* ---- adapter ---- */

/* Meta region: fixed 64 bytes, little-endian, zero-padded. The v1 wire
 * values remain unchanged so records captured before the naming cleanup
 * continue to restore. */
#define LETSINFER_META_MAGIC "DSATMET1"
#define LETSINFER_META_BYTES 64u
#define LETSINFER_META_VERSION 1u

/* Fingerprint layout: 8-byte magic + six LE u32 identity fields. */
#define LETSINFER_FP_MAGIC "DS4ATLPC"

#define AC_MAX_BANKS 64

struct ds4_letsinfer_cache {
    int *note_key[AC_MAX_BANKS];
    int note_len[AC_MAX_BANKS];
    letsinfer_prefix_api api;
    LetsInferPrefixStore *store;
    uint8_t fingerprint[32];
    int min_tokens;
    bool capture;                /* DS4_LETSINFER_CACHE_CAPTURE=0 disables stores */
    bool prefix_lookup;          /* DS4_LETSINFER_CACHE_PREFIX=1 (experimental) */
    uint64_t last_reclaim_unix;
    ds4_letsinfer_log_fn log;
    void *log_ud;
};

static void ac_logf(const ds4_letsinfer_cache *ac, int level, const char *fmt, ...) {
    if (!ac || !ac->log) return;
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    ac->log(ac->log_ud, level, msg);
}

static void le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static uint32_t le_get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void le_put64(uint8_t *p, uint64_t v) {
    le_put32(p, (uint32_t)v);
    le_put32(p + 4, (uint32_t)(v >> 32));
}

static uint64_t le_get64(const uint8_t *p) {
    return (uint64_t)le_get32(p) | ((uint64_t)le_get32(p + 4) << 32);
}

static bool env_flag(const char *name, bool dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    return !(v[0] == '0' && v[1] == '\0');
}

static uint64_t env_u64(const char *name, uint64_t dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    char *end = NULL;
    const unsigned long long parsed = strtoull(v, &end, 10);
    if (!end || *end) return dflt;
    return (uint64_t)parsed;
}

#define AC_SYM(field, name) \
    do { \
        *(void **)(&api->field) = dlsym(api->dl, name); \
        if (!api->field) { \
            snprintf(err, err_len, "missing symbol %s", name); \
            return false; \
        } \
    } while (0)

static bool letsinfer_prefix_api_load(letsinfer_prefix_api *api, const char *lib_path,
                          char *err, size_t err_len) {
    api->dl = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!api->dl) {
        snprintf(err, err_len, "dlopen %.128s: %.100s", lib_path, dlerror());
        return false;
    }
    AC_SYM(abi_version, "letsinfer_prefix_abi_version");
    AC_SYM(store_open, "letsinfer_prefix_store_open");
    AC_SYM(store_close, "letsinfer_prefix_store_close");
    AC_SYM(begin_capture, "letsinfer_prefix_store_begin_capture");
    AC_SYM(writer_region, "letsinfer_prefix_writer_region");
    AC_SYM(writer_commit_async, "letsinfer_prefix_writer_commit_async");
    AC_SYM(writer_cancel, "letsinfer_prefix_writer_cancel");
    AC_SYM(longest_prefix, "letsinfer_prefix_store_longest_prefix");
    AC_SYM(reader_token_count, "letsinfer_prefix_reader_token_count");
    AC_SYM(reader_region_count, "letsinfer_prefix_reader_region_count");
    AC_SYM(reader_region_byte_count, "letsinfer_prefix_reader_region_byte_count");
    AC_SYM(reader_region_name, "letsinfer_prefix_reader_region_name");
    AC_SYM(reader_read_region, "letsinfer_prefix_reader_read_region");
    AC_SYM(reader_region_data, "letsinfer_prefix_reader_region_data");
    AC_SYM(reader_touch, "letsinfer_prefix_reader_touch");
    AC_SYM(reader_release_resident, "letsinfer_prefix_reader_release_resident");
    AC_SYM(reader_close, "letsinfer_prefix_reader_close");
    AC_SYM(reclaim_expired, "letsinfer_prefix_store_reclaim_expired");
    AC_SYM(stats_line, "letsinfer_prefix_store_stats_line");
    const uint32_t abi = api->abi_version();
    if (abi != LETSINFER_PREFIX_ABI_VERSION_EXPECTED) {
        snprintf(err, err_len, "bridge ABI %u, expected %u",
                 abi, LETSINFER_PREFIX_ABI_VERSION_EXPECTED);
        return false;
    }
    return true;
}

/* Default bridge path: libletsinfer_prefix_capi.so beside the running binary
 * (Linux); fall back to the dlopen default search order. */
static bool default_lib_path(char *out, size_t out_len) {
#ifdef __linux__
    char exe[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n > 0) {
        exe[n] = '\0';
        char *slash = strrchr(exe, '/');
        if (slash) {
            *slash = '\0';
            const int written = snprintf(out, out_len,
                                         "%s/libletsinfer_prefix_capi.so", exe);
            return written > 0 && (size_t)written < out_len;
        }
    }
#endif
    const int written = snprintf(out, out_len, "libletsinfer_prefix_capi.so");
    return written > 0 && (size_t)written < out_len;
}

ds4_letsinfer_cache *ds4_letsinfer_cache_open_from_env(ds4_engine *engine,
                                                    int serving_ctx,
                                                    ds4_letsinfer_log_fn log,
                                                    void *log_ud) {
    if (!env_flag("DS4_LETSINFER_CACHE", false)) return NULL;
    ds4_letsinfer_cache stack = {0};
    stack.log = log;
    stack.log_ud = log_ud;
    const ds4_letsinfer_cache *lg = &stack;   /* logging before allocation */

    if (!engine) {
        ac_logf(lg, 2, "letsinfer-cache: disabled (no engine)");
        return NULL;
    }
    const char *dir = getenv("DS4_LETSINFER_CACHE_DIR");
    if (!dir || !dir[0]) {
        ac_logf(lg, 2, "letsinfer-cache: DS4_LETSINFER_CACHE=1 but DS4_LETSINFER_CACHE_DIR "
                       "is unset; cache disabled");
        return NULL;
    }
    char lib_buf[PATH_MAX];
    const char *lib = getenv("DS4_LETSINFER_CACHE_LIB");
    if (!lib || !lib[0]) {
        if (!default_lib_path(lib_buf, sizeof(lib_buf))) {
            ac_logf(lg, 2, "letsinfer-cache: bridge path too long; cache disabled");
            return NULL;
        }
        lib = lib_buf;
    }
    char err[256] = {0};
    letsinfer_prefix_api api = {0};
    if (!letsinfer_prefix_api_load(&api, lib, err, sizeof(err))) {
        ac_logf(lg, 2, "letsinfer-cache: bridge unavailable (%s); cache disabled", err);
        if (api.dl) dlclose(api.dl);
        return NULL;
    }

    const uint64_t capacity = env_u64("DS4_LETSINFER_CACHE_MB", 65536) << 20;
    const uint64_t ttl = env_u64("DS4_LETSINFER_CACHE_TTL_S", 7ull * 24 * 3600);
    const uint64_t min_tokens = env_u64("DS4_LETSINFER_CACHE_MIN_TOKENS", 512);
    const uint64_t resident = env_u64("DS4_LETSINFER_CACHE_RESIDENT_MB", 0) << 20;
    const int direct = env_flag("DS4_LETSINFER_CACHE_DIRECT", true) ? 1 : 0;

    LetsInferPrefixStore *store = api.store_open(dir, capacity, ttl, min_tokens,
                                      resident, direct, err, sizeof(err));
    if (!store) {
        ac_logf(lg, 2, "letsinfer-cache: store open failed (%s); cache disabled", err);
        dlclose(api.dl);
        return NULL;
    }

    ds4_letsinfer_cache *ac = calloc(1, sizeof(*ac));
    if (!ac) {
        api.store_close(store);
        dlclose(api.dl);
        return NULL;
    }
    ac->api = api;
    ac->store = store;
    ac->log = log;
    ac->log_ud = log_ud;
    ac->min_tokens = (int)min_tokens;
    ac->capture = env_flag("DS4_LETSINFER_CACHE_CAPTURE", true);
    ac->prefix_lookup = env_flag("DS4_LETSINFER_CACHE_PREFIX", false);
    ac->last_reclaim_unix = (uint64_t)time(NULL);

    memcpy(ac->fingerprint, LETSINFER_FP_MAGIC, 8);
    le_put32(ac->fingerprint + 8, 1u);   /* adapter record ABI */
    le_put32(ac->fingerprint + 12, (uint32_t)ds4_engine_model_id(engine));
    le_put32(ac->fingerprint + 16, (uint32_t)ds4_engine_routed_quant_bits(engine));
    le_put32(ac->fingerprint + 20, (uint32_t)serving_ctx);
    le_put32(ac->fingerprint + 24, DS4_SESSION_PAYLOAD_VERSION);
    le_put32(ac->fingerprint + 28, (uint32_t)ds4_engine_layer_count(engine));

    char stats[512] = {0};
    if (ac->api.stats_line(ac->store, stats, sizeof(stats)) != 0) stats[0] = '\0';
    ac_logf(ac, 0, "letsinfer-cache: enabled dir=%s lib=%s capacity_mb=%llu ttl_s=%llu "
                   "min_tokens=%llu resident_mb=%llu direct=%d prefix_lookup=%d %s",
            dir, lib,
            (unsigned long long)(capacity >> 20), (unsigned long long)ttl,
            (unsigned long long)min_tokens, (unsigned long long)(resident >> 20),
            direct, ac->prefix_lookup ? 1 : 0, stats);
    return ac;
}

void ds4_letsinfer_cache_close(ds4_letsinfer_cache *ac) {
    if (!ac) return;
    char stats[512] = {0};
    if (ac->api.stats_line(ac->store, stats, sizeof(stats)) == 0)
        ac_logf(ac, 0, "letsinfer-cache: closing %s", stats);
    for (int b = 0; b < AC_MAX_BANKS; b++) free(ac->note_key[b]);
    ac->api.store_close(ac->store);
    dlclose(ac->api.dl);
    free(ac);
}

bool ds4_letsinfer_cache_enabled(const ds4_letsinfer_cache *ac) {
    return ac != NULL;
}

static uint32_t *tokens_to_u32(const int *tokens, int n) {
    if (!tokens || n <= 0) return NULL;
    uint32_t *out = malloc((size_t)n * sizeof(uint32_t));
    if (!out) return NULL;
    for (int i = 0; i < n; i++) {
        if (tokens[i] < 0) { free(out); return NULL; }
        out[i] = (uint32_t)tokens[i];
    }
    return out;
}

/* Opportunistic TTL reclamation, at most once per hour, from the retire
 * path (never the admission path). */
static void ac_maybe_reclaim(ds4_letsinfer_cache *ac) {
    const uint64_t now = (uint64_t)time(NULL);
    if (now < ac->last_reclaim_unix + 3600) return;
    ac->last_reclaim_unix = now;
    (void)ac->api.reclaim_expired(ac->store, now);
}

void ds4_letsinfer_cache_store_bank(ds4_letsinfer_cache *ac,
                                ds4_batch_ctx *batch_ctx,
                                int bank,
                                const int *prompt_tokens, int n_prompt,
                                const char *warm_text, size_t warm_text_len) {
    if (!ac || !ac->capture || !batch_ctx || bank < 0) return;
    if (!prompt_tokens || n_prompt <= 0 || !warm_text || warm_text_len == 0) return;
    if (n_prompt < ac->min_tokens) return;

    const int committed = ds4_batch_ctx_bank_committed(batch_ctx, bank, NULL);
    if (committed <= 0 || committed < n_prompt) return;

    const uint64_t payload_bytes = ds4_cont_bank_payload_bytes(batch_ctx,
                                                              (uint32_t)bank);
    if (payload_bytes == 0) return;

    uint32_t *key = tokens_to_u32(prompt_tokens, n_prompt);
    if (!key) return;

    /* Admission before any record-sized allocation or GPU readback. */
    const char *names[3] = { "ds4_meta", "warm_text", "bank_payload" };
    const uint64_t sizes[3] = { LETSINFER_META_BYTES, (uint64_t)warm_text_len,
                                payload_bytes };
    int rejection = 0;
    LetsInferPrefixWriter *writer = ac->api.begin_capture(ac->store, ac->fingerprint,
                                               key, (size_t)n_prompt,
                                               names, sizes, 3, &rejection);
    free(key);
    if (!writer) {
        /* 2=writer busy and 3=already stored are the normal paced/dedup
         * outcomes; log them at the kv-cache level, everything else as a
         * warning. */
        ac_logf(ac, rejection == 2 || rejection == 3 ? 1 : 2,
                "letsinfer-cache: capture skipped bank=%d prompt_tokens=%d "
                "payload_bytes=%llu rejection=%d",
                bank, n_prompt, (unsigned long long)payload_bytes, rejection);
        return;
    }

    char err[256] = {0};
    uint64_t got = 0;
    uint8_t *meta = ac->api.writer_region(writer, 0, &got);
    uint8_t *text = ac->api.writer_region(writer, 1, NULL);
    uint8_t *payload = ac->api.writer_region(writer, 2, NULL);
    if (!meta || got != LETSINFER_META_BYTES || !text || !payload) {
        ac->api.writer_cancel(writer);
        ac_logf(ac, 2, "letsinfer-cache: capture aborted bank=%d (region mapping)",
                bank);
        return;
    }

    memset(meta, 0, LETSINFER_META_BYTES);
    memcpy(meta, LETSINFER_META_MAGIC, 8);
    le_put32(meta + 8, LETSINFER_META_VERSION);
    le_put32(meta + 12, (uint32_t)committed);
    le_put64(meta + 16, (uint64_t)warm_text_len);
    le_put64(meta + 24, payload_bytes);
    le_put32(meta + 32, (uint32_t)n_prompt);
    memcpy(text, warm_text, warm_text_len);

    /* Serialize the bank state straight into the record buffer (the same
     * wire bytes the native disk tier writes; no temp file). */
    FILE *fp = fmemopen(payload, (size_t)payload_bytes, "wb");
    if (!fp) {
        ac->api.writer_cancel(writer);
        ac_logf(ac, 2, "letsinfer-cache: capture aborted bank=%d (fmemopen: %s)",
                bank, strerror(errno));
        return;
    }
    const double t0 = ac_now_ms();
    const int rc = ds4_cont_bank_save_payload(batch_ctx, (uint32_t)bank, fp,
                                              err, sizeof(err));
    const int close_rc = fclose(fp);
    if (rc != 0 || close_rc != 0) {
        ac->api.writer_cancel(writer);
        ac_logf(ac, 2, "letsinfer-cache: capture aborted bank=%d (save: %s)",
                bank, err[0] ? err : strerror(errno));
        return;
    }
    if (ac->api.writer_commit_async(writer, 0, err, sizeof(err)) != 0) {
        ac_logf(ac, 2, "letsinfer-cache: commit rejected bank=%d (%s)", bank, err);
        return;
    }
    ac_logf(ac, 1, "letsinfer-cache: capture queued bank=%d prompt_tokens=%d "
                   "committed=%d payload_bytes=%llu stage=%.1f ms",
            bank, n_prompt, committed, (unsigned long long)payload_bytes,
            ac_now_ms() - t0);
    ac_maybe_reclaim(ac);
}

void ds4_letsinfer_cache_note_bank(ds4_letsinfer_cache *ac, int bank,
                               const int *prompt_tokens, int n_prompt) {
    if (!ac || bank < 0 || bank >= AC_MAX_BANKS) return;
    free(ac->note_key[bank]);
    ac->note_key[bank] = NULL;
    ac->note_len[bank] = 0;
    if (!prompt_tokens || n_prompt <= 0 || n_prompt < ac->min_tokens) return;
    int *cp = malloc((size_t)n_prompt * sizeof(int));
    if (!cp) return;
    memcpy(cp, prompt_tokens, (size_t)n_prompt * sizeof(int));
    ac->note_key[bank] = cp;
    ac->note_len[bank] = n_prompt;
}

void ds4_letsinfer_cache_clear_note(ds4_letsinfer_cache *ac, int bank) {
    if (!ac || bank < 0 || bank >= AC_MAX_BANKS) return;
    free(ac->note_key[bank]);
    ac->note_key[bank] = NULL;
    ac->note_len[bank] = 0;
}

void ds4_letsinfer_cache_store_noted(ds4_letsinfer_cache *ac,
                                 ds4_batch_ctx *batch_ctx, int bank,
                                 const char *warm_text, size_t warm_text_len) {
    if (!ac || bank < 0 || bank >= AC_MAX_BANKS) return;
    if (!ac->note_key[bank] || ac->note_len[bank] <= 0) return;
    ds4_letsinfer_cache_store_bank(ac, batch_ctx, bank,
                               ac->note_key[bank], ac->note_len[bank],
                               warm_text, warm_text_len);
    /* Keep the note: repeat captures dedup inside the store. */
}

int ds4_letsinfer_cache_lookup(ds4_letsinfer_cache *ac,
                           const int *prompt_tokens, int n_prompt,
                           void **handle_out) {
    if (handle_out) *handle_out = NULL;
    if (!ac || !handle_out || !prompt_tokens || n_prompt <= 0) return 0;
    if (n_prompt < ac->min_tokens) return 0;
    uint32_t *key = tokens_to_u32(prompt_tokens, n_prompt);
    if (!key) return 0;
    /* MVP: exact full-prompt match (minimum = the full prompt).  The
     * experimental prefix mode admits shorter stored prompts; the server's
     * matcher + engine frontier validation stay authoritative either way. */
    const size_t minimum = ac->prefix_lookup ? (size_t)ac->min_tokens
                                             : (size_t)n_prompt;
    LetsInferPrefixReader *reader = ac->api.longest_prefix(ac->store, ac->fingerprint,
                                                key, (size_t)n_prompt, minimum);
    free(key);
    if (!reader) return 0;
    const size_t count = ac->api.reader_token_count(reader);
    if (count == 0 || count > (size_t)INT_MAX) {
        ac->api.reader_close(reader);
        return 0;
    }
    *handle_out = reader;
    return (int)count;
}

void ds4_letsinfer_cache_release(ds4_letsinfer_cache *ac, void *handle) {
    if (!ac || !handle) return;
    ac->api.reader_close((LetsInferPrefixReader *)handle);
}

int ds4_letsinfer_cache_restore(ds4_letsinfer_cache *ac,
                            void *handle,
                            ds4_batch_ctx *batch_ctx,
                            uint32_t bank,
                            char **record_text_out,
                            size_t *record_text_len_out) {
    if (record_text_out) *record_text_out = NULL;
    if (record_text_len_out) *record_text_len_out = 0;
    if (!ac || !handle) return 0;
    LetsInferPrefixReader *reader = (LetsInferPrefixReader *)handle;
    char err[256] = {0};
    char *text = NULL;
    const uint8_t *payload = NULL;
    FILE *fp = NULL;
    int loaded = 0;
    const double t0 = ac_now_ms();

    if (!batch_ctx || !record_text_out) goto out;
    if (ac->api.reader_region_count(reader) != 3) {
        ac_logf(ac, 2, "letsinfer-cache: restore skipped (region count)");
        goto out;
    }
    char name[16] = {0};
    if (ac->api.reader_region_name(reader, 0, name, sizeof(name)) != 0 ||
        strcmp(name, "ds4_meta") != 0 ||
        ac->api.reader_region_byte_count(reader, 0) != LETSINFER_META_BYTES) {
        ac_logf(ac, 2, "letsinfer-cache: restore skipped (meta region)");
        goto out;
    }
    uint8_t meta[LETSINFER_META_BYTES];
    if (ac->api.reader_read_region(reader, 0, meta, LETSINFER_META_BYTES,
                                   err, sizeof(err)) != 0) {
        ac_logf(ac, 2, "letsinfer-cache: restore failed (meta: %s)", err);
        goto out;
    }
    if (memcmp(meta, LETSINFER_META_MAGIC, 8) != 0 ||
        le_get32(meta + 8) != LETSINFER_META_VERSION) {
        ac_logf(ac, 2, "letsinfer-cache: restore skipped (meta magic/version)");
        goto out;
    }
    const uint32_t committed_expected = le_get32(meta + 12);
    const uint64_t text_bytes = le_get64(meta + 16);
    const uint64_t payload_bytes = le_get64(meta + 24);
    if (committed_expected == 0 ||
        text_bytes == 0 || text_bytes > (uint64_t)SIZE_MAX / 2 ||
        payload_bytes == 0 || payload_bytes > (uint64_t)SIZE_MAX / 2 ||
        ac->api.reader_region_byte_count(reader, 1) != text_bytes ||
        ac->api.reader_region_byte_count(reader, 2) != payload_bytes) {
        ac_logf(ac, 2, "letsinfer-cache: restore skipped (meta geometry)");
        goto out;
    }

    text = malloc((size_t)text_bytes + 1);
    if (!text) {
        ac_logf(ac, 2, "letsinfer-cache: restore failed (allocation %llu bytes)",
                (unsigned long long)text_bytes);
        goto out;
    }
    if (ac->api.reader_read_region(reader, 1, (uint8_t *)text, text_bytes,
                                   err, sizeof(err)) != 0) {
        ac_logf(ac, 2, "letsinfer-cache: restore failed (text: %s)", err);
        goto out;
    }
    text[text_bytes] = '\0';
    uint64_t payload_view_bytes = 0;
    payload = ac->api.reader_region_data(reader, 2, &payload_view_bytes,
                                         err, sizeof(err));
    if (!payload || payload_view_bytes != payload_bytes) {
        ac_logf(ac, 2, "letsinfer-cache: restore failed (payload view: %s)",
                err[0] ? err : "geometry mismatch");
        goto out;
    }

    fp = fmemopen((void *)payload, (size_t)payload_bytes, "rb");
    if (!fp) {
        ac_logf(ac, 2, "letsinfer-cache: restore failed (fmemopen: %s)",
                strerror(errno));
        goto out;
    }
    if (ds4_cont_bank_restore_payload(batch_ctx, bank, fp, payload_bytes,
                                      err, sizeof(err)) != 0) {
        ac_logf(ac, 2, "letsinfer-cache: restore failed bank=%u (%s)", bank, err);
        goto out;
    }
    const int *htok = NULL;
    const int hl = ds4_batch_ctx_bank_committed(batch_ctx, (int)bank, &htok);
    if (!htok || hl <= 0 || (uint32_t)hl != committed_expected) {
        /* Restore claimed success but the bank disagrees: structural
         * breakage.  Leave the bank to the engine's reset-invalid handling
         * and treat the record as a miss (its CRCs passed, so keep it for
         * a matching build rather than evicting). */
        ac_logf(ac, 2, "letsinfer-cache: restore inconsistent bank=%u committed=%d "
                       "expected=%u", bank, hl, committed_expected);
        goto out;
    }

    ac->api.reader_touch(reader);
    ac->api.reader_release_resident(reader);
    *record_text_out = text;
    if (record_text_len_out) *record_text_len_out = (size_t)text_bytes;
    text = NULL;
    loaded = hl;
    ac_logf(ac, 1, "letsinfer-cache: restore hit bank=%u committed=%d "
                   "payload_bytes=%llu load=%.1f ms",
            bank, hl, (unsigned long long)payload_bytes,
            ac_now_ms() - t0);

out:
    if (fp) fclose(fp);
    free(text);
    ac->api.reader_close(reader);
    return loaded;
}
