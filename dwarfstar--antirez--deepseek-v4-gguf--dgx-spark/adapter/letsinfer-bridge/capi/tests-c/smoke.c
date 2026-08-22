/* SPDX-License-Identifier: AGPL-3.0-only
 * smoke.c - end-to-end C exercise of the letsinfer_prefix_* ABI through dlopen, the
 * same way ds4-server consumes it: open store, capture a 3-region record,
 * async commit, exact lookup, region reads with byte verification, miss on
 * fingerprint/token mismatch, reopen persistence.  Exits 0 on success.
 *
 * Build+run (arm64 Linux container, from letsinfer-bridge/):
 *   cargo build --release
 *   gcc -O2 -Wall -Wextra -o /tmp/letsinfer_prefix-smoke capi/tests-c/smoke.c -ldl
 *   /tmp/letsinfer_prefix-smoke target/release/libletsinfer_prefix_capi.so /tmp/letsinfer_prefix-store
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); return 1; } } while (0)

typedef struct LetsInferPrefixStore LetsInferPrefixStore;
typedef struct LetsInferPrefixWriter LetsInferPrefixWriter;
typedef struct LetsInferPrefixReader LetsInferPrefixReader;

static void *must_sym(void *dl, const char *name) {
    void *sym = dlsym(dl, name);
    if (!sym) { fprintf(stderr, "missing symbol %s\n", name); exit(1); }
    return sym;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <libletsinfer_prefix_capi.so> <store-dir>\n", argv[0]);
        return 2;
    }
    void *dl = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!dl) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    uint32_t (*abi_version)(void) = must_sym(dl, "letsinfer_prefix_abi_version");
    LetsInferPrefixStore *(*store_open)(const char *, uint64_t, uint64_t, uint64_t, uint64_t, int, char *, size_t) = must_sym(dl, "letsinfer_prefix_store_open");
    void (*store_close)(LetsInferPrefixStore *) = must_sym(dl, "letsinfer_prefix_store_close");
    LetsInferPrefixWriter *(*begin_capture)(LetsInferPrefixStore *, const uint8_t *, const uint32_t *, size_t, const char *const *, const uint64_t *, size_t, int *) = must_sym(dl, "letsinfer_prefix_store_begin_capture");
    uint8_t *(*writer_region)(LetsInferPrefixWriter *, size_t, uint64_t *) = must_sym(dl, "letsinfer_prefix_writer_region");
    int (*writer_commit_async)(LetsInferPrefixWriter *, int, char *, size_t) = must_sym(dl, "letsinfer_prefix_writer_commit_async");
    LetsInferPrefixReader *(*longest_prefix)(LetsInferPrefixStore *, const uint8_t *, const uint32_t *, size_t, size_t) = must_sym(dl, "letsinfer_prefix_store_longest_prefix");
    size_t (*reader_token_count)(const LetsInferPrefixReader *) = must_sym(dl, "letsinfer_prefix_reader_token_count");
    size_t (*reader_region_count)(const LetsInferPrefixReader *) = must_sym(dl, "letsinfer_prefix_reader_region_count");
    uint64_t (*reader_region_byte_count)(const LetsInferPrefixReader *, size_t) = must_sym(dl, "letsinfer_prefix_reader_region_byte_count");
    int (*reader_region_name)(const LetsInferPrefixReader *, size_t, char *, size_t) = must_sym(dl, "letsinfer_prefix_reader_region_name");
    int (*reader_read_region)(LetsInferPrefixReader *, size_t, uint8_t *, uint64_t, char *, size_t) = must_sym(dl, "letsinfer_prefix_reader_read_region");
    const uint8_t *(*reader_region_data)(LetsInferPrefixReader *, size_t, uint64_t *, char *, size_t) = must_sym(dl, "letsinfer_prefix_reader_region_data");
    void (*reader_touch)(const LetsInferPrefixReader *) = must_sym(dl, "letsinfer_prefix_reader_touch");
    void (*reader_close)(LetsInferPrefixReader *) = must_sym(dl, "letsinfer_prefix_reader_close");
    int (*stats_line)(const LetsInferPrefixStore *, char *, size_t) = must_sym(dl, "letsinfer_prefix_store_stats_line");

    CHECK(abi_version() == 2);

    char err[256] = {0};
    LetsInferPrefixStore *store = store_open(argv[2], 64ull << 20, 3600, 4, 0, 0, err, sizeof(err));
    if (!store) { fprintf(stderr, "open: %s\n", err); return 1; }

    uint8_t fp[32];
    for (int i = 0; i < 32; i++) fp[i] = (uint8_t)(i * 7 + 1);
    uint32_t tokens[64];
    for (int i = 0; i < 64; i++) tokens[i] = (uint32_t)(1000 + i);

    const char *names[3] = { "ds4_meta", "warm_text", "bank_payload" };
    const uint64_t sizes[3] = { 64, 1234, 1u << 20 };
    int rejection = 0;
    LetsInferPrefixWriter *writer = begin_capture(store, fp, tokens, 64, names, sizes, 3, &rejection);
    CHECK(writer != NULL);
    for (size_t r = 0; r < 3; r++) {
        uint64_t got = 0;
        uint8_t *region = writer_region(writer, r, &got);
        CHECK(region && got == sizes[r]);
        for (uint64_t i = 0; i < got; i++) region[i] = (uint8_t)(i * (r + 3));
    }
    CHECK(writer_commit_async(writer, 1, err, sizeof(err)) == 0);

    /* The background writer commits asynchronously; poll the lookup. */
    LetsInferPrefixReader *reader = NULL;
    for (int spin = 0; spin < 200 && !reader; spin++) {
        reader = longest_prefix(store, fp, tokens, 64, 64);
        if (!reader) usleep(20000);
    }
    CHECK(reader != NULL);
    CHECK(reader_token_count(reader) == 64);
    CHECK(reader_region_count(reader) == 3);
    for (size_t r = 0; r < 3; r++) {
        char name[16] = {0};
        CHECK(reader_region_name(reader, r, name, sizeof(name)) == 0);
        CHECK(strcmp(name, names[r]) == 0);
        const uint64_t bytes = reader_region_byte_count(reader, r);
        CHECK(bytes == sizes[r]);
        uint8_t *buf = malloc(bytes);
        CHECK(buf && reader_read_region(reader, r, buf, bytes, err, sizeof(err)) == 0);
        for (uint64_t i = 0; i < bytes; i++)
            if (buf[i] != (uint8_t)(i * (r + 3))) { fprintf(stderr, "region %zu byte %llu mismatch\n", r, (unsigned long long)i); return 1; }
        free(buf);
    }
    uint64_t view_bytes = 0;
    const uint8_t *view = reader_region_data(reader, 2, &view_bytes, err, sizeof(err));
    CHECK(view && view_bytes == sizes[2]);
    for (uint64_t i = 0; i < view_bytes; i++)
        if (view[i] != (uint8_t)(i * 5)) { fprintf(stderr, "region view byte %llu mismatch\n", (unsigned long long)i); return 1; }
    reader_touch(reader);
    reader_close(reader);

    /* Exact-match rule: a shorter stored prompt must miss when the minimum
     * equals the query length; wrong fingerprint must miss. */
    CHECK(longest_prefix(store, fp, tokens, 63, 63) == NULL);
    uint8_t fp2[32];
    memcpy(fp2, fp, 32);
    fp2[0] ^= 0xff;
    CHECK(longest_prefix(store, fp2, tokens, 64, 64) == NULL);
    /* Prefix mode (minimum < stored length): the record serves as a prefix
     * of a longer prompt. */
    uint32_t longer[80];
    memcpy(longer, tokens, sizeof(tokens));
    for (int i = 64; i < 80; i++) longer[i] = 7u;
    LetsInferPrefixReader *pref = longest_prefix(store, fp, longer, 80, 4);
    CHECK(pref != NULL && reader_token_count(pref) == 64);
    reader_close(pref);
    /* Duplicate capture dedups. */
    rejection = 0;
    CHECK(begin_capture(store, fp, tokens, 64, names, sizes, 3, &rejection) == NULL);
    CHECK(rejection == 3 /* already stored */);

    char line[512] = {0};
    CHECK(stats_line(store, line, sizeof(line)) == 0);
    printf("stats: %s\n", line);
    store_close(store);

    /* Restart persistence: reopen and hit again. */
    store = store_open(argv[2], 64ull << 20, 3600, 4, 0, 0, err, sizeof(err));
    CHECK(store != NULL);
    reader = longest_prefix(store, fp, tokens, 64, 64);
    CHECK(reader != NULL && reader_token_count(reader) == 64);
    uint8_t meta[64];
    CHECK(reader_read_region(reader, 0, meta, 64, err, sizeof(err)) == 0);
    CHECK(meta[1] == (uint8_t)3);
    reader_close(reader);
    store_close(store);
    dlclose(dl);
    printf("letsinfer_prefix smoke OK\n");
    return 0;
}
