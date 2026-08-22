/* SPDX-License-Identifier: AGPL-3.0-only
 * ds4_letsinfer_cache.h - DwarfStar adapter for the Let's Infer persistent
 * prefix cache (letsinfer_prefix_store via the libletsinfer_prefix_capi.so C
 * ABI, loaded with dlopen).
 *
 * Opt-in and inert by default: every entry point is a no-op unless
 * DS4_LETSINFER_CACHE=1 was set at ds4_letsinfer_cache_open_from_env() time and the
 * bridge library + store opened successfully.  All failures are fail-open:
 * one log line, NULL/0 return, normal prefill continues.  The cache must
 * never crash the server or serve wrong state; the server-side warm-record
 * matcher and the engine's frontier validation stay authoritative after a
 * restore, exactly as for the native disk KV tier.
 *
 * Environment:
 *   DS4_LETSINFER_CACHE=1            enable (anything else: fully disabled)
 *   DS4_LETSINFER_CACHE_DIR=PATH     store root on NVMe (required when enabled)
 *   DS4_LETSINFER_CACHE_LIB=PATH     bridge .so (default: libletsinfer_prefix_capi.so
 *                                next to the ds4-server binary, then dlopen
 *                                default search order)
 *   DS4_LETSINFER_CACHE_MB=N         durable byte budget, MiB (default 65536)
 *   DS4_LETSINFER_CACHE_TTL_S=N      sliding expiry seconds (default 604800 = 7d)
 *   DS4_LETSINFER_CACHE_MIN_TOKENS=N minimum prompt tokens to capture (default 512)
 *   DS4_LETSINFER_CACHE_RESIDENT_MB=N host-RAM resident tier, MiB (default 0 = off)
 *   DS4_LETSINFER_CACHE_DIRECT=0     disable O_DIRECT bulk reads (default on)
 *   DS4_LETSINFER_CACHE_CAPTURE=0    restore-only mode: never store new records
 *   DS4_LETSINFER_CACHE_PREFIX=1     EXPERIMENTAL: admit stored prompts that are
 *                                a strict token prefix of the request
 *                                (default: exact full-prompt key match only)
 */
#ifndef DS4_LETSINFER_CACHE_H
#define DS4_LETSINFER_CACHE_H

#include "ds4.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct ds4_letsinfer_cache ds4_letsinfer_cache;

/* Log levels match ds4_server's server_log() types; the callback runs on
 * whichever thread called the adapter. */
typedef void (*ds4_letsinfer_log_fn)(void *ud, int level, const char *msg);

/* Open from environment.  Returns NULL when disabled or on ANY failure
 * (logged).  engine identity (model id, routed quant bits) plus serving_ctx
 * form the record compatibility fingerprint: records written by a different
 * model, quant, serving ctx, or payload ABI are invisible, mirroring
 * --kv-cache-reject-different-quant semantics. */
ds4_letsinfer_cache *ds4_letsinfer_cache_open_from_env(ds4_engine *engine,
                                                    int serving_ctx,
                                                    ds4_letsinfer_log_fn log,
                                                    void *log_ud);
void ds4_letsinfer_cache_close(ds4_letsinfer_cache *ac);

/* True when the adapter is live (open succeeded). */
bool ds4_letsinfer_cache_enabled(const ds4_letsinfer_cache *ac);

/* Capture one idle cont bank keyed by the admitting request's canonical
 * prompt token IDs.  Runs at retire on the worker thread under gen_mu (the
 * bank is idle by construction).  The GPU->host payload readback is
 * synchronous (same cost class as the native DS4_SERVER_BANK_CHECKPOINT
 * store); CRC + NVMe write happen on the store's bounded background writer.
 * Admission (dedup, byte budget, writer busy, min tokens) is checked before
 * any buffer allocation or readback.  Never blocks on a busy writer. */
void ds4_letsinfer_cache_store_bank(ds4_letsinfer_cache *ac,
                                  ds4_batch_ctx *batch_ctx,
                                  int bank,
                                  const int *prompt_tokens, int n_prompt,
                                  const char *warm_text, size_t warm_text_len);

/* Deferred capture: the prompt key is only known at retire, but retire-time
 * bank state is NOT settled (tail tokens commit after on_done; a retire-time
 * snapshot reproducibly misread its prompt after restore, 2026-08-05).
 * note_bank stashes the key at retire; store_noted captures at the settled
 * sites (bank-evict / checkpoint / shutdown) with the bank's current warm
 * text; clear_note drops a stale key when the bank's warm record is
 * invalidated. */
void ds4_letsinfer_cache_note_bank(ds4_letsinfer_cache *ac, int bank,
                                 const int *prompt_tokens, int n_prompt);
void ds4_letsinfer_cache_clear_note(ds4_letsinfer_cache *ac, int bank);
void ds4_letsinfer_cache_store_noted(ds4_letsinfer_cache *ac,
                                   ds4_batch_ctx *batch_ctx, int bank,
                                   const char *warm_text, size_t warm_text_len);

/* Lookup handle for a restore decision (victim-bank pick happens between
 * lookup and restore).  Returns the record's key token count (> 0) on a
 * hit and sets *handle_out, or 0 as a safe miss. */
int ds4_letsinfer_cache_lookup(ds4_letsinfer_cache *ac,
                             const int *prompt_tokens, int n_prompt,
                             void **handle_out);

/* Restore the looked-up record INTO cont bank `bank` (tensors + counters +
 * committed history via ds4_cont_bank_restore_payload).  On success returns
 * the restored committed token count and hands back a heap copy of the
 * record's warm text (caller owns; install as the bank's warm record so the
 * normal matcher/partial-cut/frontier validation take over).  On ANY
 * failure returns 0 with the bank left to the engine's reset-invalid
 * handling, exactly like a corrupt native disk entry.  Always consumes and
 * releases `handle`. */
int ds4_letsinfer_cache_restore(ds4_letsinfer_cache *ac,
                              void *handle,
                              ds4_batch_ctx *batch_ctx,
                              uint32_t bank,
                              char **record_text_out,
                              size_t *record_text_len_out);

/* Release a lookup handle without restoring (e.g. no victim bank). */
void ds4_letsinfer_cache_release(ds4_letsinfer_cache *ac, void *handle);

#endif
