// SPDX-License-Identifier: MIT
// Runtime-local DSpark policy for short concurrent request cohorts.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#define DS4_DSPARK_AUTO_BATCH_MAX_EXTENT 24576u
#define DS4_DSPARK_AUTO_BATCH_PREFILL_LIVE 512u

typedef struct {
    bool     active;
    uint32_t id;
    uint32_t fit_rows;
    uint32_t capture_rows;
    uint32_t d2r_min_cols;
    bool     sorted_wide;
} ds4_dspark_batch_policy;

/* Reproduce the measured Entrpi concurrency recipes from one serving process.
 * `max_busy_extent` is prompt + generation budget for the largest live or
 * pending request. A zero/unknown extent and anything above 24K fail closed to
 * the manifest's long-context settings. This covers Entrpi's deepest 16K
 * prefix case without touching the 32K performance-matrix boundary. C1
 * deliberately stays on the long-context path. */
static inline ds4_dspark_batch_policy ds4_dspark_batch_policy_select(
        bool enabled, uint32_t n_live, uint32_t max_busy_extent) {
    ds4_dspark_batch_policy p = {0};
    if (!enabled || n_live < 2u || max_busy_extent == 0u ||
        max_busy_extent > DS4_DSPARK_AUTO_BATCH_MAX_EXTENT) {
        return p;
    }

    p.active = true;
    p.capture_rows = 16u;
    p.sorted_wide = true;
    if (n_live <= 4u) {
        p.id = 1u;
        p.fit_rows = 8u;
        p.capture_rows = 8u;
        p.d2r_min_cols = 32u;
    } else if (n_live <= 8u) {
        p.id = 2u;
        p.fit_rows = 16u;
        p.d2r_min_cols = 64u;
    } else if (n_live <= 12u) {
        p.id = 3u;
        p.fit_rows = 24u;
        p.d2r_min_cols = 96u;
    } else {
        p.id = 4u;
        p.fit_rows = 32u;
        p.d2r_min_cols = 128u;
    }
    return p;
}

static inline ds4_dspark_batch_policy ds4_dspark_batch_policy_promote(
        ds4_dspark_batch_policy cohort, ds4_dspark_batch_policy observed) {
    if (observed.active && (!cohort.active || observed.id > cohort.id))
        return observed;
    return cohort;
}

/* The retained short-concurrency processes used DwarfStar's 512-token live
 * admission interleave. Let's Infer's long-context recipe deliberately sets the
 * configured width to zero, so select the old value only while two or more
 * known short requests overlap. A solo request does not need interleaving;
 * an unknown or long request keeps the configured long-context behavior. */
static inline uint32_t ds4_dspark_batch_policy_prefill_live(
        bool enabled, uint32_t n_busy, bool extents_known,
        uint32_t max_busy_extent, uint32_t configured) {
    if (enabled && n_busy >= 2u && extents_known && max_busy_extent != 0u &&
        max_busy_extent <= DS4_DSPARK_AUTO_BATCH_MAX_EXTENT) {
        return DS4_DSPARK_AUTO_BATCH_PREFILL_LIVE;
    }
    return configured;
}
