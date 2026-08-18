// SPDX-License-Identifier: MIT

#include "ds4_batch_policy.h"

#include <assert.h>

static void expect_policy(uint32_t n_live, uint32_t id, uint32_t fit,
                          uint32_t capture, uint32_t d2r) {
    const ds4_dspark_batch_policy p =
        ds4_dspark_batch_policy_select(true, n_live, 224u);
    assert(p.active);
    assert(p.id == id);
    assert(p.fit_rows == fit);
    assert(p.capture_rows == capture);
    assert(p.d2r_min_cols == d2r);
    assert(p.sorted_wide);
}

int main(void) {
    assert(!ds4_dspark_batch_policy_select(false, 8u, 224u).active);
    assert(!ds4_dspark_batch_policy_select(true, 1u, 224u).active);
    assert(!ds4_dspark_batch_policy_select(true, 8u, 0u).active);
    assert(ds4_dspark_batch_policy_select(
               true, 8u, DS4_DSPARK_AUTO_BATCH_MAX_EXTENT).active);
    assert(!ds4_dspark_batch_policy_select(
                true, 8u, DS4_DSPARK_AUTO_BATCH_MAX_EXTENT + 1u).active);

    expect_policy(2u, 1u, 8u, 8u, 32u);
    expect_policy(4u, 1u, 8u, 8u, 32u);
    expect_policy(5u, 2u, 16u, 16u, 64u);
    expect_policy(8u, 2u, 16u, 16u, 64u);
    expect_policy(9u, 3u, 24u, 16u, 96u);
    expect_policy(12u, 3u, 24u, 16u, 96u);
    expect_policy(13u, 4u, 32u, 16u, 128u);
    expect_policy(128u, 4u, 32u, 16u, 128u);

    ds4_dspark_batch_policy cohort = {0};
    cohort = ds4_dspark_batch_policy_promote(
        cohort, ds4_dspark_batch_policy_select(true, 12u, 224u));
    cohort = ds4_dspark_batch_policy_promote(
        cohort, ds4_dspark_batch_policy_select(true, 4u, 224u));
    assert(cohort.id == 3u);
    assert(cohort.fit_rows == 24u);

    assert(ds4_dspark_batch_policy_prefill_live(
               true, 4u, true, 224u, 0u) == 512u);
    assert(ds4_dspark_batch_policy_prefill_live(
               true, 2u, true, DS4_DSPARK_AUTO_BATCH_MAX_EXTENT, 0u) == 512u);
    assert(ds4_dspark_batch_policy_prefill_live(
               false, 4u, true, 224u, 0u) == 0u);
    assert(ds4_dspark_batch_policy_prefill_live(
               true, 1u, true, 224u, 0u) == 0u);
    assert(ds4_dspark_batch_policy_prefill_live(
               true, 4u, false, 224u, 0u) == 0u);
    assert(ds4_dspark_batch_policy_prefill_live(
               true, 4u, true,
               DS4_DSPARK_AUTO_BATCH_MAX_EXTENT + 1u, 0u) == 0u);
    assert(ds4_dspark_batch_policy_prefill_live(
               true, 4u, true, 224u, 256u) == 512u);
    return 0;
}
