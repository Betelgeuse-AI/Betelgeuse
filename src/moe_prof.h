// Lightweight wall-clock profiling for the MoE demand path. Enabled via MOE_PROF=1.
// Inline variables share one instance across all translation units (C++17).
#pragma once
#include <atomic>
#include <cstdint>
#include <chrono>
#include <cstdio>

struct moe_prof_state {
    std::atomic<uint64_t> plan_us{0};
    std::atomic<uint64_t> plan_sel_us{0};
    std::atomic<uint64_t> plan_victim_us{0};
    std::atomic<uint64_t> plan_assign_us{0};
    std::atomic<uint64_t> victim_us{0};
    std::atomic<uint64_t> victim_fill_us{0};
    std::atomic<uint64_t> victim_fill_reserve_us{0};
    std::atomic<uint64_t> victim_fill_memcpy_us{0};
    std::atomic<uint64_t> nvme_us{0};
    std::atomic<uint64_t> kernel_us{0};
    std::atomic<uint64_t> ple_us{0};
    std::atomic<uint64_t> prekick_us{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> served{0};
    std::atomic<uint64_t> cqe_n[5] = {};
    std::atomic<uint64_t> cqe_us[5] = {};
    std::atomic<uint64_t> spec_fallback{0};
    std::atomic<uint64_t> nvme_prep_submit_us{0};
    std::atomic<uint64_t> nvme_prep_cpu_us{0};
    std::atomic<uint64_t> nvme_submit_syscall_us{0};
    std::atomic<uint64_t> rfd_direct_count{0};
    std::atomic<uint64_t> rfd_buf_count{0};
    std::atomic<uint64_t> pk_skip_pending{0};
    std::atomic<uint64_t> pk_skip_demand{0};
    std::atomic<uint64_t> pk_skip_vram{0};
    std::atomic<uint64_t> pk_skip_cache{0};
    std::atomic<uint64_t> pk_submitted{0};
    std::atomic<uint64_t> pk_skip_pred_unset{0};
    std::atomic<uint64_t> pk_state1_seen{0};
    std::atomic<uint64_t> nvme_drain_spec_us{0};
    std::atomic<uint64_t> nvme_publish_h2d_us{0};
    std::atomic<uint64_t> conveyor_eval_us{0};
    std::atomic<uint64_t> nvme_cqe_wait_us{0};
    std::atomic<uint64_t> hits[48] = {};
    std::atomic<uint64_t> misses[48] = {};
    // Mission 62: TG (n_tokens == 1) vs PP (n_tokens > 1) splits.
    std::atomic<uint64_t> tg_plan_us{0};
    std::atomic<uint64_t> tg_plan_victim_us{0};
    std::atomic<uint64_t> tg_victim_us{0};
    std::atomic<uint64_t> tg_nvme_us{0};
    std::atomic<uint64_t> tg_nvme_publish_h2d_us{0};
    std::atomic<uint64_t> tg_nvme_cqe_wait_us{0};
    std::atomic<uint64_t> tg_kernel_us{0};
    std::atomic<uint64_t> tg_served{0};
    std::atomic<uint64_t> tg_rfd_direct_count{0};
    std::atomic<uint64_t> tg_bytes_read{0};
    std::atomic<uint64_t> pp_plan_us{0};
    std::atomic<uint64_t> pp_plan_victim_us{0};
    std::atomic<uint64_t> pp_victim_us{0};
    std::atomic<uint64_t> pp_nvme_us{0};
    std::atomic<uint64_t> pp_nvme_publish_h2d_us{0};
    std::atomic<uint64_t> pp_nvme_cqe_wait_us{0};
    std::atomic<uint64_t> pp_kernel_us{0};
    std::atomic<uint64_t> pp_served{0};
    std::atomic<uint64_t> pp_rfd_direct_count{0};
    std::atomic<uint64_t> pp_bytes_read{0};
    std::atomic<uint64_t> tg_hits[48] = {};
    std::atomic<uint64_t> tg_misses[48] = {};
    std::atomic<uint64_t> pp_hits[48] = {};
    std::atomic<uint64_t> pp_misses[48] = {};
    std::atomic<uint64_t> doorbell_t0_us{0};
};

inline moe_prof_state g_moe_prof;

inline bool moe_prof_enabled() {
    static int v = -1;
    if (v < 0) { const char * e = getenv("MOE_PROF"); v = e && atoi(e) != 0 ? 1 : 0; }
    return v == 1;
}

inline uint64_t moe_prof_now() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
