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
    std::atomic<uint64_t> nvme_us{0};
    std::atomic<uint64_t> kernel_us{0};
    std::atomic<uint64_t> ple_us{0};
    std::atomic<uint64_t> prekick_us{0};
    std::atomic<uint64_t> bytes_read{0};
    std::atomic<uint64_t> served{0};
    std::atomic<uint64_t> hits[48] = {};
    std::atomic<uint64_t> misses[48] = {};
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
