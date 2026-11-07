// test_moe_pager.cpp -- standalone verification for the O_DIRECT expert pager.
//
// Proves the thing that matters before it is wired into the graph: the bytes
// the pager DMAs for a given (layer, expert) are byte-identical to the bytes
// sitting at that expert's slice in the GGUF.
//
// It does this by reading the same slice twice through two independent paths
// (a plain buffered pread, and the pager's O_DIRECT+io_uring path) and comparing.
#include "moe_pager.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cerrno>

static const char * MODEL =
    "/mnt/windows/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS/IQ2_XS/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf";

static bool read_reference(int fd, const moe_expert_slice & slice, void * destination, size_t bytes) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    const int buffered = open(path, O_RDONLY);
    if (buffered < 0) return false;
    size_t got = 0;
    while (got < bytes) {
        const ssize_t result = pread(buffered, static_cast<char *>(destination) + got, bytes - got, (off_t) (slice.off + slice.pre + got));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) break;
        got += (size_t) result;
    }
    close(buffered);
    return got == bytes;
}

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : MODEL;
    const int NL = argc > 2 ? atoi(argv[2]) : 48;
    const int NE = argc > 3 ? atoi(argv[3]) : 512;
    const int NU = argc > 4 ? atoi(argv[4]) : 10;

    moe_pager pager;
    std::string err;
    if (!pager.init(model_path, NL, NE, NU, err)) {
        fprintf(stderr, "FAIL: pager.init(%s): %s\n", model_path, err.c_str());
        return 1;
    }
    printf("pager: layers=%d experts=%d used=%d bytes/expert=%zu (%.3f MiB)\n",
           pager.n_layer(), pager.n_expert(), pager.n_used(),
           pager.bytes_per_expert(), pager.bytes_per_expert() / 1048576.0);

    int fd = open(model_path, O_RDONLY);
    if (fd < 0) { perror("open model"); return 1; }

    // Native expert sizes exclude direct-I/O alignment padding.
    const int probes[][2] = { {0,0}, {0,1}, {std::min(7, NL-1), std::min(123, NE-1)}, {std::min(23, NL-1), std::min(511, NE-1)}, {NL-1, NE-1} };
    int bad = 0;
    size_t biggest = 0;
    for (auto & pr : probes) {
        for (int r = 0; r < 3; r++) biggest = std::max(biggest, pager.expert_role_bytes(pr[0], r));
    }
    char *h_ref = (char *) malloc(biggest);
    char *h_pag = (char *) malloc(biggest);
    char *d_dst = nullptr;
    cudaMalloc(&d_dst, 3 * biggest);
    cudaMemset(d_dst, 0, 3 * biggest);
    for (auto & pr : probes) {
        const int il = pr[0], e = pr[1];
        const moe_expert_slice * sl = pager.slices(il, e);
        size_t want[3];
        for (int r = 0; r < 3; r++) want[r] = pager.expert_role_bytes(il, r);
        void *dg = d_dst, *du = d_dst + want[0], *dd = d_dst + want[0] + want[1];
        if (!pager.fetch(il, e, dg, want[0], du, want[1], dd, want[2])) {
            fprintf(stderr, "FAIL: fetch(%d,%d)\n", il, e); bad = 1; continue;
        }
        bool ok = true;
        const void *dsts[3] = { dg, du, dd };
        const char *roles[3] = { "gate", "up", "down" };
        for (int r = 0; r < 3 && ok; r++) {
            if (!read_reference(sl[r].fd >= 0 ? sl[r].fd : fd, sl[r], h_ref, want[r])) {
                fprintf(stderr, "FAIL: reference pread L%d E%d %s\n", il, e, roles[r]); bad = 1; ok = false; break;
            }
            cudaMemcpy(h_pag, dsts[r], want[r], cudaMemcpyDeviceToHost);
            if (memcmp(h_ref, h_pag, want[r]) != 0) ok = false;
        }
        printf("  fetch(L%2d,E%3d): compared %zu bytes  %s\n",
               il, e, want[0] + want[1] + want[2], ok ? "EXACT MATCH" : "*** MISMATCH ***");
        if (!ok) bad = 1;
    }
    // Async path: queue prefetches, then consume them via commit and verify
    // byte-exactness. This is the exact sequence the prefetch pipeline uses:
    // non-blocking reads issued ahead, demand commits that find them in flight.
    // Use per-layer pool geometry (gate/up/down byte sizes differ by layer).
    {
        const int aprobes[][2] = { {std::min(1, NL-1), std::min(19, NE-1)}, {std::min(8, NL-1), std::min(44, NE-1)}, {std::min(11, NL-1), std::min(300, NE-1)}, {std::min(37, NL-1), std::min(7, NE-1)} };
        // issue all prefetches first so reads overlap in the ring
        for (auto & pr : aprobes) {
            if (!pager.prefetch(pr[0], pr[1])) {
                fprintf(stderr, "FAIL: prefetch(%d,%d) refused\n", pr[0], pr[1]); bad = 1;
            }
        }
        for (auto & pr : aprobes) {
            const int il = pr[0], e = pr[1];
            const moe_expert_slice * sl = pager.slices(il, e);
            size_t want[3];
            for (int r = 0; r < 3; r++) want[r] = pager.expert_role_bytes(il, r);
            // destination sized generously; commit only writes want[] bytes
            static char *d_dst = nullptr;
            static size_t d_cap = 0;
            size_t need = want[0] + want[1] + want[2];
            if (need > d_cap) {
                if (d_dst) cudaFree(d_dst);
                cudaMalloc(&d_dst, need);
                d_cap = need;
            }
            void *dg = d_dst, *du = d_dst + want[0], *dd = d_dst + want[0] + want[1];
            if (!pager.commit(il, e, dg, want[0], du, want[1], dd, want[2])) {
                fprintf(stderr, "FAIL: commit(%d,%d)\n", il, e); bad = 1; continue;
            }
            bool ok = true;
            const void *dsts[3] = { dg, du, dd };
            for (int r = 0; r < 3 && ok; r++) {
                std::vector<char> ref(want[r]), got(want[r]);
                if (!read_reference(sl[r].fd >= 0 ? sl[r].fd : fd, sl[r], ref.data(), want[r])) {
                    fprintf(stderr, "FAIL: reference pread async L%d E%d r%d\n", il, e, r); bad = 1; ok = false; break;
                }
                cudaMemcpy(got.data(), dsts[r], want[r], cudaMemcpyDeviceToHost);
                if (memcmp(ref.data(), got.data(), want[r]) != 0) { ok = false; }
            }
            printf("  async(L%2d,E%3d): %zu bytes via prefetch+commit  %s (pf_hit=%ld)\n",
                   il, e, need, ok ? "EXACT MATCH" : "*** MISMATCH ***", pager.stats_pf_hit());
            if (!ok) bad = 1;
        }
    }
    // Caller-owned host copies must survive reuse of the pager staging arena.
    {
        const int layer = NL / 2;
        const int expert = NE / 3;
        const moe_expert_slice * slices = pager.slices(layer, expert);
        const size_t gate_bytes = pager.expert_role_bytes(layer, 0);
        const size_t up_bytes   = pager.expert_role_bytes(layer, 1);
        const size_t down_bytes = pager.expert_role_bytes(layer, 2);
        const size_t total_bytes = gate_bytes + up_bytes + down_bytes;

        char * device = nullptr;
        char * cache = nullptr;
        if (cudaMalloc(&device, total_bytes) != cudaSuccess ||
            cudaHostAlloc(&cache, total_bytes, cudaHostAllocDefault) != cudaSuccess) {
            bad = 1;
        } else {
            memset(cache, 0x5A, total_bytes);
            moe_batch_target bad_cap = {expert, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes - 1};
            bool rejected_cap = !pager.fetch_batch(layer, &bad_cap, 1, nullptr);
            bool cap_intact = std::all_of(cache, cache + total_bytes, [](char byte) { return byte == 0x5A; });
            printf("  capacity-rejection: %s\n", (rejected_cap && cap_intact) ? "PASS" : "FAIL");
            if (!rejected_cap || !cap_intact) bad = 1;

            memset(cache, 0x3C, total_bytes);
            moe_batch_target bad_size = {expert, device, gate_bytes - 1, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
            bool rejected_size = !pager.fetch_batch(layer, &bad_size, 1, nullptr);
            bool size_intact = std::all_of(cache, cache + total_bytes, [](char byte) { return byte == 0x3C; });
            printf("  size-rejection: %s\n", (rejected_size && size_intact) ? "PASS" : "FAIL");
            if (!rejected_size || !size_intact) bad = 1;

            memset(cache, 0, total_bytes);
            moe_batch_target target = {expert, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
            bool exact = pager.fetch_batch(layer, &target, 1, nullptr);
            size_t offset = 0;
            for (int role = 0; role < 3 && exact; ++role) {
                const size_t bytes = pager.expert_role_bytes(layer, role);
                std::vector<char> reference(bytes), actual(bytes);
                exact = read_reference(slices[role].fd, slices[role], reference.data(), bytes) &&
                        cudaMemcpy(actual.data(), device + offset, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        reference == actual &&
                        memcmp(reference.data(), cache + offset, bytes) == 0;
                offset += bytes;
            }
            printf("  write-allocate(L%2d,E%3d): %zu bytes in host and device %s\n",
                   layer, expert, total_bytes, exact ? "EXACT MATCH" : "MISMATCH");
            if (!exact) bad = 1;

            const int expert2 = (expert + 1) % NE;
            const moe_expert_slice * slices2 = pager.slices(layer, expert2);
            const size_t total_bytes2 = pager.expert_role_bytes(layer, 0) + pager.expert_role_bytes(layer, 1) + pager.expert_role_bytes(layer, 2);
            if (total_bytes2 <= total_bytes) {
                moe_batch_target target2 = {expert2, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
                bool reuse_ok = pager.fetch_batch(layer, &target2, 1, nullptr);
                size_t off2 = 0;
                for (int role = 0; role < 3 && reuse_ok; ++role) {
                    const size_t bytes = pager.expert_role_bytes(layer, role);
                    std::vector<char> reference(bytes), actual(bytes);
                    reuse_ok = read_reference(slices2[role].fd, slices2[role], reference.data(), bytes) &&
                               cudaMemcpy(actual.data(), device + off2, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                               reference == actual &&
                               memcmp(reference.data(), cache + off2, bytes) == 0;
                    off2 += bytes;
                }
                bool refetch_ok = reuse_ok && pager.fetch_batch(layer, &target, 1, nullptr);
                size_t off1 = 0;
                for (int role = 0; role < 3 && refetch_ok; ++role) {
                    const size_t bytes = pager.expert_role_bytes(layer, role);
                    std::vector<char> reference(bytes);
                    refetch_ok = read_reference(slices[role].fd, slices[role], reference.data(), bytes) &&
                                 memcmp(reference.data(), cache + off1, bytes) == 0;
                    off1 += bytes;
                }
                printf("  reuse-stability(L%2d,E%3d<->E%3d): %s\n", layer, expert, expert2, refetch_ok ? "PASS" : "FAIL");
                if (!refetch_ok) bad = 1;
            }

            // Consume an outstanding prefetch through the batched path.
            const int expert_pref = (expert + 5) % NE;
            if (pager.prefetch(layer, expert_pref)) {
                pager.reap();
                moe_batch_target target_pref = {expert_pref, device, gate_bytes, device + gate_bytes, up_bytes, device + gate_bytes + up_bytes, down_bytes, cache, total_bytes};
                const long pf_hit_before = pager.stats_pf_hit();
                bool pref_ok = pager.fetch_batch(layer, &target_pref, 1, nullptr);
                const moe_expert_slice * slices_pref = pager.slices(layer, expert_pref);
                size_t off_pref = 0;
                for (int role = 0; role < 3 && pref_ok; ++role) {
                    const size_t bytes = pager.expert_role_bytes(layer, role);
                    std::vector<char> reference(bytes);
                    pref_ok = read_reference(slices_pref[role].fd, slices_pref[role], reference.data(), bytes) &&
                              memcmp(reference.data(), cache + off_pref, bytes) == 0;
                    off_pref += bytes;
                }
                printf("  prefetch-batch(L%2d,E%3d): %s (pf_hit=%ld->%ld)\n",
                       layer, expert_pref, pref_ok ? "PASS" : "FAIL", pf_hit_before, pager.stats_pf_hit());
                if (!pref_ok) bad = 1;
            }

            printf("  telemetry: disk_wait_us=%ld dma_ms=%.3f dma_wait_us=%ld dma_bytes=%ld\n",
                   (long) pager.stats_batch_disk_wait_us(), pager.stats_batch_h2d_ms(),
                   (long) pager.stats_dma_wait_us(), (long) pager.stats_dma_bytes());
        }
        if (device) cudaFree(device);
        if (cache) cudaFreeHost(cache);
    }
    // Mix host-cache hits and pending prefetches, then reuse the staging arena.
    if (!pager.init_cache(pager.bytes_per_expert() * 4)) bad = 1;
    const int batch_layers[] = {0, NL / 2, NL - 1};
    for (int il : batch_layers) {
        const int count = std::min(36, NE);
        size_t want[3];
        for (int role = 0; role < 3; ++role) want[role] = pager.expert_role_bytes(il, role);
        const size_t stride = want[0] + want[1] + want[2];
        char * device = nullptr;
        if (cudaMalloc(&device, stride * (size_t) count) != cudaSuccess) { bad = 1; break; }
        if (!pager.commit(il, 0, device, want[0], device + want[0], want[1], device + want[0] + want[1], want[2])) bad = 1;
        if (count > 1 && !pager.prefetch(il, count - 1)) bad = 1;
        std::vector<moe_batch_target> targets((size_t) count);
        for (int e = 0; e < count; ++e) {
            char * destination = device + (size_t) e * stride;
            targets[(size_t) e] = {e, destination, want[0], destination + want[0], want[1], destination + want[0] + want[1], want[2]};
        }
        for (int pass = 0; pass < (NE > count ? 2 : 1); ++pass) {
            if (pass == 1) {
                for (int e = 0; e < count; ++e) targets[(size_t) e].expert = (e + count) % NE;
            }
            if (!pager.fetch_batch(il, targets.data(), count)) { bad = 1; break; }
            bool exact = true;
            for (int e = 0; e < count; ++e) {
                const moe_expert_slice * slices = pager.slices(il, targets[(size_t) e].expert);
                size_t offset = 0;
                for (int role = 0; role < 3; ++role) {
                    std::vector<char> reference(want[role]), actual(want[role]);
                    if (!read_reference(slices[role].fd, slices[role], reference.data(), want[role]) ||
                            cudaMemcpy(actual.data(), device + (size_t) e * stride + offset, want[role], cudaMemcpyDeviceToHost) != cudaSuccess || reference != actual) exact = false;
                    offset += want[role];
                }
            }
            printf("  batch(L%2d,N%2d,pass%d): %s\n", il, count, pass, exact ? "EXACT MATCH" : "MISMATCH");
            if (!exact) bad = 1;
        }
        cudaFree(device);
    }
    close(fd);
    free(h_ref); free(h_pag);
    cudaFree(d_dst);

    printf("\nresult: %s   io_ops=%ld io_bytes=%ld pf_hit=%ld disk_wait_us=%ld dma_ms=%.3f\n",
           bad ? "FAIL" : "PASS", pager.stats_io_ops(), pager.stats_io_bytes(), pager.stats_pf_hit(),
           (long) pager.stats_batch_disk_wait_us(), pager.stats_batch_h2d_ms());
    return bad;
}
