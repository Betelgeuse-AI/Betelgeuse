#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// E2M1 maps to half with a bias shift of -14. All bits fit in the high byte.
__device__ __forceinline__ uint2 e2m1_half_bytes(uint32_t v) {
    return make_uint2(((v & 0x08080808u) << 4) | ((v & 0x07070707u) << 1),
                      (v & 0x80808080u) | ((v & 0x70707070u) >> 3));
}

template <int B>
__device__ __forceinline__ half2 e2m1_half2(uint2 v) {
    const uint32_t pairs = __byte_perm(v.x, v.y, B < 2 ? 0x5140 : 0x7362);
    const uint32_t h = __byte_perm(pairs, 0, B % 2 == 0 ? 0x1404 : 0x3424);
    return *reinterpret_cast<const half2 *>(&h);
}

// E4M3 maps to half with a bias shift of -8, including subnormal values.
template <int B>
__device__ __forceinline__ half2 e4m3_half2(uint32_t v) {
    const uint32_t a = __byte_perm(v, 0, B == 0 ? 0x4140 : 0x4342);
    uint32_t h = ((a & 0x00800080u) << 8) | ((a & 0x007f007fu) << 7);
    return *reinterpret_cast<const half2 *>(&h);
}

__device__ __forceinline__ float e4m3_scale(uint8_t v) {
    const unsigned bits = ((v & 0x80u) << 8) | ((v & 0x7fu) << 7);
    return (v & 0x7fu) == 0x7fu ? __int_as_float(0x7fc00000) : __half2float(__ushort_as_half(bits)) * 256.0f;
}

// Form the low and high half bytes for all four weights before interleaving them.
__device__ __forceinline__ uint2 e4m3_half_bytes(uint32_t v) {
    return make_uint2((v & 0x01010101u) << 7, (v & 0x80808080u) | ((v & 0x7e7e7e7eu) >> 1));
}

template <int B>
__device__ __forceinline__ half2 e4m3_half2(uint2 bytes) {
    const uint32_t h = __byte_perm(bytes.x, bytes.y, B == 0 ? 0x5140 : 0x7362);
    return *reinterpret_cast<const half2 *>(&h);
}

// Build unscaled E2M1 half pairs for the tensor core tile.
__device__ __forceinline__ uint2 e2m1_half_lut_bytes(uint32_t v) {
    const uint32_t signs = __byte_perm(v << 4, v, 0x5140);
    const uint32_t magnitudes = __byte_perm(0x3e3c3800u, 0x46444240u, v & 0x7777u);
    return make_uint2(__byte_perm((signs & 0x80808080u) | magnitudes, 0, 0x1404),
                      __byte_perm((signs & 0x80808080u) | magnitudes, 0, 0x3424));
}

// Selector bit 3 replicates the sign of a positive LUT byte, giving zero.
// Run both signs and subtract bytewise to get four exact E2M1 values times two.
__device__ __forceinline__ uint32_t e2m1_i8x4(uint32_t v) {
    uint32_t pos, neg;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(pos) : "r"(0x03020100u), "r"(0x0c080604u), "r"(v));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(neg) : "r"(0x03020100u), "r"(0x0c080604u), "r"(v ^ 0x8888u));
    return __vsub4(pos, neg);
}
