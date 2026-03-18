/* SPDX-License-Identifier: MIT */
/*
 * GGML Rocket NPU Dequantization
 *
 * Implements dequantization of quantized weights for Rocket NPU.
 * Supports both CPU dequantization and NPU-aware quantization paths.
 */

#include "ggml-rocket.h"
#include "ggml-backend-impl.h"
#include "../ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>

extern "C" {
#include "rocket_interface.h"
}

/**
 * Dequantize Q8_0 weights to FP16
 *
 * Q8_0 format: 2 bytes scale (FP16) + 32 bytes quantized values
 * Block size: 32 values per block
 *
 * @param src: Source Q8_0 data
 * @param dst: Destination FP16 data
 * @param n: Number of values to dequantize
 */
void ggml_rocket_dequant_q8_0_to_fp16(
    const uint8_t * src,
    uint16_t * dst,
    int n) {

    /* TODO: Implement Q8_0 dequantization
     * 
     * For each block of 32 values:
     * 1. Read 2-byte FP16 scale
     * 2. Read 32 int8 quantized values
     * 3. For each value: dst[i] = scale * (float)src[i]
     * 4. Convert float to FP16
     */
}

/**
 * Dequantize Q4_0 weights to FP16
 *
 * Q4_0 format: 2 bytes scale (FP16) + 16 bytes quantized values (4-bit)
 * Block size: 32 values per block
 *
 * @param src: Source Q4_0 data
 * @param dst: Destination FP16 data
 * @param n: Number of values to dequantize
 */
void ggml_rocket_dequant_q4_0_to_fp16(
    const uint8_t * src,
    uint16_t * dst,
    int n) {

    /* TODO: Implement Q4_0 dequantization
     * 
     * For each block of 32 values:
     * 1. Read 2-byte FP16 scale
     * 2. Read 16 bytes of 4-bit quantized values
     * 3. For each value: dst[i] = scale * (float)((nibble - 8) / 8.0)
     * 4. Convert float to FP16
     */
}

/**
 * Dequantize quantized weights to FP16
 *
 * Supports Q4_0, Q4_1, Q8_0, Q8_1 formats.
 * Other formats will return -1 (not supported).
 *
 * @param src: Source quantized data
 * @param dst: Destination FP16 data
 * @param n: Number of values to dequantize
 * @param type: GGML quantization type
 * @return 0 on success, -1 if type not supported
 */
int ggml_rocket_dequantize_weights(
    const uint8_t * src,
    uint16_t * dst,
    int n,
    enum ggml_type type) {

    switch (type) {
        case GGML_TYPE_Q8_0:
            ggml_rocket_dequant_q8_0_to_fp16(src, dst, n);
            return 0;

        case GGML_TYPE_Q4_0:
            ggml_rocket_dequant_q4_0_to_fp16(src, dst, n);
            return 0;

        case GGML_TYPE_F16:
            /* Already FP16, just copy */
            memcpy(dst, src, n * sizeof(uint16_t));
            return 0;

        default:
            /* Unsupported quantization type */
            return -1;
    }
}

