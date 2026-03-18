/* SPDX-License-Identifier: MIT */
/*
 * GGML Rocket NPU Matmul Execution
 *
 * Implements actual matmul execution on Rocket NPU.
 * Register command format based on reverse engineering by Jasbir Matharu (mtx512).
 */

#include "ggml-rocket.h"
#include "ggml-backend-impl.h"
#include "../ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

extern "C" {
#include "rocket_interface.h"
#include "npu_matmul.h"
}

/**
 * Execute matmul on Rocket NPU
 *
 * Implements: C = A @ B^T
 * Where A is input (M x K), B is weights (N x K), C is output (M x N)
 *
 * @param ctx: Rocket backend context
 * @param src0: Input tensor A (M x K, FP16)
 * @param src1: Weight tensor B (N x K, quantized or FP16)
 * @param dst: Output tensor C (M x N, FP32)
 * @return 0 on success, -1 on failure
 */
int ggml_rocket_matmul_exec(
    struct ggml_backend_rocket_context * ctx,
    struct ggml_tensor * src0,
    struct ggml_tensor * src1,
    struct ggml_tensor * dst) {

    if (!ctx || !src0 || !src1 || !dst) {
        return -1;
    }

    /* Get dimensions */
    int64_t m = src0->ne[1];  /* rows of A */
    int64_t k = src0->ne[0];  /* cols of A = rows of B */
    int64_t n = src1->ne[1];  /* cols of B (output channels) */

    /* Validate dimensions */
    if ((m != 1 && m % 4 != 0) || k % 32 != 0 || n % 16 != 0) {
        return -1;
    }

    /* TODO: Implement actual execution:
     * 
     * 1. Allocate DMA buffers:
     *    - input_bo: for input tensor (M x K x 2 bytes for FP16)
     *    - weights_bo: for weight tensor (N x K x 1 or 2 bytes)
     *    - output_bo: for output tensor (M x N x 4 bytes for FP32)
     *
     * 2. Copy data to DMA buffers:
     *    - Copy src0 data to input_bo
     *    - Copy/dequantize src1 data to weights_bo
     *
     * 3. Generate matmul commands:
     *    - Create matmul_params_t structure
     *    - Call gen_matmul_fp16() to generate register commands
     *
     * 4. Submit to NPU:
     *    - Call rocket_submit() with generated commands
     *    - Wait for completion with rocket_wait()
     *
     * 5. Copy output:
     *    - Copy output_bo data to dst tensor
     *
     * 6. Cleanup:
     *    - Free DMA buffers
     */

    /* For Phase 1c, return failure to fall back to CPU */
    return -1;
}

