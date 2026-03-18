/* SPDX-License-Identifier: MIT */
/*
 * GGML Rocket NPU Operations
 *
 * Implements tensor operations for Rocket NPU backend.
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
}

/* Forward declarations */
struct ggml_backend_rocket_context;

/**
 * Compute matmul operation on Rocket NPU
 *
 * Implements: C = A @ B^T
 * Where A is input (M x K), B is weights (N x K), C is output (M x N)
 */
static enum ggml_status ggml_backend_rocket_compute_mul_mat(
    struct ggml_backend_rocket_context * ctx,
    struct ggml_tensor * tensor) {

    /* Get input tensors */
    struct ggml_tensor * src0 = tensor->src[0];  /* A: input (M x K) */
    struct ggml_tensor * src1 = tensor->src[1];  /* B: weights (N x K) */

    if (!src0 || !src1) {
        return GGML_STATUS_FAILED;
    }

    /* Get dimensions */
    int64_t m = src0->ne[1];  /* rows of A */
    int64_t k = src0->ne[0];  /* cols of A = rows of B */
    int64_t n = src1->ne[1];  /* cols of B (output channels) */

    /* Validate dimensions for NPU */
    /* NPU requires: M multiple of 4 (or 1), K multiple of 32, N multiple of 16 */
    if ((m != 1 && m % 4 != 0) || k % 32 != 0 || n % 16 != 0) {
        /* Dimensions not supported by NPU, fall back to CPU */
        return GGML_STATUS_FAILED;
    }

    /* Check if weights are quantized */
    bool weights_quantized = ggml_is_quantized(src1->type);

    /* For now, we only support FP16 input and quantized weights */
    if (src0->type != GGML_TYPE_F16) {
        /* Input must be FP16 for NPU */
        return GGML_STATUS_FAILED;
    }

    if (!weights_quantized && src1->type != GGML_TYPE_F16) {
        /* Weights must be quantized or FP16 */
        return GGML_STATUS_FAILED;
    }

    /* TODO: Implement actual matmul execution:
     * 1. Allocate DMA buffers for input, weights, output
     * 2. Copy input and weights to DMA buffers
     * 3. Dequantize weights if needed
     * 4. Call gen_matmul_fp16() to generate register commands
     * 5. Submit to NPU via rocket_submit()
     * 6. Wait for completion
     * 7. Copy output back to tensor
     */

    /* For now, fall back to CPU */
    return GGML_STATUS_FAILED;
}

/**
 * Compute a tensor operation on Rocket NPU
 *
 * For now, we only support matmul operations.
 * Other operations fall back to CPU.
 */
static enum ggml_status ggml_backend_rocket_compute_op(
    struct ggml_backend_rocket_context * ctx,
    struct ggml_tensor * tensor) {

    switch (tensor->op) {
        case GGML_OP_MUL_MAT:
            return ggml_backend_rocket_compute_mul_mat(ctx, tensor);

        default:
            /* All other operations not supported on NPU */
            return GGML_STATUS_FAILED;
    }
}

/**
 * Graph compute - execute all operations in the computation graph
 *
 * For Phase 1c, we return GGML_STATUS_FAILED to indicate that
 * all operations should fall back to CPU. This allows the backend
 * to be registered and discovered without actually executing on NPU yet.
 *
 * In Phase 1c.2, we will implement actual NPU execution here.
 */
enum ggml_status ggml_backend_rocket_graph_compute(
    ggml_backend_t backend,
    ggml_cgraph * cgraph) {

    struct ggml_backend_rocket_context * ctx =
        (struct ggml_backend_rocket_context *)backend->context;

    if (!ctx || !cgraph) {
        return GGML_STATUS_FAILED;
    }

    /* TODO: Implement actual graph computation on NPU
     * For now, fall back to CPU for all operations
     */

    return GGML_STATUS_FAILED;
}

