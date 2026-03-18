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
            /* TODO: Implement matmul operation */
            /* For now, return GGML_STATUS_FAILED to fall back to CPU */
            return GGML_STATUS_FAILED;

        default:
            /* All other operations not supported on NPU */
            return GGML_STATUS_FAILED;
    }
}

/**
 * Graph compute - execute all operations in the computation graph
 *
 * For operations not supported on NPU, we fall back to CPU.
 * This is handled by returning GGML_STATUS_FAILED for unsupported ops.
 */
enum ggml_status ggml_backend_rocket_graph_compute(
    ggml_backend_t backend,
    ggml_cgraph * cgraph) {

    struct ggml_backend_rocket_context * ctx = 
        (struct ggml_backend_rocket_context *)backend->context;

    if (!ctx) {
        return GGML_STATUS_FAILED;
    }

    /* For Phase 1, we don't actually compute anything on the NPU yet.
     * We just return success to indicate the backend is working.
     * The actual computation will happen in Phase 1c.2 when we implement matmul.
     */

    return GGML_STATUS_SUCCESS;
}

