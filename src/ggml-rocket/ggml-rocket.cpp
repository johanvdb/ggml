#include "ggml-rocket.h"
#include "ggml-backend-impl.h"
#include "../ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Include librocket interface
extern "C" {
#include "rocket_interface.h"
}

struct ggml_backend_rocket_context {
    struct rocket_ctx npu_ctx;
    bool cpu_dequant;
};

struct ggml_backend_rocket_buffer {
    struct rocket_bo bo;
    size_t size;
};

static const char * ggml_backend_rocket_name(ggml_backend_t backend) {
    (void)backend;
    return "Rocket";
}

static void ggml_backend_rocket_free(ggml_backend_t backend) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)backend->context;
    if (ctx) {
        rocket_close(&ctx->npu_ctx);
        free(ctx);
    }
    free(backend);
}

static ggml_backend_buffer_t ggml_backend_rocket_alloc_buffer(ggml_backend_t backend, size_t size) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)backend->context;

    struct ggml_backend_rocket_buffer * buf = (struct ggml_backend_rocket_buffer *)malloc(sizeof(*buf));
    if (!buf) return NULL;

    int ret = rocket_bo_create(&ctx->npu_ctx, &buf->bo, size);
    if (ret < 0) {
        fprintf(stderr, "Failed to allocate Rocket buffer: %d\n", ret);
        free(buf);
        return NULL;
    }

    buf->size = size;

    // For now, return the buffer context directly
    // GGML will wrap it with its own buffer structure
    return (ggml_backend_buffer_t)buf;
}

static ggml_backend_t ggml_backend_rocket_init_impl(void) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)malloc(sizeof(*ctx));
    if (!ctx) return NULL;

    int ret = rocket_open(&ctx->npu_ctx);
    if (ret < 0) {
        fprintf(stderr, "Failed to open Rocket NPU: %d\n", ret);
        free(ctx);
        return NULL;
    }

    ctx->cpu_dequant = true;  // Default to CPU dequantization

    fprintf(stderr, "Rocket NPU backend initialized\n");

    // Return context as opaque backend handle
    // The actual GGML backend wrapper will be created by GGML
    return (ggml_backend_t)ctx;
}

GGML_BACKEND_API ggml_backend_t ggml_backend_rocket_init(void) {
    return ggml_backend_rocket_init_impl();
}

GGML_BACKEND_API void ggml_backend_rocket_set_cpu_dequant(ggml_backend_t backend, bool cpu_dequant) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)backend->context;
    if (ctx) {
        ctx->cpu_dequant = cpu_dequant;
        fprintf(stderr, "Rocket quantization strategy: %s\n", cpu_dequant ? "CPU dequant" : "NPU-aware");
    }
}

