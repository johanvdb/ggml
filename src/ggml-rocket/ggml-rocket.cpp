#include "ggml-rocket.h"
#include "ggml-backend-impl.h"
#include "../ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

// Forward declarations - will be implemented in separate files
extern "C" {
    // From librocket
    struct rocket_ctx;
    struct rocket_bo;
    
    int rocket_open(struct rocket_ctx *ctx);
    void rocket_close(struct rocket_ctx *ctx);
    int rocket_bo_create(struct rocket_ctx *ctx, struct rocket_bo *bo, size_t size);
    void rocket_bo_destroy(struct rocket_ctx *ctx, struct rocket_bo *bo);
    void *rocket_bo_map(struct rocket_ctx *ctx, struct rocket_bo *bo);
    void rocket_bo_unmap(struct rocket_bo *bo);
    int rocket_bo_prep(struct rocket_ctx *ctx, struct rocket_bo *bo, int64_t timeout_ns);
    int rocket_bo_fini(struct rocket_ctx *ctx, struct rocket_bo *bo);
}

struct ggml_backend_rocket_context {
    struct rocket_ctx * npu_ctx;
    bool cpu_dequant;
};

struct ggml_backend_rocket_buffer {
    struct rocket_bo * bo;
    size_t size;
};

static const char * ggml_backend_rocket_name(ggml_backend_t backend) {
    (void)backend;
    return "Rocket";
}

static void ggml_backend_rocket_free(ggml_backend_t backend) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)backend->context;
    if (ctx) {
        if (ctx->npu_ctx) {
            rocket_close(ctx->npu_ctx);
            free(ctx->npu_ctx);
        }
        free(ctx);
    }
    free(backend);
}

static ggml_backend_buffer_t ggml_backend_rocket_alloc_buffer(ggml_backend_t backend, size_t size) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)backend->context;
    
    struct ggml_backend_rocket_buffer * buf = (struct ggml_backend_rocket_buffer *)malloc(sizeof(*buf));
    if (!buf) return NULL;
    
    buf->bo = (struct rocket_bo *)malloc(sizeof(struct rocket_bo));
    if (!buf->bo) {
        free(buf);
        return NULL;
    }
    
    int ret = rocket_bo_create(ctx->npu_ctx, buf->bo, size);
    if (ret < 0) {
        fprintf(stderr, "Failed to allocate Rocket buffer: %d\n", ret);
        free(buf->bo);
        free(buf);
        return NULL;
    }
    
    buf->size = size;
    
    // Create GGML buffer wrapper
    ggml_backend_buffer_t ggml_buf = (ggml_backend_buffer_t)malloc(sizeof(struct ggml_backend_buffer));
    if (!ggml_buf) {
        rocket_bo_destroy(ctx->npu_ctx, buf->bo);
        free(buf->bo);
        free(buf);
        return NULL;
    }
    
    ggml_buf->context = buf;
    ggml_buf->backend = backend;
    ggml_buf->iface.get_name = NULL;  // Will be set by GGML
    ggml_buf->iface.free_buffer = NULL;  // Will be set by GGML
    
    return ggml_buf;
}

static ggml_backend_t ggml_backend_rocket_init_impl(void) {
    struct ggml_backend_rocket_context * ctx = (struct ggml_backend_rocket_context *)malloc(sizeof(*ctx));
    if (!ctx) return NULL;
    
    ctx->npu_ctx = (struct rocket_ctx *)malloc(sizeof(struct rocket_ctx));
    if (!ctx->npu_ctx) {
        free(ctx);
        return NULL;
    }
    
    int ret = rocket_open(ctx->npu_ctx);
    if (ret < 0) {
        fprintf(stderr, "Failed to open Rocket NPU: %d\n", ret);
        free(ctx->npu_ctx);
        free(ctx);
        return NULL;
    }
    
    ctx->cpu_dequant = true;  // Default to CPU dequantization
    
    // Create GGML backend
    ggml_backend_t backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));
    if (!backend) {
        rocket_close(ctx->npu_ctx);
        free(ctx->npu_ctx);
        free(ctx);
        return NULL;
    }
    
    backend->context = ctx;
    backend->iface.get_name = ggml_backend_rocket_name;
    backend->iface.free = ggml_backend_rocket_free;
    backend->iface.alloc_buffer = ggml_backend_rocket_alloc_buffer;
    
    fprintf(stderr, "Rocket NPU backend initialized\n");
    return backend;
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

