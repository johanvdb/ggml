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

// ============================================================================
// Backend Registration Interface (for GGML device enumeration)
// ============================================================================

struct ggml_backend_rocket_reg_context {
    bool device_available;
};

static const char * ggml_backend_rocket_reg_get_name(ggml_backend_reg_t reg) {
    (void)reg;
    return "Rocket";
}

static size_t ggml_backend_rocket_reg_get_device_count(ggml_backend_reg_t reg) {
    struct ggml_backend_rocket_reg_context * ctx = (struct ggml_backend_rocket_reg_context *)reg->context;

    if (!ctx) {
        return 0;
    }

    // Check if Rocket device is available
    // We do this by trying to open it
    struct rocket_ctx test_ctx;
    int ret = rocket_open(&test_ctx);

    if (ret == 0) {
        // Device is available
        rocket_close(&test_ctx);
        ctx->device_available = true;
        return 1;  // One Rocket NPU device
    } else {
        // Device not available
        ctx->device_available = false;
        return 0;  // No devices
    }
}

static ggml_backend_dev_t ggml_backend_rocket_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    struct ggml_backend_rocket_reg_context * ctx = (struct ggml_backend_rocket_reg_context *)reg->context;

    if (!ctx || index != 0 || !ctx->device_available) {
        return NULL;
    }

    // Return a device handle (for now, just return a non-null pointer)
    // In a full implementation, this would be a proper device structure
    return (ggml_backend_dev_t)ctx;
}

static const struct ggml_backend_reg_i ggml_backend_rocket_reg_i = {
    /* .get_name         = */ ggml_backend_rocket_reg_get_name,
    /* .get_device_count = */ ggml_backend_rocket_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rocket_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_rocket_reg(void) {
    static struct ggml_backend_rocket_reg_context ctx = {
        .device_available = false,
    };

    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rocket_reg_i,
        /* .context     = */ &ctx,
    };

    return &reg;
}

// ============================================================================
// Direct Backend Initialization (for explicit use)
// ============================================================================

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

