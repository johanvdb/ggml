/* SPDX-License-Identifier: MIT */
/*
 * GGML Rocket NPU Backend Implementation
 *
 * Register command format based on reverse engineering by Jasbir Matharu (mtx512).
 * See: https://github.com/mtx512/rk3588-npu
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

struct ggml_backend_rocket_context {
    struct rocket_ctx npu_ctx;
    bool cpu_dequant;
};

struct ggml_backend_rocket_buffer {
    struct rocket_bo bo;
    size_t size;
};

/* Forward declaration */
enum ggml_status ggml_backend_rocket_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph);

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

    ctx->cpu_dequant = true;
    return (ggml_backend_t)ctx;
}

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

    struct rocket_ctx test_ctx;
    int ret = rocket_open(&test_ctx);

    if (ret == 0) {
        rocket_close(&test_ctx);
        ctx->device_available = true;
        return 1;
    }

    ctx->device_available = false;
    return 0;
}

static ggml_backend_dev_t ggml_backend_rocket_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    struct ggml_backend_rocket_reg_context * ctx = (struct ggml_backend_rocket_reg_context *)reg->context;

    if (!ctx || index != 0 || !ctx->device_available) {
        return NULL;
    }

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

