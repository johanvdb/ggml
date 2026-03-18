#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get Rocket backend registration
 *
 * This is called by GGML to enumerate available Rocket devices.
 * GGML will call get_device_count() to check if a Rocket NPU is available.
 *
 * @return Backend registration handle
 */
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rocket_reg(void);

/**
 * Initialize Rocket NPU backend
 *
 * @return Backend handle, or NULL on failure
 */
GGML_BACKEND_API ggml_backend_t ggml_backend_rocket_init(void);

/**
 * Set quantization strategy for matmul operations
 *
 * @param backend: Backend handle
 * @param cpu_dequant: If true, dequantize on CPU before NPU matmul
 *                     If false, use quantization-aware matmul on NPU
 */
GGML_BACKEND_API void ggml_backend_rocket_set_cpu_dequant(ggml_backend_t backend, bool cpu_dequant);

#ifdef __cplusplus
}
#endif

