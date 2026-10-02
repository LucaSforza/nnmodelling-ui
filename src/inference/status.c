#include "inference/inference.h"
#include <stdint.h>
#include <stddef.h>

const char *nn_inference_category(NNInferenceStatus status)
{
    switch (status) {
    case NN_INFERENCE_SUCCESS: return "success";
    case NN_INFERENCE_COMPILATION_ERROR: return "lua-compilation";
    case NN_INFERENCE_SEMANTIC_ERROR: return "model";
    case NN_INFERENCE_UNRESOLVED: return "incomplete";
    case NN_INFERENCE_RUNTIME_FAULT: return "internal";
    }
    return "internal";
}

const char *nn_inference_severity(NNInferenceStatus status)
{
    switch (status) {
    case NN_INFERENCE_SUCCESS: return "info";
    case NN_INFERENCE_COMPILATION_ERROR:
    case NN_INFERENCE_SEMANTIC_ERROR: return "error";
    case NN_INFERENCE_UNRESOLVED: return "warning";
    case NN_INFERENCE_RUNTIME_FAULT: return "internal";
    }
    return "internal";
}

size_t nn_inference_error_line(const char *message)
{
    if (!message) return 0;
    for (const char *p = message; *p; ++p) {
        if (*p != ':') continue;
        const char *digits = p + 1;
        if (*digits < '0' || *digits > '9') continue;
        size_t line = 0;
        do {
            unsigned digit = (unsigned)(*digits - '0');
            if (line > (SIZE_MAX - digit) / 10) return 0;
            line = line * 10 + digit;
            ++digits;
        } while (*digits >= '0' && *digits <= '9');
        if (*digits == ':' && line) return line;
    }
    return 0;
}
