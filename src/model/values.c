#include "model.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

void nn_value_dispose(NNValue *value) {
    if (!value) return;
    if (value->type == NN_VALUE_STRING) free(value->as.string);
    else if (value->type == NN_VALUE_ARRAY) {
        for (size_t i = 0; value->as.array.items && i < value->as.array.count; ++i)
            nn_value_dispose(&value->as.array.items[i]);
        free(value->as.array.items);
    }
    memset(value, 0, sizeof(*value));
}

bool nn_value_copy(NNValue *destination, const NNValue *source) {
    if (!destination || !source) return false;
    NNValue copy = { .type = source->type };
    switch (source->type) {
    case NN_VALUE_BOOL: copy.as.boolean = source->as.boolean; break;
    case NN_VALUE_INT: copy.as.integer = source->as.integer; break;
    case NN_VALUE_REAL: copy.as.real = source->as.real; break;
    case NN_VALUE_STRING:
        if (!source->as.string || !(copy.as.string = nn_text_copy(source->as.string))) return false;
        break;
    case NN_VALUE_ARRAY:
        if (source->as.array.count && !source->as.array.items) return false;
        if (source->as.array.count > (size_t)-1 / sizeof(NNValue)) return false;
        if (source->as.array.count) {
            copy.as.array.items = calloc(source->as.array.count, sizeof(NNValue));
            if (!copy.as.array.items) return false;
            for (size_t i = 0; i < source->as.array.count; ++i) {
                if (!nn_value_copy(&copy.as.array.items[i], &source->as.array.items[i])) {
                    copy.as.array.count = i;
                    nn_value_dispose(&copy);
                    return false;
                }
            }
            copy.as.array.count = source->as.array.count;
        }
        break;
    default: return false;
    }
    *destination = copy;
    return true;
}
