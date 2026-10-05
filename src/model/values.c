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
    } else if (value->type == NN_VALUE_OBJECT) {
        for (size_t i = 0; value->as.object.items && i < value->as.object.count; ++i) {
            free(value->as.object.items[i].key);
            nn_value_dispose(&value->as.object.items[i].value);
        }
        free(value->as.object.items);
    }
    memset(value, 0, sizeof(*value));
}

static bool copy_value(NNValue *destination, const NNValue *source, unsigned depth) {
    if (!destination || !source || depth > 64) return false;
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
                if (!copy_value(&copy.as.array.items[i], &source->as.array.items[i], depth + 1)) {
                    copy.as.array.count = i;
                    nn_value_dispose(&copy);
                    return false;
                }
            }
            copy.as.array.count = source->as.array.count;
        }
        break;
    case NN_VALUE_OBJECT:
        if (source->as.object.count > 1024 ||
            (source->as.object.count && !source->as.object.items)) return false;
        copy.as.object.items = calloc(source->as.object.count ? source->as.object.count : 1,
                                      sizeof(NNParameter));
        if (!copy.as.object.items) return false;
        for (size_t i = 0; i < source->as.object.count; ++i) {
            const NNParameter *entry = &source->as.object.items[i];
            NNParameter *item = &copy.as.object.items[i];
            ++copy.as.object.count;
            if (!entry->key || !*entry->key) { nn_value_dispose(&copy); return false; }
            for (size_t j = 0; j < i; ++j)
                if (!strcmp(entry->key, source->as.object.items[j].key)) {
                    nn_value_dispose(&copy); return false;
                }
            item->key = nn_text_copy(entry->key);
            if (!item->key || !copy_value(&item->value, &entry->value, depth + 1)) {
                nn_value_dispose(&copy); return false;
            }
        }
        break;
    default: return false;
    }
    *destination = copy;
    return true;
}

bool nn_value_copy(NNValue *destination, const NNValue *source) {
    return copy_value(destination, source, 0);
}

const NNValue *nn_value_member(const NNValue *object, const char *key) {
    if (!object || object->type != NN_VALUE_OBJECT || !key) return NULL;
    for (size_t i = 0; i < object->as.object.count; ++i)
        if (!strcmp(object->as.object.items[i].key, key)) return &object->as.object.items[i].value;
    return NULL;
}
