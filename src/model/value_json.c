#include "model/value_json.h"
#include "utils/utils.h"
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>

static bool parse_value(yyjson_val *source, NNValue *target, unsigned depth)
{
    if (depth > 64) return false;
    memset(target, 0, sizeof(*target));
    if (yyjson_is_bool(source)) {
        target->type = NN_VALUE_BOOL;
        target->as.boolean = yyjson_get_bool(source);
    } else if (yyjson_is_int(source)) {
        if (yyjson_is_uint(source) && yyjson_get_uint(source) > LLONG_MAX) return false;
        target->type = NN_VALUE_INT;
        target->as.integer = yyjson_get_sint(source);
    } else if (yyjson_is_real(source)) {
        target->type = NN_VALUE_REAL;
        target->as.real = yyjson_get_real(source);
        if (!isfinite(target->as.real)) return false;
    } else if (yyjson_is_str(source)) {
        if (strlen(yyjson_get_str(source)) != yyjson_get_len(source)) return false;
        target->type = NN_VALUE_STRING;
        target->as.string = nn_text_copy(yyjson_get_str(source));
        if (!target->as.string) return false;
    } else if (yyjson_is_arr(source)) {
        target->type = NN_VALUE_ARRAY;
        size_t count = yyjson_arr_size(source);
        if (count > 1024) return false;
        target->as.array.items = calloc(count ? count : 1,
                                        sizeof(NNValue));
        if (!target->as.array.items) return false;
        target->as.array.count = count;
        for (size_t i = 0; i < target->as.array.count; ++i)
            if (!parse_value(yyjson_arr_get(source, i), &target->as.array.items[i], depth + 1)) {
                nn_value_dispose(target);
                return false;
            }
    } else if (yyjson_is_obj(source)) {
        size_t count = yyjson_obj_size(source);
        if (count > 1024) return false;
        target->type = NN_VALUE_OBJECT;
        target->as.object.items = calloc(count ? count : 1, sizeof(NNParameter));
        if (!target->as.object.items) return false;
        yyjson_obj_iter iter = yyjson_obj_iter_with(source);
        yyjson_val *key;
        while ((key = yyjson_obj_iter_next(&iter))) {
            NNParameter *item = &target->as.object.items[target->as.object.count++];
            const char *text = yyjson_get_str(key);
            if (!text || !*text || strlen(text) != yyjson_get_len(key)) {
                nn_value_dispose(target); return false;
            }
            for (size_t i = 0; i + 1 < target->as.object.count; ++i)
                if (!strcmp(text, target->as.object.items[i].key)) {
                    nn_value_dispose(target); return false;
                }
            item->key = nn_text_copy(text);
            if (!item->key || !parse_value(yyjson_obj_iter_get_val(key), &item->value, depth + 1)) {
                nn_value_dispose(target); return false;
            }
        }
    } else return false;
    return true;
}

bool nn_value_from_json(yyjson_val *source, NNValue *target)
{
    return parse_value(source, target, 0);
}
