#include "model_internal.h"
#include "model_utils.h"
#include "utils/utils.h"

#include <stdlib.h>
#include <string.h>

bool nn_model_set_parameter(NNModel *model, const char *node_id, const char *key,
                            const NNValue *value, char *error, size_t cap) {
    size_t i = nn_model_node_index(model, node_id);
    if (i == (size_t)-1) return nn_fail(error, cap, "node not found");
    if (!nn_model_valid_id(key) || !value) return nn_fail(error, cap, "invalid parameter");
    NNValue copy = {0};
    if (!nn_value_copy(&copy, value)) return nn_fail(error, cap, "invalid value or out of memory");
    NNNode *node = &model->nodes[i].view;
    for (size_t p = 0; p < node->parameter_count; ++p) {
        if (!strcmp(node->parameters[p].key, key)) {
            nn_value_dispose(&((NNParameter *)node->parameters)[p].value);
            ((NNParameter *)node->parameters)[p].value = copy;
            nn_error_set(error, cap, ""); return true;
        }
    }
    char *key_copy = nn_text_copy(key);
    if (!key_copy || node->parameter_count == (size_t)-1 / sizeof(NNParameter)) {
        free(key_copy); nn_value_dispose(&copy); return nn_fail(error, cap, "out of memory");
    }
    NNParameter *grown = realloc((void *)node->parameters, (node->parameter_count + 1) * sizeof(NNParameter));
    if (!grown) { free(key_copy); nn_value_dispose(&copy); return nn_fail(error, cap, "out of memory"); }
    grown[node->parameter_count++] = (NNParameter){ .key = key_copy, .value = copy };
    node->parameters = grown;
    nn_error_set(error, cap, ""); return true;
}
