#include "model/model.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static bool add(NNModel *m, const char *id, const char *scope) {
    char error[128];
    return nn_model_add_node(m, id, id, "core.layer", "1", scope, 0, 0, error, sizeof(error));
}

static bool connect(NNModel *m, const char *id, const char *s, const char *sh,
                    const char *t, const char *th, char *error) {
    return nn_model_connect(m, id, s, sh, t, th, error, 128);
}

int main(void) {
    NNModel *m = nn_model_new();
    assert(m);
    char error[128];

    assert(add(m, "a", "root"));
    assert(add(m, "b", "root"));
    assert(add(m, "c", "root"));
    assert(add(m, "other", "nested"));
    assert(nn_model_add_node(m, "grid", "grid", "core.layer", "1", "root",
                             10, -10, error, sizeof(error)));
    assert(nn_model_find_node(m, "grid")->x == 20 && nn_model_find_node(m, "grid")->y == -20);
    assert(!add(m, "a", "root"));
    assert(nn_model_node_count(m) == 5);

    assert(connect(m, "ab", "a", "out", "b", "in", error));
    assert(connect(m, "bc", "b", "out", "c", "in", error));
    assert(!connect(m, "ca", "c", "out", "a", "in", error));
    assert(!connect(m, "occupied", "a", "other", "b", "in", error));
    assert(!connect(m, "wrong-scope", "a", "out", "other", "in", error));
    assert(nn_model_edge_count(m) == 2);
    assert(!strcmp(error, "edge endpoints have different scopes"));

    NNValue leaf = { .type = NN_VALUE_STRING, .as.string = "original" };
    NNValue array = { .type = NN_VALUE_ARRAY, .as.array = { .items = &leaf, .count = 1 } };
    assert(nn_model_set_parameter(m, "a", "weights", &array, error, sizeof(error)));
    leaf.as.string = "changed";
    const NNNode *node = nn_model_find_node(m, "a");
    assert(node && node->parameter_count == 1);
    const NNValue *stored = &node->parameters[0].value;
    assert(stored->type == NN_VALUE_ARRAY);
    assert(!strcmp(stored->as.array.items[0].as.string, "original"));

    NNValue replacement = { .type = NN_VALUE_INT, .as.integer = 42 };
    assert(nn_model_set_parameter(m, "a", "weights", &replacement, error, sizeof(error)));
    assert(node->parameters[0].value.type == NN_VALUE_INT);
    assert(node->parameters[0].value.as.integer == 42);

    assert(nn_model_move_node(m, "a", 12.5, -3, error, sizeof(error)));
    assert(nn_model_rename_node(m, "a", "renamed", error, sizeof(error)));
    char mapping[] = "prediction";
    assert(nn_model_set_boundary_handle(m, "a", mapping, error, sizeof(error)));
    mapping[0] = 'X';
    assert(!strcmp(nn_model_find_node(m, "a")->boundary_handle_id, "prediction"));
    assert(nn_model_find_node(m, "a")->x == 20 && nn_model_find_node(m, "a")->y == 0);
    assert(!nn_model_move_node(m, "a", NAN, 40, error, sizeof(error)));
    assert(strstr(error, "finite"));
    assert(nn_model_find_node(m, "a")->x == 20 && nn_model_find_node(m, "a")->y == 0);
    assert(!nn_model_move_node(m, "a", 2147483650.0, 40, error, sizeof(error)));
    assert(strstr(error, "range"));
    assert(nn_model_find_node(m, "a")->x == 20 && nn_model_find_node(m, "a")->y == 0);
    assert(!nn_model_add_node(m, "invalid", "invalid", "core.layer", "1", "root",
                              0, INFINITY, error, sizeof(error)));
    assert(nn_model_find_node(m, "invalid") == NULL);
    assert(!strcmp(nn_model_find_node(m, "a")->label, "renamed"));
    assert(nn_model_node_at(m, 5) == NULL && nn_model_edge_at(m, 2) == NULL);
    const NNNode *borrowed = nn_model_find_node(m, "c");
    assert(borrowed && nn_model_remove_node(m, borrowed->id, error, sizeof(error)));
    assert(nn_model_node_count(m) == 4 && nn_model_edge_count(m) == 1);
    assert(nn_model_remove_node(m, "b", error, sizeof(error)));
    assert(nn_model_node_count(m) == 3 && nn_model_edge_count(m) == 0);
    assert(!nn_model_disconnect(m, "ab", error, sizeof(error)));

    nn_model_free(m);
    puts("model invariants: ok");
    return 0;
}
