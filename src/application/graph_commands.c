#include "application_internal.h"
#include "application/application.h"
#include "utils/utils.h"
#include "inference/inference.h"
#include "yyjson.h"

#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool generated_child_id(const NNModel *model, const char *owner_id,
                               const char *kind, char *result, size_t capacity,
                               char *error, size_t error_capacity)
{
    for (unsigned attempt = 0; attempt < 10000; ++attempt) {
        int length = attempt
            ? snprintf(result, capacity, "%s-%s-%u", owner_id, kind, attempt)
            : snprintf(result, capacity, "%s-%s", owner_id, kind);
        if (length < 0 || (size_t)length >= capacity)
            return nn_fail(error, error_capacity, "generated subflow child ID is too long");
        bool unique = nn_model_find_node(model, result) == NULL;
        for (size_t i = 0; unique && i < nn_model_edge_count(model); ++i)
            if (!strcmp(nn_model_edge_at(model, i)->id, result)) unique = false;
        if (unique) return true;
    }
    return nn_fail(error, error_capacity, "unable to generate a unique subflow child ID");
}

static void rollback_subflow(NNModel *model, const char *owner_id,
                             char child_ids[][256], size_t child_count)
{
    char ignored[64];
    while (child_count)
        (void)nn_model_remove_node(model, child_ids[--child_count], ignored, sizeof(ignored));
    (void)nn_model_remove_node(model, owner_id, ignored, sizeof(ignored));
}

static bool add_package_defaults(NNApplication *app, NNModel *model,
                                 const char *id, const NNPackage *package,
                                 char *error, size_t capacity)
{
    for (size_t i = 0; i < package->parameter_count; ++i)
        if (!nn_app_add_default(app, model, id, package, &package->parameters[i],
                                error, capacity)) return false;
    return true;
}

bool nn_app_add_node(NNApplication *app, const char *id, const char *package_id,
                     const char *version, const char *scope, double x, double y,
                     char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!id || !*id || !package_id || !version || !scope || !isfinite(x) || !isfinite(y))
        return nn_fail(error, cap, "invalid node fields or non-finite position");
    const NNPackage *package = nn_catalog_find(nn_project_catalog(app->project),
                                                package_id, version);
    if (!package) return nn_fail(error, cap, "package is not active in this project");
    const NNCatalog *catalog = nn_project_catalog(app->project);
    const NNPackage *input_package = nn_app_kind_is(package, "subflow")
        ? nn_catalog_find(catalog, "core.input", "0.1.0") : NULL;
    if (nn_app_kind_is(package, "subflow") && !input_package)
        return nn_fail(error, cap, "required core.input package is unavailable");
    NNModel *model = nn_project_model(app->project);
    if (*scope) {
        const NNNode *owner = nn_model_find_node(model, scope);
        if (!owner || !nn_app_kind_is(nn_app_find_package(app, owner), "subflow"))
            return nn_fail(error, cap, "scope must name an existing subflow");
    }
    char local_error[256] = "";
    if (!nn_model_add_node(model, id, package->name, package->id, package->version,
                           scope, x, y, local_error, sizeof(local_error)))
        return nn_fail(error, cap, local_error);
    for (size_t i = 0; i < package->parameter_count; ++i) {
        const NNParameterDef *definition = &package->parameters[i];
        /* Preserve the existing editor behavior for object-valued defaults. */
        if (!strcmp(definition->type, "stereotype") && definition->has_default &&
            definition->default_value.type == NN_PARAMETER_JSON) continue;
        if (!nn_app_add_default(app, model, id, package, definition, error, cap)) {
            char ignored[64];
            (void)nn_model_remove_node(model, id, ignored, sizeof(ignored));
            return false;
        }
    }
    if (nn_app_kind_is(package, "subflow")) {
        char spawned_ids[3][256] = {{0}};
        size_t spawned = 0;
        char failure[256] = "";
        if (!generated_child_id(model, id, "input", spawned_ids[spawned],
                                sizeof(spawned_ids[spawned]), failure, sizeof(failure))) {
            rollback_subflow(model, id, spawned_ids, spawned);
            return nn_fail(error, cap, failure);
        }
        if (!nn_model_add_node(model, spawned_ids[spawned], input_package->name,
                               input_package->id, input_package->version, id, 0, 0,
                               failure, sizeof(failure))) {
            rollback_subflow(model, id, spawned_ids, spawned);
            return nn_fail(error, cap, failure);
        }
        ++spawned;
        if (!add_package_defaults(app, model, spawned_ids[spawned - 1], input_package,
                                  failure, sizeof(failure))) {
            rollback_subflow(model, id, spawned_ids, spawned);
            return nn_fail(error, cap, failure);
        }
        for (size_t i = 0; i < package->output_count; ++i) {
            const NNOutputDef *output = &package->outputs[i];
            const char *terminal_id = !strcmp(output->type, "loss")
                ? "core.loss-output" : "core.output";
            const NNPackage *terminal = nn_catalog_find(catalog, terminal_id, "0.1.0");
            double terminal_x = 0;
            double terminal_y = 240.0 + (double)i * 120.0;
            char kind[256];
            int kind_length = snprintf(kind, sizeof(kind), "boundary-%s", output->id);
            bool kind_valid = kind_length >= 0 && (size_t)kind_length < sizeof(kind);
            bool unique = kind_valid && generated_child_id(
                model, id, kind, spawned_ids[spawned], sizeof(spawned_ids[spawned]),
                failure, sizeof(failure));
            if (!kind_valid) nn_error_set(failure, sizeof(failure), "generated subflow terminal ID is too long");
            if (!terminal && !failure[0])
                nn_error_set(failure, sizeof(failure), "required boundary package is unavailable");
            if (!unique && !failure[0])
                nn_error_set(failure, sizeof(failure), "unable to generate a unique subflow terminal ID");
            bool ready = terminal && unique;
            if (ready) {
                bool added = nn_model_add_node(model, spawned_ids[spawned], output->id,
                                               terminal->id, terminal->version, id,
                                               terminal_x, terminal_y,
                                               failure, sizeof(failure));
                if (added) ++spawned;
                bool mapped = added && nn_model_set_boundary_handle(
                    model, spawned_ids[spawned - 1], output->id, failure, sizeof(failure));
                ready = added && mapped;
            }
            if (!ready) {
                rollback_subflow(model, id, spawned_ids, spawned);
                if (!failure[0])
                    nn_error_set(failure, sizeof(failure), "unable to spawn subflow output terminal");
                return nn_fail(error, cap, failure);
            }
        }
    }
    nn_project_mark_dirty(app->project);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_remove_node(NNApplication *app, const char *id, char *error, size_t cap)
{
    if (!app || !app->project || !id) return nn_fail(error, cap, "no active project or invalid node ID");
    NNModel *model = nn_project_model(app->project);
    const NNNode *node = nn_model_find_node(model, id);
    if (!node) return nn_fail(error, cap, "node not found");
    if (nn_app_kind_is(nn_app_find_package(app, node), "subflow")) {
        for (size_t i = 0; i < nn_model_node_count(model); ++i) {
            const NNNode *child = nn_model_node_at(model, i);
            if (!strcmp(child->scope_id, id))
                return nn_fail(error, cap, "subflow still contains nodes");
        }
    }
    if (!nn_model_remove_node(model, id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_move_node(NNApplication *app, const char *id, double x, double y,
                      char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!isfinite(x) || !isfinite(y)) return nn_fail(error, cap, "position must be finite");
    if (!nn_model_move_node(nn_project_model(app->project), id, x, y, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_rename_node(NNApplication *app, const char *id, const char *label,
                        char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!nn_model_rename_node(nn_project_model(app->project), id, label, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_connect(NNApplication *app, const char *id, const char *source,
                    const char *source_handle, const char *target,
                    const char *target_handle, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    NNModel *model = nn_project_model(app->project);
    const NNNode *source_node = nn_model_find_node(model, source);
    const NNNode *target_node = nn_model_find_node(model, target);
    if (!source_node || !target_node) return nn_fail(error, cap, "edge endpoint not found");
    const NNPackage *source_package = nn_app_find_package(app, source_node);
    const NNPackage *target_package = nn_app_find_package(app, target_node);
    if (!source_package || !target_package) return nn_fail(error, cap, "edge package is unresolved");
    if (!nn_app_valid_output_handle(source_package, source_handle))
        return nn_fail(error, cap, "invalid output handle");
    if (!nn_app_valid_input_handle(target_package, target_handle))
        return nn_fail(error, cap, "invalid input handle");
    const char *type = nn_app_output_type(app, source, source_handle);
    if ((nn_app_kind_is(target_package, "output") && (!type || strcmp(type, "output"))) ||
        (nn_app_kind_is(target_package, "loss-output") && (!type || strcmp(type, "loss"))))
        return nn_fail(error, cap, "output type is incompatible with terminal");
    if (nn_app_kind_is(target_package, "join")) {
        size_t requested;
        (void)nn_join_handle_order(target_handle, &requested);
        for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
            const NNEdge *edge = nn_model_edge_at(model, i);
            size_t occupied;
            if (!strcmp(edge->target_id, target) &&
                nn_join_handle_order(edge->target_handle_id, &occupied) && occupied == requested)
                return nn_fail(error, cap, "join input position is already occupied");
        }
    }
    if (!nn_model_connect(model, id, source, source_handle, target, target_handle, error, cap))
        return false;
    nn_project_mark_dirty(app->project);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_disconnect(NNApplication *app, const char *id, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (!nn_model_disconnect(nn_project_model(app->project), id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}
bool nn_app_set_boundary_handle(NNApplication *app, const char *node_id,
                                const char *handle_id, char *error, size_t cap)
{
    if (!app || !app->project || !node_id || !handle_id || !*handle_id)
        return nn_fail(error, cap, "invalid boundary mapping");
    NNModel *model = nn_project_model(app->project);
    const NNNode *node = nn_model_find_node(model, node_id);
    const NNPackage *terminal = nn_app_find_package(app, node);
    if (!node || (!nn_app_kind_is(terminal, "output") && !nn_app_kind_is(terminal, "loss-output")))
        return nn_fail(error, cap, "boundary mapping requires an output terminal");
    const NNNode *owner = *node->scope_id ? nn_model_find_node(model, node->scope_id) : NULL;
    const NNPackage *owner_package = nn_app_find_package(app, owner);
    if (!owner || !nn_app_kind_is(owner_package, "subflow"))
        return nn_fail(error, cap, "root terminals cannot have boundary mappings");
    const NNOutputDef *mapping = NULL;
    for (size_t i = 0; i < owner_package->output_count; ++i)
        if (!strcmp(owner_package->outputs[i].id, handle_id)) mapping = &owner_package->outputs[i];
    if (!mapping) return nn_fail(error, cap, "unknown subflow output handle");
    const char *expected_kind = !strcmp(mapping->type, "loss") ? "loss-output" : "output";
    if (!nn_app_kind_is(terminal, expected_kind))
        return nn_fail(error, cap, "boundary handle type does not match terminal");
    if (node->boundary_handle_id && !strcmp(node->boundary_handle_id, handle_id)) {
        nn_error_set(error, cap, "");
        return true;
    }
    if (!nn_model_set_boundary_handle(model, node_id, handle_id, error, cap)) return false;
    nn_project_mark_dirty(app->project);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, cap, "");
    return true;
}
