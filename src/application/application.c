#include "application/application_internal.h"
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

void nn_app_invalidate_analysis(NNApplication *app)
{
    if (!app) return;
    nn_inference_free(app->analysis);
    app->analysis = NULL;
}

bool nn_app_kind_is(const NNPackage *package, const char *kind)
{
    return package && package->kind && !strcmp(package->kind, kind);
}

const NNPackage *nn_app_find_package(const NNApplication *app, const NNNode *node)
{
    return app && app->project && node
        ? nn_catalog_find(nn_project_catalog(app->project), node->package_id,
                         node->package_version)
        : NULL;
}
NNApplication *nn_app_new(const char *core_root)
{
    if (!core_root || !*core_root) return NULL;
    NNApplication *app = calloc(1, sizeof(*app));
    if (!app) return NULL;
    app->core_root = nn_text_copy(core_root);
    if (!app->core_root) { free(app); return NULL; }
    nn_app_history_reset(app);
    return app;
}

void nn_app_free(NNApplication *app)
{
    if (!app) return;
    nn_app_invalidate_analysis(app);
    nn_app_history_reset(app);
    nn_project_close(app->project);
    free(app->core_root);
    free(app);
}

bool nn_app_open(NNApplication *app, const char *directory, char *error, size_t cap)
{
    if (!app || !directory || !*directory) return nn_fail(error, cap, "invalid project path");
    if (app->history.group_active) return nn_fail(error, cap, "cannot open a project during an edit group");
    NNProject *staged = nn_project_open(directory, app->core_root, error, cap);
    if (!staged) return false;
    NNProject *old = app->project;
    app->project = staged;
    nn_app_invalidate_analysis(app);
    nn_app_history_reset(app);
    nn_project_close(old);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_create(NNApplication *app, const char *parent, const char *id,
                   const char *name, bool mnist, char *error, size_t cap)
{
    if (!app) return nn_fail(error, cap, "application is null");
    if (app->history.group_active) return nn_fail(error, cap, "cannot create a project during an edit group");
    NNProject *staged = nn_project_create(parent, id, name, mnist,
                                          app->core_root, error, cap);
    if (!staged) return false;
    NNProject *old = app->project;
    app->project = staged;
    nn_app_invalidate_analysis(app);
    nn_app_history_reset(app);
    nn_project_close(old);
    nn_error_set(error, cap, "");
    return true;
}

bool nn_app_save(NNApplication *app, char *error, size_t cap)
{
    if (!app || !app->project) return nn_fail(error, cap, "no active project");
    if (app->history.group_active) return nn_fail(error, cap, "cannot save during an edit group");
    if (!nn_project_save(app->project, error, cap)) return false;
    nn_app_history_mark_saved(app);
    return true;
}

bool nn_app_close(NNApplication *app, bool discard, char *error, size_t cap)
{
    if (!app) return nn_fail(error, cap, "application is null");
    if (app->history.group_active) return nn_fail(error, cap, "cannot close during an edit group");
    if (!app->project) { nn_error_set(error, cap, ""); return true; }
    if (nn_project_dirty(app->project) && !discard &&
        !nn_project_save(app->project, error, cap)) return false;
    nn_project_close(app->project);
    app->project = NULL;
    nn_app_invalidate_analysis(app);
    nn_app_history_reset(app);
    nn_error_set(error, cap, "");
    return true;
}

const NNProject *nn_app_project(const NNApplication *app)
{
    return app ? app->project : NULL;
}

const NNModel *nn_app_model(const NNApplication *app)
{
    return app && app->project ? nn_project_model(app->project) : NULL;
}

const NNInferenceReport *nn_app_analysis(NNApplication *app, char *error, size_t cap)
{
    if (!app || !app->project) {
        nn_fail(error, cap, "no active project");
        return NULL;
    }
    if (!app->analysis) {
        app->analysis = nn_infer_project(app->project);
        if (!app->analysis) {
            nn_fail(error, cap, "unable to construct project analysis report");
            return NULL;
        }
    }
    nn_error_set(error, cap, "");
    return app->analysis;
}
