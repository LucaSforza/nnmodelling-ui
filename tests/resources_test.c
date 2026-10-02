#define _XOPEN_SOURCE 700
#include "application/application.h"
#include "inference/inference.h"

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *rule = "return function(context, parameters, services) return {status='success',output=context.inputs[1]} end";
static const char *definition =
    "{\"name\":\"Local affine\",\"kind\":\"layer\",\"description\":\"Test\","
    "\"view\":{\"color\":\"#4779c4\",\"width\":200,\"height\":100},\"parameters\":{"
    "\"features\":{\"type\":\"integer\",\"minimum\":1,\"default\":32,\"position\":\"top\"},"
    "\"factor\":{\"type\":\"number\",\"default\":1,\"position\":\"bottom\"},"
    "\"shape\":{\"type\":\"json\",\"default\":[1,2]},"
    "\"mode\":{\"type\":\"string\",\"choices\":[\"a\",\"b\"],\"default\":\"a\"}}}";
static const char *dataset = "{\"name\":\"Images\",\"batch\":{\"inputs\":{\"image\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},\"targets\":{}}}";

static void path(char *buffer, size_t cap, const char *root, const char *relative)
{
    assert(snprintf(buffer, cap, "%s/%s", root, relative) < (int)cap);
}

static char *read_text(const char *filename)
{
    FILE *file = fopen(filename, "rb"); assert(file);
    assert(fseek(file, 0, SEEK_END) == 0); long count = ftell(file); assert(count >= 0);
    rewind(file); char *text = malloc((size_t)count + 1); assert(text);
    assert(fread(text, 1, (size_t)count, file) == (size_t)count);
    text[count] = '\0'; assert(fclose(file) == 0); return text;
}

static void write_text(const char *filename, const char *text)
{
    FILE *file = fopen(filename, "wb"); assert(file);
    size_t length = strlen(text);
    assert(fwrite(text, 1, length, file) == length);
    assert(fclose(file) == 0);
}

static void remove_tree(const char *root)
{
    struct stat info; if (lstat(root, &info)) return;
    if (!S_ISDIR(info.st_mode)) { assert(unlink(root) == 0); return; }
    DIR *dir = opendir(root); assert(dir); struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char child[4096]; path(child, sizeof(child), root, entry->d_name); remove_tree(child);
    }
    closedir(dir); assert(rmdir(root) == 0);
}

int main(void)
{
    char temporary[] = "/tmp/opencode/nn-resources-XXXXXX";
    char *root = mkdtemp(temporary); assert(root);
    NNApplication *app = nn_app_new("stereotype-packages/core"); assert(app);
    char error[512] = "";
    assert(!nn_app_create(app, root, "..", "Escape", false, error, sizeof(error)));
    assert(nn_app_create(app, root, "resources", "Resources", false, error, sizeof(error)));
    char directory[4096]; path(directory, sizeof(directory), root, "resources");
    char model_path[4096]; path(model_path, sizeof(model_path), directory, "model.json");
    char *before = read_text(model_path);
    assert(strstr(before, "\n  \"nodes\"")); assert(before[strlen(before)-1] == '\n');
    assert(nn_app_add_node(app, "unsaved", "core.relu", "0.1.0", "", 17, 29, error, sizeof(error)));
    assert(nn_project_dirty(nn_app_project(app)));
    assert(!nn_app_create_stereotype(app, "local.bad", "0.1.0", definition, "while true do end", "{}", error, sizeof(error)));
    assert(!nn_app_create_stereotype(app, "local.bad", "0.1.0", definition, rule, "{\"absent\":\"0.1.0\"}", error, sizeof(error)));
    assert(!nn_app_create_stereotype(app, "core.linear", "0.1.0", definition, rule, "{}", error, sizeof(error)));
    char check[4096]; path(check, sizeof(check), directory, "packages/local.bad-0.1.0");
    assert(access(check, F_OK) != 0);
    char *after = read_text(model_path); assert(!strcmp(before, after)); free(after); free(before);
    assert(nn_project_dirty(nn_app_project(app)));
    assert(nn_model_find_node(nn_app_model(app), "unsaved"));

    assert(nn_app_create_stereotype(app, "local.affine", "0.1.0", definition, rule, "{\"core.linear\":\"0.1.0\"}", error, sizeof(error)));
    assert(!nn_project_dirty(nn_app_project(app)));
    assert(nn_app_add_node(app, "local", "local.affine", "0.1.0", "", 1, 2, error, sizeof(error)));
    const NNPackage *package = nn_catalog_find(nn_project_catalog(nn_app_project(app)), "local.affine", "0.1.0");
    assert(package && package->parameter_count == 4);
    assert(package->output_count == 1 && !strcmp(package->outputs[0].id, "out") &&
           !strcmp(package->outputs[0].type, "output"));
    assert(!strcmp(package->parameters[0].position, "top"));
    assert(!strcmp(package->parameters[1].position, "bottom"));
    char *value = nn_app_parameter_text(app, "local", "factor"); assert(value && !strcmp(value, "1")); nn_app_free_text(value);
    assert(!nn_app_create_stereotype(app, "local.affine", "0.1.0", definition, rule, "{}", error, sizeof(error)));
    assert(!nn_app_create_stereotype(app, "bad.choices", "0.1.0",
        "{\"name\":\"Bad\",\"kind\":\"layer\",\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},"
        "\"parameters\":{\"mode\":{\"type\":\"string\",\"choices\":[3],\"default\":\"a\"}}}", rule, "{}", error, sizeof(error)));
    assert(!nn_app_create_stereotype(app, "bad.outputs", "0.1.0",
        "{\"name\":\"Bad\",\"kind\":\"layer\",\"outputs\":["
        "{\"id\":\"a\",\"type\":\"loss\"},{\"id\":\"b\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#444444\",\"width\":200,\"height\":100},\"parameters\":{}}",
        rule, "{}", error, sizeof(error)));

    assert(!nn_app_create_dataset(app, "bad.dataset", "0.1.0",
        "{\"name\":\"Bad\",\"batch\":{\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[0]}},\"targets\":{}}}", true, error, sizeof(error)));
    assert(!nn_app_create_dataset(app, "duplicates", "0.1.0",
        "{\"name\":\"Bad\",\"batch\":{\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[1]},\"x\":{\"dtype\":\"float32\",\"shape\":[2]}},\"targets\":{}}}", false, error, sizeof(error)));
    assert(nn_app_create_dataset(app, "local.images", "0.1.0", dataset, true, error, sizeof(error)));
    assert(!nn_project_dirty(nn_app_project(app)));
    assert(nn_project_active_dataset(nn_app_project(app)));
    assert(nn_app_add_node(app, "input", "core.input", "0.1.0", "", 0, 0, error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "input", "binding", "missing", error, sizeof(error)));
    assert(nn_app_select_dataset(app, "local.images", "0.1.0", error, sizeof(error))); /* unresolved allowed */
    assert(!nn_app_select_dataset(app, "missing", "0.1.0", error, sizeof(error)));

    const char *loss_definition =
        "{\"name\":\"Local loss\",\"kind\":\"layer\",\"outputs\":["
        "{\"id\":\"objective\",\"type\":\"loss\"}],"
        "\"view\":{\"color\":\"#4779c4\",\"width\":200,\"height\":100},\"parameters\":{}}";
    assert(nn_app_create_stereotype(app, "local.loss-source", "0.1.0", loss_definition,
                                    rule, "{}", error, sizeof(error)));
    assert(nn_app_add_node(app, "local-loss", "local.loss-source", "0.1.0", "",
                           0, 0, error, sizeof(error)));
    assert(nn_app_connect(app, "local-loss-edge", "local-loss", "objective",
                          "loss-output", "in", error, sizeof(error)));
    assert(nn_project_dirty(nn_app_project(app)));
    char *candidate_before = read_text(model_path);
    assert(candidate_before);
    char source_definition_path[4096];
    path(source_definition_path, sizeof(source_definition_path), directory,
         "packages/local.loss-source-0.1.0/definition.json");
    write_text(source_definition_path,
        "{\"name\":\"Local loss\",\"kind\":\"layer\",\"outputs\":["
        "{\"id\":\"objective\",\"type\":\"output\"}],"
        "\"view\":{\"color\":\"#4779c4\",\"width\":200,\"height\":100},\"parameters\":{}}");
    assert(!nn_app_create_stereotype(app, "local.unrelated", "0.1.0", definition,
                                     rule, "{}", error, sizeof(error)));
    assert(strstr(error, "terminal") || strstr(error, "handle"));
    assert(nn_project_dirty(nn_app_project(app)));
    assert(!strcmp(nn_app_output_type(app, "local-loss", "objective"), "loss"));
    char *candidate_after = read_text(model_path);
    assert(candidate_after && !strcmp(candidate_before, candidate_after));
    free(candidate_before); free(candidate_after);
    path(check, sizeof(check), directory, "packages/local.unrelated-0.1.0");
    assert(access(check, F_OK) != 0);
    write_text(source_definition_path, loss_definition);

    /* A failed final model replacement must roll back files and dirty state. */
    char backup[4096]; path(backup, sizeof(backup), directory, "model.backup");
    assert(rename(model_path, backup) == 0 && mkdir(model_path, 0755) == 0);
    assert(!nn_app_create_dataset(app, "rollback", "0.1.0", dataset, true, error, sizeof(error)));
    assert(nn_project_dataset_count(nn_app_project(app)) == 1);
    assert(!strcmp(nn_project_active_dataset(nn_app_project(app))->id, "local.images"));
    assert(nn_project_dirty(nn_app_project(app)));
    path(check, sizeof(check), directory, "datasets/rollback-0.1.0"); assert(access(check, F_OK) != 0);
    assert(rmdir(model_path) == 0 && rename(backup, model_path) == 0);

    /* Symlink category directories are never followed, and never removed. */
    assert(nn_app_create(app, root, "linked", "Linked", false, error, sizeof(error)));
    char linked[4096]; path(linked, sizeof(linked), root, "linked/packages");
    assert(symlink(directory, linked) == 0);
    assert(!nn_app_create_stereotype(app, "escape", "0.1.0", definition, rule, "{}", error, sizeof(error)));
    struct stat info; assert(lstat(linked, &info) == 0 && S_ISLNK(info.st_mode));
    assert(nn_app_open(app, directory, error, sizeof(error)));
    assert(nn_model_find_node(nn_app_model(app), "local"));
    assert(nn_model_find_node(nn_app_model(app), "unsaved"));
    assert(nn_project_dataset_count(nn_app_project(app)) == 1);
    path(check, sizeof(check), directory, "packages/local.affine-0.1.0/definition.json");
    char *text = read_text(check); assert(strstr(text, "\n  \"parameters\"")); assert(text[strlen(text)-1] == '\n'); free(text);

    assert(nn_app_create_vae(app, root, "vae-copy", "VAE copy", error, sizeof(error)));
    assert(nn_model_node_count(nn_app_model(app)) == 23);
    assert(nn_app_node_is_subflow(app, "encoder") && nn_app_node_is_subflow(app, "decoder"));
    assert(nn_catalog_find(nn_project_catalog(nn_app_project(app)), "vae.reparameterize", "0.1.0"));
    NNInferenceReport *report = nn_infer_project(nn_app_project(app)); assert(report);
    size_t successes = 0;
    for (size_t i = 0; i < nn_inference_count(report); ++i)
        if (nn_inference_at(report, i)->status == NN_INFERENCE_SUCCESS) ++successes;
    assert(successes == 23); nn_inference_free(report);
    nn_app_free(app); remove_tree(root);
    puts("resource authoring and persistence: ok"); return 0;
}
