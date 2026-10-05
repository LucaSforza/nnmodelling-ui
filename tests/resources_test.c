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
    char scaffold_path[4096];
    path(scaffold_path, sizeof(scaffold_path), directory, "packages/local.affine-0.1.0/pyproject.toml");
    char *scaffold = read_text(scaffold_path);
    assert(strstr(scaffold, "nnmodelling-runtime>=0.1.0")); free(scaffold);
    path(scaffold_path, sizeof(scaffold_path), directory, "packages/local.affine-0.1.0/pytorch.py");
    scaffold = read_text(scaffold_path);
    assert(strstr(scaffold, "def build(parameters, context, services)"));
    assert(strstr(scaffold, "NotImplementedError")); free(scaffold);
    path(scaffold_path, sizeof(scaffold_path), directory, "packages/local.affine-0.1.0/manifest.json");
    scaffold = read_text(scaffold_path);
    assert(strstr(scaffold, "\"pytorch\"")); assert(strstr(scaffold, "pytorch.py")); free(scaffold);
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
    path(scaffold_path, sizeof(scaffold_path), directory, "datasets/local.images-0.1.0/pyproject.toml");
    scaffold = read_text(scaffold_path);
    assert(strstr(scaffold, "nnmodelling-runtime>=0.1.0")); free(scaffold);
    path(scaffold_path, sizeof(scaffold_path), directory, "datasets/local.images-0.1.0/dataset.py");
    scaffold = read_text(scaffold_path);
    assert(strstr(scaffold, "class Dataset(")); assert(strstr(scaffold, "def tokenize("));
    assert(strstr(scaffold, "def untokenize(")); assert(strstr(scaffold, "def load("));
    assert(strstr(scaffold, "NotImplementedError")); free(scaffold);
    path(scaffold_path, sizeof(scaffold_path), directory, "datasets/local.images-0.1.0/manifest.json");
    scaffold = read_text(scaffold_path);
    assert(strstr(scaffold, "\"python\"")); assert(strstr(scaffold, "dataset.py")); free(scaffold);
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
    assert(nn_app_create_llm(app, root, "mini-llm", "mini LLM", error, sizeof(error)));
    assert(!strcmp(nn_project_id(nn_app_project(app)), "mini-llm"));
    assert(!strcmp(nn_project_name(nn_app_project(app)), "mini LLM"));
    assert(nn_project_active_dataset(nn_app_project(app)));
    assert(nn_catalog_find(nn_project_catalog(nn_app_project(app)),
                           "llm.causal-mask", "1.0.0"));
    size_t llm_nodes = nn_model_node_count(nn_app_model(app));
    assert(llm_nodes > 23);
    report = nn_infer_project(nn_app_project(app)); assert(report);
    assert(nn_inference_count(report) == llm_nodes);
    for (size_t i = 0; i < nn_inference_count(report); ++i)
        assert(nn_inference_at(report, i)->status == NN_INFERENCE_SUCCESS);
    nn_inference_free(report);
    const NNProject *llm_project = nn_app_project(app);
    assert(!nn_app_create_llm(app, root, "mini-llm", "Duplicate", error, sizeof(error)));
    assert(nn_app_project(app) == llm_project);
    assert(!nn_app_create_llm(app, root, "../escape", "Invalid", error, sizeof(error)));
    assert(nn_app_project(app) == llm_project);
    path(directory, sizeof(directory), root, "mini-llm");
    assert(nn_app_close(app, false, error, sizeof(error)));
    assert(nn_app_open(app, directory, error, sizeof(error)));
    assert(nn_model_node_count(nn_app_model(app)) == llm_nodes);
    assert(!strcmp(nn_project_name(nn_app_project(app)), "mini LLM"));

    /* Dataset edits patch visible metadata while retaining opaque resource data. */
    assert(nn_app_create(app, root, "dataset-edit", "Dataset edit", false,
                         error, sizeof(error)));
    path(directory, sizeof(directory), root, "dataset-edit");
    path(model_path, sizeof(model_path), directory, "model.json");
    assert(nn_app_create_dataset(app, "editable.data", "1.2.0",
        "{\"name\":\"Before\",\"description\":\"old\",\"custom\":{\"keep\":true},"
        "\"batch\":{\"customBatch\":17,\"inputs\":{\"image\":{\"dtype\":\"float32\","
        "\"shape\":[\"B\",8],\"normalization\":\"unit\"}},\"targets\":{}}}",
        true, error, sizeof(error)));
    char dataset_dir[4096], manifest_path[4096], definition_path[4096];
    path(dataset_dir, sizeof(dataset_dir), directory, "datasets/editable.data-1.2.0");
    path(manifest_path, sizeof(manifest_path), dataset_dir, "manifest.json");
    path(definition_path, sizeof(definition_path), dataset_dir, "metadata.json");
    write_text(manifest_path,
        "{\"schemaVersion\":1,\"id\":\"editable.data\",\"version\":\"1.2.0\","
        "\"entrypoints\":{\"definition\":\"metadata.json\",\"python\":\"dataset.py\"}}");
    write_text(definition_path,
        "{\"name\":\"Before\",\"description\":\"old\",\"custom\":{\"keep\":true},"
        "\"batch\":{\"customBatch\":17,\"inputs\":{\"image\":{\"dtype\":\"float32\","
        "\"shape\":[\"B\",8],\"normalization\":\"unit\"}},\"targets\":{}}}");
    char asset_path[4096];
    path(asset_path, sizeof(asset_path), dataset_dir, "dataset.py");
    write_text(asset_path, "# adapter stays byte-for-byte\n");
    path(asset_path, sizeof(asset_path), dataset_dir, "sample.bin");
    write_text(asset_path, "sample bytes\n");
    char *unchanged;
    char *manifest_before_get = read_text(manifest_path);
    char *definition_copy = nn_app_dataset_definition(app, "editable.data", "1.2.0",
                                                       error, sizeof(error));
    assert(definition_copy && strstr(definition_copy, "normalization"));
    nn_app_free_text(definition_copy);
    unchanged = read_text(manifest_path);
    assert(!strcmp(unchanged, manifest_before_get));
    free(unchanged); free(manifest_before_get);
    assert(nn_app_add_node(app, "pending", "core.relu", "0.1.0", "", 11, 21,
                           error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));
    const NNInferenceReport *old_report = nn_app_analysis(app, error, sizeof(error));
    assert(old_report);
    char *model_before_invalid = read_text(model_path);
    char *definition_before_invalid = read_text(definition_path);
    const char *invalid_edits[] = {
        "{\"name\":\"Invalid\",\"batch\":{\"inputs\":{},\"targets\":{}}}",
        "{\"name\":\"Wrong identity\",\"batch\":{\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[1]}},\"targets\":{}}}"
    };
    assert(!nn_app_update_dataset(app, "editable.data", "1.2.0", invalid_edits[0], error, sizeof(error)));
    assert(!nn_app_update_dataset(app, "absent", "1.2.0", invalid_edits[1], error, sizeof(error)));
    assert(nn_app_can_undo(app) && nn_project_dirty(nn_app_project(app)));
    assert(nn_app_analysis(app, error, sizeof(error)) == old_report);
    unchanged = read_text(model_path); assert(!strcmp(unchanged, model_before_invalid)); free(unchanged);
    unchanged = read_text(definition_path); assert(!strcmp(unchanged, definition_before_invalid)); free(unchanged);
    free(model_before_invalid); free(definition_before_invalid);

    assert(nn_app_update_dataset(app, "editable.data", "1.2.0",
        "{\"name\":\"After\",\"description\":\"new\",\"batch\":{"
        "\"inputs\":{\"image\":{\"dtype\":\"float64\",\"shape\":[\"B\",16]}},"
        "\"targets\":{\"class\":{\"dtype\":\"int64\",\"shape\":[\"B\"]}}}}",
        error, sizeof(error)));
    assert(!nn_project_dirty(nn_app_project(app)) && !nn_app_can_undo(app));
    const NNDataset *active = nn_project_active_dataset(nn_app_project(app));
    assert(active && !strcmp(active->id, "editable.data") && !strcmp(active->version, "1.2.0"));
    char *updated = read_text(definition_path);
    assert(strstr(updated, "\"custom\"") && strstr(updated, "\"keep\": true"));
    assert(strstr(updated, "\"customBatch\": 17") && strstr(updated, "\"normalization\": \"unit\""));
    assert(strstr(updated, "\"class\"")); free(updated);
    path(asset_path, sizeof(asset_path), dataset_dir, "dataset.py");
    unchanged = read_text(asset_path); assert(!strcmp(unchanged, "# adapter stays byte-for-byte\n")); free(unchanged);
    path(asset_path, sizeof(asset_path), dataset_dir, "sample.bin");
    unchanged = read_text(asset_path); assert(!strcmp(unchanged, "sample bytes\n")); free(unchanged);
    assert(nn_model_find_node(nn_app_model(app), "pending"));
    assert(nn_app_analysis(app, error, sizeof(error)));
    assert(nn_app_close(app, false, error, sizeof(error)));
    assert(nn_app_open(app, directory, error, sizeof(error)));
    assert(!strcmp(nn_project_active_dataset(nn_app_project(app))->id, "editable.data"));
    assert(nn_model_find_node(nn_app_model(app), "pending"));
    definition_copy = nn_app_dataset_definition(app, "editable.data", "1.2.0", error, sizeof(error));
    assert(definition_copy && strstr(definition_copy, "After") && strstr(definition_copy, "customBatch"));
    nn_app_free_text(definition_copy);

    /* The declared definition entrypoint is confined and must remain a regular file. */
    char symlink_target[4096], definition_backup[4096];
    path(symlink_target, sizeof(symlink_target), directory, "external-metadata.json");
    path(definition_backup, sizeof(definition_backup), directory, "metadata.backup");
    write_text(symlink_target, "{\"name\":\"outside\",\"batch\":{\"inputs\":{},\"targets\":{}}}");
    char *manifest_before_reject = read_text(manifest_path);
    assert(rename(definition_path, definition_backup) == 0);
    assert(symlink(symlink_target, definition_path) == 0);
    assert(!nn_app_dataset_definition(app, "editable.data", "1.2.0", error, sizeof(error)));
    unchanged = read_text(manifest_path);
    assert(!strcmp(unchanged, manifest_before_reject));
    free(unchanged); free(manifest_before_reject);
    assert(unlink(definition_path) == 0 && rename(definition_backup, definition_path) == 0);

    /* A failed final model save restores the old metadata and app history. */
    assert(nn_app_add_node(app, "rollback-pending", "core.relu", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    char *old_definition = read_text(definition_path);
    char *saved_model = read_text(model_path);
    char model_backup[4096]; path(model_backup, sizeof(model_backup), directory, "model.backup");
    assert(rename(model_path, model_backup) == 0 && mkdir(model_path, 0755) == 0);
    assert(!nn_app_update_dataset(app, "editable.data", "1.2.0",
        "{\"name\":\"Must roll back\",\"description\":\"rollback\",\"batch\":{\"inputs\":{\"image\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},\"targets\":{}}}",
        error, sizeof(error)));
    assert(nn_project_dirty(nn_app_project(app)) && nn_app_can_undo(app));
    unchanged = read_text(definition_path); assert(!strcmp(unchanged, old_definition)); free(unchanged);
    assert(nn_model_find_node(nn_app_model(app), "rollback-pending"));
    assert(rmdir(model_path) == 0 && rename(model_backup, model_path) == 0);
    unchanged = read_text(model_path); assert(!strcmp(unchanged, saved_model)); free(unchanged);
    free(old_definition); free(saved_model);

    nn_app_free(app); remove_tree(root);
    puts("resource authoring and persistence: ok"); return 0;
}
