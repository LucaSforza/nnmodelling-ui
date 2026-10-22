#define _XOPEN_SOURCE 700
#include "application/application.h"
#include "yyjson.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void write_text(const char *path, const char *text)
{
    FILE *file = fopen(path, "wb");
    assert(file);
    assert(fwrite(text, 1, strlen(text), file) == strlen(text));
    assert(fclose(file) == 0);
}

static char *read_text(const char *path)
{
    FILE *file = fopen(path, "rb");
    assert(file && fseek(file, 0, SEEK_END) == 0);
    long length = ftell(file);
    assert(length >= 0 && fseek(file, 0, SEEK_SET) == 0);
    char *text = malloc((size_t)length + 1);
    assert(text && fread(text, 1, (size_t)length, file) == (size_t)length);
    text[length] = '\0';
    assert(fclose(file) == 0);
    return text;
}

static void install_parameter_rule(const char *project, const char *id,
                                   const char *version, const char *source)
{
    char directory[4096], manifest[4096], rule[4096];
    assert(snprintf(directory, sizeof(directory), "%s/packages/%s-%s",
                    project, id, version) < (int)sizeof(directory));
    assert(snprintf(manifest, sizeof(manifest), "%s/manifest.json", directory) <
           (int)sizeof(manifest));
    assert(snprintf(rule, sizeof(rule), "%s/parameters.lua", directory) <
           (int)sizeof(rule));
    char *text = read_text(manifest);
    yyjson_doc *doc = yyjson_read(text, strlen(text), 0);
    yyjson_mut_doc *mutable = doc ? yyjson_doc_mut_copy(doc, NULL) : NULL;
    assert(mutable);
    yyjson_mut_val *entrypoints = yyjson_mut_obj_get(yyjson_mut_doc_get_root(mutable), "entrypoints");
    yyjson_mut_val *entry = yyjson_mut_obj(mutable);
    assert(entrypoints && entry);
    assert(yyjson_mut_obj_add_str(mutable, entry, "language", "lua"));
    assert(yyjson_mut_obj_add_str(mutable, entry, "file", "parameters.lua"));
    assert(yyjson_mut_obj_add_val(mutable, entrypoints, "parameters", entry));
    char *updated = yyjson_mut_write(mutable, YYJSON_WRITE_PRETTY, NULL);
    assert(updated);
    write_text(manifest, updated);
    write_text(rule, source);
    free(updated);
    yyjson_mut_doc_free(mutable);
    yyjson_doc_free(doc);
    free(text);
}

static const yyjson_val *row_for(const yyjson_val *root, const char *node)
{
    const yyjson_val *rows = yyjson_obj_get(root, "rows");
    yyjson_arr_iter iter = yyjson_arr_iter_with(rows);
    yyjson_val *item;
    while ((item = yyjson_arr_iter_next(&iter))) {
        const char *id = yyjson_get_str(yyjson_obj_get(item, "node_id"));
        if (id && !strcmp(id, node)) return item;
    }
    return NULL;
}

int main(void)
{
    char temporary[] = "/tmp/nn-parameter-count-XXXXXX";
    char *root = mkdtemp(temporary);
    char cwd[4096];
    assert(root && getcwd(cwd, sizeof(cwd)));
    char core[4096];
    assert(snprintf(core, sizeof(core), "%s/stereotype-packages/core", cwd) < (int)sizeof(core));
    NNApplication *app = nn_app_new(core);
    char error[512] = "";
    assert(app && nn_app_create(app, root, "count-demo", "Count demo", "0.1.0", false,
                                error, sizeof(error)));
    assert(nn_app_add_node(app, "dense", "core.linear", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "dense", "in_features", "3", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "dense", "out_features", "4", error, sizeof(error)));
    bool dirty = nn_project_dirty(nn_app_project(app));
    char *text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text && error[0] == '\0');
    yyjson_doc *doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    const yyjson_val *result = yyjson_doc_get_root(doc);
    assert(yyjson_get_int(yyjson_obj_get(result, "total")) == 16);
    assert(yyjson_get_int(yyjson_obj_get(result, "trainable")) == 16);
    assert(yyjson_get_bool(yyjson_obj_get(result, "complete")));
    const yyjson_val *row = row_for(result, "dense");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "success"));
    assert(yyjson_get_int(yyjson_obj_get(row, "total")) == 16);
    assert(nn_project_dirty(nn_app_project(app)) == dirty);
    yyjson_doc_free(doc);
    nn_app_free_text(text);

    assert(nn_app_add_node(app, "repeat", "core.repeat", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "repeat", "times", "2", error, sizeof(error)));
    assert(nn_app_add_node(app, "inner", "core.linear", "0.1.0", "repeat", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "inner", "in_features", "2", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "inner", "out_features", "3", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "inner", "bias", "false", error, sizeof(error)));
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text && error[0] == '\0');
    doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    result = yyjson_doc_get_root(doc);
    assert(yyjson_get_int(yyjson_obj_get(result, "total")) == 28);
    assert(yyjson_get_int(yyjson_obj_get(result, "trainable")) == 28);
    row = row_for(result, "repeat");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 12);
    row = row_for(result, "inner");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 6);
    yyjson_doc_free(doc);
    nn_app_free_text(text);

    assert(nn_app_add_node(app, "horizontal", "core.horizontal-repeat", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "horizontal", "times", "2", error, sizeof(error)));
    assert(nn_app_add_node(app, "horizontal-inner", "core.linear", "0.1.0", "horizontal", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "horizontal-inner", "in_features", "2", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "horizontal-inner", "out_features", "2", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "horizontal-inner", "bias", "false", error, sizeof(error)));
    assert(nn_app_add_node(app, "embedding", "core.embedding", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "embedding", "num_embeddings", "10", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "embedding", "embedding_dim", "3", error, sizeof(error)));
    assert(nn_app_add_node(app, "conv", "core.conv2d", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "conv", "in_channels", "2", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "conv", "out_channels", "4", error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "conv", "kernel_size", "3", error, sizeof(error)));
    assert(nn_app_add_node(app, "batch-norm", "core.batch-norm2d", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "batch-norm", "num_features", "4", error, sizeof(error)));
    assert(nn_app_add_node(app, "layer-norm", "core.layer-norm", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "layer-norm", "normalized_shape", "5", error, sizeof(error)));
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    result = yyjson_doc_get_root(doc);
    assert(yyjson_get_int(yyjson_obj_get(result, "total")) == 160);
    row = row_for(result, "horizontal");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 8);
    row = row_for(result, "embedding");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 30);
    row = row_for(result, "conv");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 76);
    row = row_for(result, "batch-norm");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 8);
    row = row_for(result, "layer-norm");
    assert(row && yyjson_get_int(yyjson_obj_get(row, "total")) == 10);
    yyjson_doc_free(doc);
    nn_app_free_text(text);

    const char *definition =
        "{\"name\":\"Unknown\",\"kind\":\"layer\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    const char *inference = "return function() return {status='unresolved'} end";
    assert(nn_app_create_stereotype(app, "local.unknown", "1.0.0", definition,
                                    inference, "{}", error, sizeof(error)));
    assert(nn_app_add_node(app, "unknown", "local.unknown", "1.0.0", "", 0, 0,
                           error, sizeof(error)));
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    result = yyjson_doc_get_root(doc);
    assert(!yyjson_get_bool(yyjson_obj_get(result, "complete")));
    assert(yyjson_get_int(yyjson_obj_get(result, "total")) == 160);
    row = row_for(result, "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "unknown"));
    yyjson_doc_free(doc); nn_app_free_text(text);
    char project[4096], manifest[4096], rule[4096];
    assert(snprintf(project, sizeof(project), "%s/count-demo", root) < (int)sizeof(project));
    assert(snprintf(manifest, sizeof(manifest), "%s/packages/local.unknown-1.0.0/manifest.json",
                    project) < (int)sizeof(manifest));
    assert(snprintf(rule, sizeof(rule), "%s/packages/local.unknown-1.0.0/parameters.lua",
                    project) < (int)sizeof(rule));
    char *manifest_text = read_text(manifest);
    yyjson_doc *manifest_doc = yyjson_read(manifest_text, strlen(manifest_text), 0);
    yyjson_mut_doc *mutable_doc = manifest_doc ? yyjson_doc_mut_copy(manifest_doc, NULL) : NULL;
    assert(mutable_doc);
    yyjson_mut_val *entrypoints = yyjson_mut_obj_get(yyjson_mut_doc_get_root(mutable_doc), "entrypoints");
    yyjson_mut_val *count_entry = yyjson_mut_obj(mutable_doc);
    assert(entrypoints && count_entry);
    assert(yyjson_mut_obj_add_str(mutable_doc, count_entry, "language", "lua"));
    assert(yyjson_mut_obj_add_str(mutable_doc, count_entry, "file", "parameters.lua"));
    assert(yyjson_mut_obj_add_val(mutable_doc, entrypoints, "parameters", count_entry));
    char *updated_manifest = yyjson_mut_write(mutable_doc, YYJSON_WRITE_PRETTY, NULL);
    assert(updated_manifest);
    write_text(manifest, updated_manifest);
    free(updated_manifest);
    yyjson_mut_doc_free(mutable_doc);
    yyjson_doc_free(manifest_doc);
    free(manifest_text);
    write_text(rule, "return function() return {status='success', total=0, trainable=0} end");
    assert(nn_app_close(app, false, error, sizeof(error)));
    if (!nn_app_open(app, project, error, sizeof(error))) fprintf(stderr, "reopen failed: %s\n", error);
    assert(nn_app_project(app));
    const NNPackage *count_package = nn_catalog_find(
        nn_project_catalog(nn_app_project(app)), "local.unknown", "1.0.0");
    assert(count_package && !strcmp(count_package->parameters_file, "parameters.lua"));
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    result = yyjson_doc_get_root(doc);
    assert(yyjson_get_bool(yyjson_obj_get(result, "complete")));
    row = row_for(result, "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "success"));
    yyjson_doc_free(doc);
    nn_app_free_text(text);

    write_text(rule, "return function() return {status='unresolved', message='shape required'} end");
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    result = yyjson_doc_get_root(doc);
    assert(!yyjson_get_bool(yyjson_obj_get(result, "complete")));
    assert(yyjson_get_int(yyjson_obj_get(result, "total")) == 160);
    row = row_for(result, "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "unresolved"));
    yyjson_doc_free(doc); nn_app_free_text(text);

    write_text(rule, "return function(");
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    row = row_for(yyjson_doc_get_root(doc), "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "error"));
    yyjson_doc_free(doc); nn_app_free_text(text);
    write_text(rule, "return function() return {status='success', total=-1, trainable=0} end");
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    row = row_for(yyjson_doc_get_root(doc), "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "error"));
    yyjson_doc_free(doc); nn_app_free_text(text);
    write_text(rule, "return function() return {status='success', total=1, trainable=2} end");
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    row = row_for(yyjson_doc_get_root(doc), "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "error"));
    yyjson_doc_free(doc); nn_app_free_text(text);
    write_text(rule, "return function() return {status='success', total=math.maxinteger + 1, trainable=0} end");
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text);
    doc = yyjson_read(text, strlen(text), 0);
    row = row_for(yyjson_doc_get_root(doc), "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "error"));
    yyjson_doc_free(doc); nn_app_free_text(text);
    write_text(rule, "local values = {}; while true do values[#values + 1] = string.rep('x', 4096) end");
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text && error[0] == '\0');
    doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    result = yyjson_doc_get_root(doc);
    assert(!yyjson_get_bool(yyjson_obj_get(result, "complete")));
    row = row_for(result, "unknown");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "error"));
    yyjson_doc_free(doc); nn_app_free_text(text);

    write_text(rule, "return function() return {status='success', total=0, trainable=0} end");
    const char *source_definition =
        "{\"name\":\"Count source\",\"kind\":\"input\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    const char *layer_definition =
        "{\"name\":\"Count caller\",\"kind\":\"layer\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    const char *join_definition =
        "{\"name\":\"Count reference join\",\"kind\":\"join\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    assert(nn_app_create_stereotype(app, "local.count-source", "1.0.0",
        source_definition,
        "return function() return {status='success', output=tensor.create({'B',4},'float32')} end",
        "{}", error, sizeof(error)));
    assert(nn_app_create_stereotype(app, "local.count-caller", "1.0.0",
        layer_definition,
        "return function(context) return {status='success', output=context.inputs[1]} end",
        "{}", error, sizeof(error)));
    assert(nn_app_create_stereotype(app, "local.count-join", "1.0.0",
        join_definition,
        "return function() return {status='unresolved'} end",
        "{}", error, sizeof(error)));
    install_parameter_rule(project, "local.count-join", "1.0.0",
        "return function(context) local input=context.inputs[1]; "
        "if not input then return {status='error', message='caller input missing'} end; "
        "return {status='success', total=input.shape[2], trainable=input.shape[2]} end");
    install_parameter_rule(project, "local.count-caller", "1.0.0",
        "return function(context, parameters, services) "
        "local nested=services.count_stereotype({id='local.count-join',version='1.0.0',parameters={}}); "
        "return {status='success',total=nested.total,trainable=nested.trainable} end");
    assert(nn_app_add_node(app, "count-source", "local.count-source", "1.0.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_add_node(app, "count-caller", "local.count-caller", "1.0.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_connect(app, "count-input", "count-source", "out", "count-caller", "in",
                          error, sizeof(error)));
    assert(nn_app_save(app, error, sizeof(error)));
    assert(nn_app_close(app, false, error, sizeof(error)));
    assert(nn_app_open(app, project, error, sizeof(error)));
    text = nn_app_parameter_counts_json(app, error, sizeof(error));
    assert(text && error[0] == '\0');
    doc = yyjson_read(text, strlen(text), 0);
    assert(doc);
    row = row_for(yyjson_doc_get_root(doc), "count-caller");
    assert(row && !strcmp(yyjson_get_str(yyjson_obj_get(row, "status")), "success"));
    assert(yyjson_get_int(yyjson_obj_get(row, "total")) == 4);
    yyjson_doc_free(doc); nn_app_free_text(text);
    nn_app_free(app);
    return 0;
}
