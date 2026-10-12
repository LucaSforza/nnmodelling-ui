#define _XOPEN_SOURCE 700
#include "application/application.h"

#include <assert.h>
#include <stdio.h>
#include <ftw.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const NNInferenceResult *result_for(const NNInferenceReport *report,
                                           const char *id)
{
    for (size_t i = 0; i < nn_inference_count(report); ++i) {
        const NNInferenceResult *result = nn_inference_at(report, i);
        if (!strcmp(result->node_id, id)) return result;
    }
    return NULL;
}

static const NNNode *scoped_package(const NNModel *model, const char *scope,
                                    const char *package)
{
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (!strcmp(node->scope_id ? node->scope_id : "", scope) &&
            !strcmp(node->package_id, package)) return node;
    }
    return NULL;
}

static int remove_entry(const char *path, const struct stat *status, int type,
                        struct FTW *walk)
{
    (void)status; (void)type; (void)walk;
    return remove(path);
}

static void assert_shape(const NNInferenceResult *result, const char *first,
                         size_t dimensions)
{
    assert(result && result->status == NN_INFERENCE_SUCCESS);
    assert(!strcmp(result->dtype, "float32"));
    assert(result->dimension_count == dimensions);
    assert(!strcmp(result->dimensions[0], "B"));
    assert(!strcmp(result->dimensions[1], first));
}

static void assert_instruction_budget(const char *core)
{
    char temporary[] = "/tmp/nn-inference-instructions-XXXXXX";
    char *parent = mkdtemp(temporary);
    assert(parent);
    char error[512] = "";
    NNApplication *app = nn_app_new(core);
    assert(app && nn_app_create(app, parent, "instructions", "Instruction test",
                                false, error, sizeof(error)));
    assert(nn_app_create_dataset(app, "local.instruction-data", "1.0.0",
        "{\"name\":\"Instruction data\",\"batch\":{\"inputs\":{\"input\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},\"targets\":{}}}",
        true, error, sizeof(error)));
    assert(nn_app_add_node(app, "instruction-input", "core.input", "0.1.0", "",
                           0, 0, error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "instruction-input", "binding", "input",
                                     error, sizeof(error)));
    const char *layer_definition =
        "{\"name\":\"Fast pass-through\",\"kind\":\"layer\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    assert(nn_app_create_stereotype(app, "local.instruction-source", "1.0.0",
        layer_definition,
        "return function(context) return {status='success', output=context.inputs[1]} end",
        "{}", error, sizeof(error)));
    const char *caller_definition =
        "{\"name\":\"Instruction caller\",\"kind\":\"layer\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    const char *caller_rule =
        "return function(context, parameters, services)\n"
        "  local reference = {id='local.instruction-source', version='1.0.0', parameters={}}\n"
        "  local inputs = {context.inputs[1]}\n"
        "  local result\n"
        "  for i = 1, 11000 do\n"
        "    result = services.infer_stereotype(reference, inputs)\n"
        "    if result.status ~= 'success' then return result end\n"
        "  end\n"
        "  return result\n"
        "end";
    assert(nn_app_create_stereotype(app, "local.instruction-caller", "1.0.0",
        caller_definition, caller_rule,
        "{\"local.instruction-source\":\"1.0.0\"}", error, sizeof(error)));
    assert(nn_app_add_node(app, "instruction-owner", "local.instruction-caller",
                           "1.0.0", "", 0, 0, error, sizeof(error)));
    assert(nn_app_connect(app, "instruction-edge", "instruction-input", "out",
                          "instruction-owner", "in", error, sizeof(error)));
    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    const NNInferenceResult *result = result_for(report, "instruction-owner");
    assert(result && result->status == NN_INFERENCE_RUNTIME_FAULT);
    assert(strstr(result->message, "cumulative Lua instruction budget limit exceeded"));
    nn_app_free(app);
    assert(nftw(parent, remove_entry, 32, FTW_DEPTH | FTW_PHYS) == 0);
}

static void assert_multi_output_replay(NNApplication *app, char *error,
                                       size_t error_capacity)
{
    const char *outputs =
        "\"outputs\":[{\"id\":\"objective\",\"type\":\"loss\"},"
        "{\"id\":\"prediction\",\"type\":\"output\"}],";
    char definition[512];
    assert(snprintf(definition, sizeof(definition),
        "{\"name\":\"Two outputs\",\"kind\":\"layer\",%s"
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}", outputs) < (int)sizeof(definition));
    assert(nn_app_create_stereotype(app, "local.cache-two", "1.0.0", definition,
        "return function() return {status='success', outputs={"
        "objective=tensor.create({},'float32'),"
        "prediction=tensor.create({'B',2},'float32')}} end",
        "{}", error, error_capacity));
    assert(snprintf(definition, sizeof(definition),
        "{\"name\":\"Two output scope\",\"kind\":\"subflow\",%s"
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}", outputs) < (int)sizeof(definition));
    assert(nn_app_create_stereotype(app, "local.cache-two-scope", "1.0.0", definition,
        "return function(context, parameters, services) "
        "local first=services.infer_subflow(context.inputs[1]); "
        "if first.status~='success' then return first end; "
        "return services.infer_subflow(context.inputs[1]) end",
        "{}", error, error_capacity));
    assert(nn_app_add_node(app, "multi-owner", "local.cache-two-scope", "1.0.0",
                           "", 0, 0, error, error_capacity));
    assert(nn_app_connect(app, "multi-parent-edge", "input", "out", "multi-owner", "in",
                          error, error_capacity));
    const NNModel *model = nn_app_model(app);
    const NNNode *inner_input = scoped_package(model, "multi-owner", "core.input");
    const NNNode *objective = NULL, *prediction = NULL;
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        if (strcmp(node->scope_id ? node->scope_id : "", "multi-owner")) continue;
        if (node->boundary_handle_id && !strcmp(node->boundary_handle_id, "objective"))
            objective = node;
        if (node->boundary_handle_id && !strcmp(node->boundary_handle_id, "prediction"))
            prediction = node;
    }
    assert(inner_input && objective && prediction);
    char inner_input_id[256], objective_id[256], prediction_id[256];
    assert(snprintf(inner_input_id, sizeof(inner_input_id), "%s", inner_input->id) <
           (int)sizeof(inner_input_id));
    assert(snprintf(objective_id, sizeof(objective_id), "%s", objective->id) <
           (int)sizeof(objective_id));
    assert(snprintf(prediction_id, sizeof(prediction_id), "%s", prediction->id) <
           (int)sizeof(prediction_id));
    assert(nn_app_add_node(app, "multi-source", "local.cache-two", "1.0.0",
                           "multi-owner", 0, 0, error, error_capacity));
    assert(nn_app_connect(app, "multi-source-edge", inner_input_id, "out",
                          "multi-source", "in", error, error_capacity));
    assert(nn_app_connect(app, "multi-loss-edge", "multi-source", "objective",
                          objective_id, "in", error, error_capacity));
    assert(nn_app_connect(app, "multi-output-edge", "multi-source", "prediction",
                          prediction_id, "in", error, error_capacity));
    const NNInferenceReport *report = nn_app_analysis(app, error, error_capacity);
    const NNInferenceResult *owner = result_for(report, "multi-owner");
    const NNInferenceResult *loss = result_for(report, objective_id);
    const NNInferenceResult *output = result_for(report, prediction_id);
    assert(owner && owner->status == NN_INFERENCE_SUCCESS && owner->output_count == 2);
    assert(!strcmp(owner->outputs[0].handle_id, "objective") &&
           !strcmp(owner->outputs[0].type, "loss") && owner->outputs[0].dimension_count == 0);
    assert(!strcmp(owner->outputs[1].handle_id, "prediction") &&
           !strcmp(owner->outputs[1].type, "output") && owner->outputs[1].dimension_count == 2 &&
           !strcmp(owner->outputs[1].dimensions[1], "2"));
    assert(loss && loss->status == NN_INFERENCE_SUCCESS && loss->dimension_count == 0 &&
           loss->dimensions);
    assert(output && output->status == NN_INFERENCE_SUCCESS && output->dimension_count == 2 &&
           !strcmp(output->dimensions[1], "2"));
}

int main(void)
{
    char temporary[] = "/tmp/nn-inference-cache-XXXXXX";
    char *parent = mkdtemp(temporary);
    assert(parent);
    char cwd[4096], core[4096], error[512] = "";
    assert(getcwd(cwd, sizeof(cwd)));
    assert(snprintf(core, sizeof(core), "%s/stereotype-packages/core", cwd) <
           (int)sizeof(core));
    NNApplication *app = nn_app_new(core);
    assert(app && nn_app_create(app, parent, "cache", "Cache test", false,
                                error, sizeof(error)));
    assert(nn_app_create_dataset(app, "local.cache-data", "1.0.0",
        "{\"name\":\"Cache data\",\"batch\":{\"inputs\":{\"input\":{\"dtype\":\"float32\",\"shape\":[\"B\",8]}},\"targets\":{}}}",
        true, error, sizeof(error)));
    assert(nn_app_add_node(app, "input", "core.input", "0.1.0", "", 0, 0,
                           error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "input", "binding", "input",
                                     error, sizeof(error)));

    const char *subflow_definition =
        "{\"name\":\"Cache probe\",\"kind\":\"subflow\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    const char *subflow_rule =
        "return function(context, parameters, services)\n"
        "  local a = context.inputs[1]\n"
        "  local b, message = tensor.append_dimension(a, 2)\n"
        "  if not b then return {status='error', message=message} end\n"
        "  local first = services.infer_subflow(a)\n"
        "  local second = services.infer_subflow(b)\n"
        "  if second.status ~= 'success' or second.output.shape[3] ~= 2 then\n"
        "    return {status='error', message='shape cache key mismatch'}\n"
        "  end\n"
        "  local third = services.infer_subflow(a)\n"
        "  return third\n"
        "end";
    assert(nn_app_create_stereotype(app, "local.cache-probe", "1.0.0",
        subflow_definition, subflow_rule, "{}", error, sizeof(error)));
    assert(nn_app_add_node(app, "cache-owner", "local.cache-probe", "1.0.0",
                           "", 0, 0, error, sizeof(error)));
    const NNNode *cache_input = scoped_package(nn_app_model(app), "cache-owner", "core.input");
    const NNNode *cache_output = scoped_package(nn_app_model(app), "cache-owner", "core.output");
    assert(cache_input && cache_output);
    char cache_input_id[256], cache_output_id[256];
    assert(snprintf(cache_input_id, sizeof(cache_input_id), "%s", cache_input->id) <
           (int)sizeof(cache_input_id));
    assert(snprintf(cache_output_id, sizeof(cache_output_id), "%s", cache_output->id) <
           (int)sizeof(cache_output_id));
    assert(nn_app_connect(app, "cache-child-edge", cache_input_id, "out",
                          cache_output_id, "in", error, sizeof(error)));
    assert(nn_app_connect(app, "cache-parent-edge", "input", "out",
                          "cache-owner", "in", error, sizeof(error)));
    assert(nn_app_set_boundary_handle(app, cache_output_id, "out", error, sizeof(error)));

    const NNInferenceReport *report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
    assert_shape(result_for(report, "cache-owner"), "8", 2);
    assert_shape(result_for(report, cache_input_id), "8", 2);
    assert_shape(result_for(report, cache_output_id), "8", 2);
    const NNInferenceResult *owner = result_for(report, "cache-owner");
    assert(owner->output_count == 1);
    assert(!strcmp(owner->outputs[0].handle_id, "out"));
    assert(!strcmp(owner->outputs[0].type, "output"));

    assert(nn_app_create_dataset(app, "local.cache-data-wide", "1.0.0",
        "{\"name\":\"Wide cache data\",\"batch\":{\"inputs\":{\"input\":{\"dtype\":\"float32\",\"shape\":[\"B\",13]}},\"targets\":{}}}",
        false, error, sizeof(error)));
    assert(nn_app_select_dataset(app, "local.cache-data-wide", "1.0.0",
                                 error, sizeof(error)));
    report = nn_app_analysis(app, error, sizeof(error));
    assert(report);
    assert_shape(result_for(report, "cache-owner"), "13", 2);
    assert_shape(result_for(report, cache_input_id), "13", 2);
    assert_shape(result_for(report, cache_output_id), "13", 2);

    const char *random_definition =
        "{\"name\":\"Random probe\",\"kind\":\"layer\","
        "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},"
        "\"parameters\":{}}";
    assert(nn_app_create_stereotype(app, "local.random-probe", "1.0.0",
        random_definition,
        "return function() return {status='success', output={dtype='float32', "
        "shape={math.random(1,8)}}} end", "{}", error, sizeof(error)));
    assert(nn_app_add_node(app, "random-node", "local.random-probe", "1.0.0",
                           "", 0, 0, error, sizeof(error)));
    assert(nn_app_connect(app, "random-edge", "input", "out", "random-node", "in",
                          error, sizeof(error)));
    report = nn_app_analysis(app, error, sizeof(error));
    const NNInferenceResult *random_result = result_for(report, "random-node");
    assert(random_result && random_result->status == NN_INFERENCE_RUNTIME_FAULT);
    assert(nn_app_remove_node(app, "random-node", error, sizeof(error)));
    assert_multi_output_replay(app, error, sizeof(error));

    assert(nn_app_add_node(app, "work-owner", "core.repeat", "0.1.0", "",
                           0, 0, error, sizeof(error)));
    assert(nn_app_connect(app, "work-parent-edge", "input", "out", "work-owner", "in",
                          error, sizeof(error)));
    assert(nn_app_set_parameter_text(app, "work-owner", "times", "42",
                                     error, sizeof(error)));
    const NNNode *work_input = scoped_package(nn_app_model(app), "work-owner", "core.input");
    const NNNode *work_output = scoped_package(nn_app_model(app), "work-owner", "core.output");
    assert(work_input && work_output);
    char work_input_id[256], work_output_id[256], previous[256];
    assert(snprintf(work_input_id, sizeof(work_input_id), "%s", work_input->id) <
           (int)sizeof(work_input_id));
    assert(snprintf(work_output_id, sizeof(work_output_id), "%s", work_output->id) <
           (int)sizeof(work_output_id));
    assert(snprintf(previous, sizeof(previous), "%s", work_input_id) < (int)sizeof(previous));
    for (size_t i = 0; i < 360; ++i) {
        char id[64], edge[80];
        assert(snprintf(id, sizeof(id), "work-node-%03zu", i) < (int)sizeof(id));
        assert(nn_app_add_node(app, id, "core.relu", "0.1.0", "work-owner",
                               0, 0, error, sizeof(error)));
        assert(snprintf(edge, sizeof(edge), "work-edge-%03zu", i) < (int)sizeof(edge));
        assert(nn_app_connect(app, edge, previous, "out", id, "in",
                              error, sizeof(error)));
        assert(snprintf(previous, sizeof(previous), "%s", id) < (int)sizeof(previous));
    }
    assert(nn_app_connect(app, "work-terminal-edge", previous, "out",
                          work_output_id, "in", error, sizeof(error)));
    for (size_t repeat = 42; repeat <= 64; repeat = repeat == 42 ? 43 : 64) {
        char value[16];
        snprintf(value, sizeof(value), "%zu", repeat);
        assert(nn_app_set_parameter_text(app, "work-owner", "times", value,
                                         error, sizeof(error)));
        report = nn_app_analysis(app, error, sizeof(error));
        assert(report);
        assert(result_for(report, "work-owner")->status == NN_INFERENCE_SUCCESS);
        for (size_t i = 0; i < 360; ++i) {
            char id[64];
            snprintf(id, sizeof(id), "work-node-%03zu", i);
            const NNInferenceResult *child = result_for(report, id);
            assert_shape(child, "13", 2);
            assert(child->output_count == 1 &&
                   !strcmp(child->outputs[0].handle_id, "out"));
        }
        if (repeat == 64) break;
    }
    assert(nn_app_set_parameter_text(app, "work-owner", "times", "400",
                                     error, sizeof(error)));
    report = nn_app_analysis(app, error, sizeof(error));
    const NNInferenceResult *work_result = result_for(report, "work-owner");
    assert(work_result && work_result->status == NN_INFERENCE_RUNTIME_FAULT);
    assert(strstr(work_result->message, "work budget limit exceeded"));

    assert_instruction_budget(core);
    nn_app_free(app);
    assert(nftw(parent, remove_entry, 32, FTW_DEPTH | FTW_PHYS) == 0);
    return 0;
}
