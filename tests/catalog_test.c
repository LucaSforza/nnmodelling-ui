#define _XOPEN_SOURCE 700
#include "catalog/catalog.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(text, 1, strlen(text), f) == strlen(text));
    assert(fclose(f) == 0);
}

static void make_path(char *out, size_t capacity, const char *left, const char *right) {
    assert(snprintf(out, capacity, "%s/%s", left, right) < (int)capacity);
}

static void make_package(const char *root, const char *folder, const char *id,
                         const char *deps) {
    char path[1024];
    assert(snprintf(path, sizeof(path), "%s/%s", root, folder) < (int)sizeof(path));
    assert(mkdir(path, 0700) == 0);
    char manifest[2048];
    assert(snprintf(manifest, sizeof(manifest),
        "{\"schemaVersion\":1,\"id\":\"%s\",\"version\":\"1.0.0\","
        "\"dependencies\":{%s},\"entrypoints\":{\"definition\":\"stereotype.json\","
        "\"inference\":{\"language\":\"lua\",\"file\":\"inference.lua\"}}}", id, deps) < (int)sizeof(manifest));
    char file[2048];
    snprintf(file, sizeof(file), "%s/manifest.json", path); write_text(file, manifest);
    snprintf(file, sizeof(file), "%s/stereotype.json", path);
    write_text(file, "{\"name\":\"Fixture\",\"kind\":\"layer\",\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},\"parameters\":{}}");
    snprintf(file, sizeof(file), "%s/inference.lua", path); write_text(file, "return function() end\n");
}

static void expect_invalid(const char *project, NNResourceRef ref) {
    char error[256] = {0};
    NNCatalog *catalog = nn_catalog_load("stereotype-packages/core", project, &ref, 1, error, sizeof(error));
    assert(catalog == NULL);
    assert(error[0] != '\0');
}

int main(void) {
    char error[256] = {0};
    NNCatalog *catalog = nn_catalog_load("stereotype-packages/core", ".", NULL, 0, error, sizeof(error));
    const NNPackage *package;
    char temp[] = "/tmp/nn-catalog-test-XXXXXX";
    char project[1024], packages[2048];
    assert(catalog && error[0] == '\0');
    assert(nn_catalog_count(catalog) == 26);
    for (size_t i = 0; i < nn_catalog_count(catalog); ++i) {
        package = nn_catalog_at(catalog, i);
        assert(package && package->parameters_file &&
               !strcmp(package->parameters_file, "parameters.lua"));
    }
    package = nn_catalog_find(catalog, "core.horizontal-repeat", "0.1.0");
    assert(package && package->dependency_count == 1);
    assert(strcmp(package->dependencies[0].id, "core.concat") == 0);
    assert(strcmp(package->inference_file, "inference.lua") == 0);
    package = nn_catalog_find(catalog, "core.cast", "0.1.0");
    assert(package && package->parameter_count == 1);
    assert(strcmp(package->parameters[0].type, "dtype") == 0);
    assert(package->parameters[0].choice_count == 10);
    assert(package->directory && package->directory[0] == '/');
    package = nn_catalog_find(catalog, "core.relu", "0.1.0");
    assert(package && package->output_count == 1);
    assert(!strcmp(package->outputs[0].id, "out"));
    assert(!strcmp(package->outputs[0].type, "output"));
    package = nn_catalog_find(catalog, "core.mse-loss", "0.1.0");
    assert(package && package->output_count == 1);
    assert(!strcmp(package->outputs[0].id, "loss"));
    assert(!strcmp(package->outputs[0].type, "loss"));
    package = nn_catalog_find(catalog, "core.loss-output", "0.1.0");
    assert(package && package->output_count == 0);
    package = nn_catalog_find(catalog, "core.repeat", "0.1.0");
    assert(package && package->parameter_count == 1);
    assert(!strcmp(package->parameters[0].key, "times"));
    assert(package->parameters[0].position && !strcmp(package->parameters[0].position, "top"));
    package = nn_catalog_find(catalog, "core.linear", "0.1.0");
    assert(package && !strcmp(package->parameters_file, "parameters.lua"));
    nn_catalog_free(catalog);

    assert(mkdtemp(temp));
    snprintf(project, sizeof(project), "%s/project", temp); assert(mkdir(project, 0700) == 0);
    snprintf(packages, sizeof(packages), "%s/packages", project); assert(mkdir(packages, 0700) == 0);
    make_package(packages, "custom", "custom.fixture", "");
    {
        NNResourceRef good = {"custom.fixture", "1.0.0", "packages/custom"};
        catalog = nn_catalog_load("stereotype-packages/core", project, &good, 1, error, sizeof(error));
        assert(catalog && nn_catalog_find(catalog, good.id, good.version));
        nn_catalog_free(catalog);
        char package_dir[8192], manifest_path[8192], parameter_path[8192];
        make_path(package_dir, sizeof(package_dir), packages, "custom");
        make_path(manifest_path, sizeof(manifest_path), package_dir, "manifest.json");
        make_path(parameter_path, sizeof(parameter_path), package_dir, "parameters.lua");
        write_text(manifest_path,
            "{\"schemaVersion\":1,\"id\":\"custom.fixture\",\"version\":\"1.0.0\","
            "\"dependencies\":{},\"entrypoints\":{\"definition\":\"stereotype.json\","
            "\"inference\":{\"language\":\"lua\",\"file\":\"inference.lua\"},"
            "\"parameters\":{\"language\":\"lua\",\"file\":\"parameters.lua\"}}}");
        write_text(parameter_path, "return function() return {status='success',total=0,trainable=0} end");
        catalog = nn_catalog_load("stereotype-packages/core", project, &good, 1, error, sizeof(error));
        assert(catalog && !strcmp(nn_catalog_find(catalog, good.id, good.version)->parameters_file,
                                   "parameters.lua"));
        nn_catalog_free(catalog);
        write_text(manifest_path,
            "{\"schemaVersion\":1,\"id\":\"custom.fixture\",\"version\":\"1.0.0\","
            "\"dependencies\":{},\"entrypoints\":{\"definition\":\"stereotype.json\","
            "\"inference\":{\"language\":\"lua\",\"file\":\"inference.lua\"},"
            "\"parameters\":{\"language\":\"lua\",\"file\":\"../outside.lua\"}}}");
        expect_invalid(project, good);
        write_text(manifest_path,
            "{\"schemaVersion\":1,\"id\":\"custom.fixture\",\"version\":\"1.0.0\","
            "\"dependencies\":{},\"entrypoints\":{\"definition\":\"stereotype.json\","
            "\"inference\":{\"language\":\"lua\",\"file\":\"inference.lua\"},"
            "\"parameters\":{\"language\":\"lua\",\"file\":\"parameters.lua\"}}}");
        assert(unlink(parameter_path) == 0);
        char outside_path[2048];
        snprintf(outside_path, sizeof(outside_path), "%s/outside.lua", project);
        write_text(outside_path, "return function() end");
        assert(symlink(outside_path, parameter_path) == 0);
        expect_invalid(project, good);
        assert(unlink(parameter_path) == 0);
        write_text(parameter_path, "return function() return {status='success',total=0,trainable=0} end");
        NNResourceRef bad_identity = {"custom.wrong", "1.0.0", "packages/custom"};
        expect_invalid(project, bad_identity);
        NNResourceRef traversal = {"custom.fixture", "1.0.0", "../outside"};
        expect_invalid(project, traversal);
    }
    {
        char definition[4096];
        assert(snprintf(definition, sizeof(definition), "%s/custom/stereotype.json", packages) < (int)sizeof(definition));
        write_text(definition,
            "{\"name\":\"Fixture\",\"kind\":\"layer\","
            "\"outputs\":[{\"id\":\"prediction\",\"type\":\"output\"},"
            "{\"id\":\"objective\",\"type\":\"loss\"}],"
            "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},\"parameters\":{}}");
        NNResourceRef good = {"custom.fixture", "1.0.0", "packages/custom"};
        catalog = nn_catalog_load("stereotype-packages/core", project, &good, 1, error, sizeof(error));
        const NNPackage *override = nn_catalog_find(catalog, good.id, good.version);
        assert(override && override->output_count == 2);
        assert(!strcmp(override->outputs[1].id, "objective"));
        nn_catalog_free(catalog);
        write_text(definition,
            "{\"name\":\"Fixture\",\"kind\":\"layer\","
            "\"outputs\":[{\"id\":\"a\",\"type\":\"loss\"},"
            "{\"id\":\"b\",\"type\":\"loss\"}],"
            "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},\"parameters\":{}}");
        expect_invalid(project, good);
        write_text(definition,
            "{\"name\":\"Fixture\",\"kind\":\"layer\","
            "\"outputs\":[{\"id\":\"out\\u0000evil\",\"type\":\"output\"}],"
            "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},\"parameters\":{}}");
        expect_invalid(project, good);
        write_text(definition,
            "{\"name\":\"Fixture\",\"kind\":\"layer\","
            "\"outputs\":[{\"id\":\"out\",\"type\":\"loss\\u0000evil\"}],"
            "\"view\":{\"color\":\"#123456\",\"width\":120,\"height\":80},\"parameters\":{}}");
        expect_invalid(project, good);
    }
    make_package(packages, "broken", "custom.broken", "\"missing.package\":\"^1.0.0\"");
    {
        NNResourceRef missing_dep = {"custom.broken", "1.0.0", "packages/broken"};
        expect_invalid(project, missing_dep);
    }
    make_package(packages, "cycle-a", "custom.cycle-a", "\"custom.cycle-b\":\"^1.0.0\"");
    make_package(packages, "cycle-b", "custom.cycle-b", "\"custom.cycle-a\":\"^1.0.0\"");
    {
        NNResourceRef cycle[2] = {
            {"custom.cycle-a", "1.0.0", "packages/cycle-a"},
            {"custom.cycle-b", "1.0.0", "packages/cycle-b"}
        };
        catalog = nn_catalog_load("stereotype-packages/core", project, cycle, 2, error, sizeof(error));
        assert(!catalog && strstr(error, "cycle"));
    }
    return 0;
}
