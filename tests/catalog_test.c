#define _XOPEN_SOURCE 700
#include "catalog.h"

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
    assert(nn_catalog_count(catalog) == 25);
    package = nn_catalog_find(catalog, "core.horizontal-repeat", "0.1.0");
    assert(package && package->dependency_count == 1);
    assert(strcmp(package->dependencies[0].id, "core.concat") == 0);
    assert(strcmp(package->inference_file, "inference.lua") == 0);
    package = nn_catalog_find(catalog, "core.cast", "0.1.0");
    assert(package && package->parameter_count == 1);
    assert(strcmp(package->parameters[0].type, "dtype") == 0);
    assert(package->parameters[0].choice_count == 10);
    assert(package->directory && package->directory[0] == '/');
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
        NNResourceRef bad_identity = {"custom.wrong", "1.0.0", "packages/custom"};
        expect_invalid(project, bad_identity);
        NNResourceRef traversal = {"custom.fixture", "1.0.0", "../outside"};
        expect_invalid(project, traversal);
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
