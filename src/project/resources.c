#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include "project_internal.h"
#include "utils/utils.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool write_new_text(const char *path, const char *text)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0644);
    if (fd < 0) return false;
    size_t length = strlen(text);
    bool okay = nn_project_write_all(fd, text, length) && fsync(fd) == 0;
    if (close(fd)) okay = false;
    return okay;
}

static char *resource_manifest(const char *id, const char *version,
                               const char *definition, const char *lua,
                               const char *dependencies)
{
    yyjson_doc *deps_doc = dependencies ? yyjson_read(dependencies, strlen(dependencies), 0) : NULL;
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc || !deps_doc || !yyjson_is_obj(yyjson_doc_get_root(deps_doc))) {
        if (doc) yyjson_mut_doc_free(doc);
        if (deps_doc) yyjson_doc_free(deps_doc);
        return NULL;
    }
    yyjson_mut_val *root = yyjson_mut_obj(doc), *entry = yyjson_mut_obj(doc), *inf = yyjson_mut_obj(doc), *python = yyjson_mut_obj(doc);
    yyjson_mut_val *deps = yyjson_val_mut_copy(doc, yyjson_doc_get_root(deps_doc));
    bool okay = root && entry && inf && python && deps &&
        yyjson_mut_obj_add_int(doc, root, "schemaVersion", 1) &&
        yyjson_mut_obj_add_strcpy(doc, root, "id", id) &&
        yyjson_mut_obj_add_strcpy(doc, root, "version", version) &&
        yyjson_mut_obj_add_val(doc, root, "dependencies", deps) &&
        yyjson_mut_obj_add_val(doc, root, "entrypoints", entry) &&
        yyjson_mut_obj_add_strcpy(doc, entry, "definition", definition) &&
        yyjson_mut_obj_add_val(doc, entry, "inference", inf) &&
        yyjson_mut_obj_add_strcpy(doc, inf, "language", "lua") &&
        yyjson_mut_obj_add_strcpy(doc, inf, "file", lua) &&
        yyjson_mut_obj_add_val(doc, entry, "pytorch", python) &&
        yyjson_mut_obj_add_strcpy(doc, python, "language", "python") &&
        yyjson_mut_obj_add_strcpy(doc, python, "file", "pytorch.py");
    char *json = NULL;
    size_t length = 0;
    if (okay) { yyjson_mut_doc_set_root(doc, root); json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL); }
    if (json) { char *grown = realloc(json, length + 2); if (!grown) { free(json); json = NULL; } else { grown[length++] = '\n'; grown[length] = 0; json = grown; } }
    yyjson_mut_doc_free(doc); yyjson_doc_free(deps_doc);
    return json;
}

static char *dataset_manifest(const char *id, const char *version)
{
    yyjson_mut_doc *doc = yyjson_mut_doc_new(NULL);
    if (!doc) return NULL;
    yyjson_mut_val *root = yyjson_mut_obj(doc), *entry = yyjson_mut_obj(doc), *python = yyjson_mut_obj(doc);
    bool okay = root && entry && python && yyjson_mut_obj_add_int(doc, root, "schemaVersion", 1) &&
        yyjson_mut_obj_add_strcpy(doc, root, "id", id) &&
        yyjson_mut_obj_add_strcpy(doc, root, "version", version) &&
        yyjson_mut_obj_add_val(doc, root, "entrypoints", entry) &&
        yyjson_mut_obj_add_strcpy(doc, entry, "definition", "dataset.json") &&
        yyjson_mut_obj_add_val(doc, entry, "python", python) &&
        yyjson_mut_obj_add_strcpy(doc, python, "language", "python") &&
        yyjson_mut_obj_add_strcpy(doc, python, "file", "dataset.py");
    char *json = NULL; size_t length = 0;
    if (okay) { yyjson_mut_doc_set_root(doc, root); json = yyjson_mut_write_opts(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, NULL, &length, NULL); }
    if (json) { char *grown = realloc(json, length + 2); if (!grown) { free(json); json = NULL; } else { grown[length++] = '\n'; grown[length] = 0; json = grown; } }
    yyjson_mut_doc_free(doc); return json;
}

static char *pretty_json(const char *source)
{
    if (!source) return NULL;
    yyjson_doc *doc = yyjson_read(source, strlen(source), 0);
    if (!doc) return NULL;
    size_t length = 0;
    char *text = yyjson_val_write_opts(yyjson_doc_get_root(doc), YYJSON_WRITE_PRETTY_TWO_SPACES,
                                       NULL, &length, NULL);
    yyjson_doc_free(doc);
    if (!text) return NULL;
    char *grown = realloc(text, length + 2);
    if (!grown) { free(text); return NULL; }
    grown[length++] = '\n'; grown[length] = '\0';
    return grown;
}

static char *python_pyproject(const char *id, const char *version)
{
    const int needed = snprintf(NULL, 0,
        "[project]\nname = \"nnmodelling-%s\"\nversion = \"%s\"\ndependencies = [\"nnmodelling-runtime>=0.1.0\"]\n",
        id, version);
    if (needed < 0) return NULL;
    char *text = malloc((size_t)needed + 1);
    if (!text) return NULL;
    if (snprintf(text, (size_t)needed + 1,
        "[project]\nname = \"nnmodelling-%s\"\nversion = \"%s\"\ndependencies = [\"nnmodelling-runtime>=0.1.0\"]\n",
        id, version) != needed) { free(text); return NULL; }
    return text;
}

static const char *dataset_scaffold(void)
{
    return "from collections.abc import Iterator\nfrom typing import Generic, TypeVar\n\nfrom torch import Tensor\nfrom nnmodelling_runtime import Batch, DatasetAdapter\n\nInputT = TypeVar(\"InputT\")\nOutputT = TypeVar(\"OutputT\")\n\n\nclass Dataset(DatasetAdapter[InputT, OutputT], Generic[InputT, OutputT]):\n    def tokenize(self, value: InputT) -> Tensor | dict[str, Tensor]:\n        raise NotImplementedError(\"Implement dataset input tokenization.\")\n\n    def untokenize(self, tensor: Tensor) -> OutputT:\n        raise NotImplementedError(\"Implement prediction decoding.\")\n\n    def load(self, split: str, batch_size: int) -> Iterator[Batch]:\n        raise NotImplementedError(\"Load and batch the requested dataset split.\")\n";
}

static const char *pytorch_scaffold(void)
{
    return "def build(parameters, context, services):\n    raise NotImplementedError(\"Implement the PyTorch stereotype build function.\")\n";
}

static bool make_resource_directory(const char *root, const char *category,
                                    const char *name, char **relative, char **absolute,
                                    bool *category_created)
{
    *relative = NULL; *absolute = NULL;
    *category_created = false;
    char *base = nn_path_join(root, category);
    if (!base) return false;
    struct stat st;
    if (lstat(base, &st)) {
        if (errno != ENOENT || mkdir(base, 0755)) { free(base); return false; }
        *category_created = true;
    } else if (!S_ISDIR(st.st_mode)) { free(base); return false; }
    char *rel = nn_path_join(category, name);
    char *dir = rel ? nn_path_join(root, rel) : NULL;
    free(base);
    if (!rel || !dir) {
        free(rel); free(dir);
        if (*category_created) { char *created = nn_path_join(root, category); if (created) { rmdir(created); free(created); } }
        return false;
    }
    if (mkdir(dir, 0755)) {
        free(rel); free(dir);
        if (*category_created) { char *created = nn_path_join(root, category); if (created) { rmdir(created); free(created); } }
        return false;
    }
    *relative = rel; *absolute = dir; return true;
}

bool nn_project_create_stereotype(NNProject *p, const char *id, const char *version,
                                  const char *definition_json, const char *lua,
                                  const char *dependencies_json, char *error, size_t cap)
{
    if (error && cap) error[0] = '\0';
    if (!p || !nn_project_valid_id(id) || !version || !definition_json || !lua || strlen(definition_json) > 4u * 1024u * 1024u ||
        (dependencies_json && strlen(dependencies_json) > 4u * 1024u * 1024u) || strlen(lua) > 1024 * 1024 ||
        !nn_project_valid_semver(version)) { nn_errorf(error, cap, "invalid stereotype identity or payload"); return false; }
    if (p->package_count >= 128) { nn_errorf(error, cap, "package catalog limit reached"); return false; }
    for (size_t i = 0; i < p->package_count; ++i) if (!strcmp(p->packages[i].id, id) && !strcmp(p->packages[i].version, version)) { nn_errorf(error, cap, "duplicate package identity"); return false; }
    for (size_t i = 0; i < nn_catalog_count(p->catalog); ++i) { const NNPackage *x = nn_catalog_at(p->catalog, i); if (!strcmp(x->id, id) && !strcmp(x->version, version)) { nn_errorf(error, cap, "core package identity is immutable"); return false; } }
    char name[256]; if (snprintf(name, sizeof(name), "%s-%s", id, version) >= (int)sizeof(name)) { nn_errorf(error, cap, "resource path too long"); return false; }
    char *rel = NULL, *dir = NULL, *definition = pretty_json(definition_json); bool category_created = false;
    char *manifest = resource_manifest(id, version, "definition.json", "inference.lua", dependencies_json ? dependencies_json : "{}");
    if ((definition && strlen(definition) > 4u * 1024u * 1024u) ||
        (manifest && strlen(manifest) > 4u * 1024u * 1024u)) {
        free(manifest); free(definition); nn_errorf(error, cap, "package JSON exceeds catalog file limit"); return false;
    }
    if (!manifest || !definition || !make_resource_directory(p->directory, "packages", name, &rel, &dir, &category_created)) { free(manifest); free(definition); free(rel); free(dir); nn_errorf(error, cap, "cannot create package resource directory"); return false; }
    char *mp = nn_path_join(dir, "manifest.json"), *dp = nn_path_join(dir, "definition.json"), *lp = nn_path_join(dir, "inference.lua");
    char *py = nn_path_join(dir, "pyproject.toml"), *pt = nn_path_join(dir, "pytorch.py");
    char *pyproject = python_pyproject(id, version);
    bool okay = mp && dp && lp && py && pt && write_new_text(mp, manifest) && write_new_text(dp, definition) &&
        pyproject && write_new_text(lp, lua) && write_new_text(py, pyproject) && write_new_text(pt, pytorch_scaffold());
    free(manifest); free(definition); free(mp); free(dp); free(lp); free(py); free(pt); free(pyproject);
    NNResourceRef *refs = NULL; NNCatalog *candidate = NULL;
    if (okay) {
        refs = calloc(p->package_count + 1, sizeof(*refs)); okay = refs != NULL;
        for (size_t i = 0; okay && i < p->package_count; ++i) {
            refs[i] = (NNResourceRef){nn_text_copy(p->packages[i].id), nn_text_copy(p->packages[i].version), nn_text_copy(p->packages[i].path)};
            okay = refs[i].id && refs[i].version && refs[i].path;
        }
        if (okay) refs[p->package_count] = (NNResourceRef){nn_text_copy(id), nn_text_copy(version), nn_text_copy(rel)};
        if (okay) okay = refs[p->package_count].id && refs[p->package_count].version && refs[p->package_count].path;
        if (okay) candidate = nn_catalog_load(p->core_root, p->directory, refs, p->package_count + 1, error, cap);
        okay = okay && candidate != NULL;
    }
    for (size_t i = 0; okay && i < nn_model_node_count(p->model); ++i) {
        const NNNode *node = nn_model_node_at(p->model, i);
        if (!nn_catalog_find(candidate, node->package_id, node->package_version)) { nn_errorf(error, cap, "candidate catalog invalidates graph package"); okay = false; }
    }
    for (size_t i = 0; okay && i < nn_model_edge_count(p->model); ++i)
        okay = nn_project_edge_topology_valid(p->model, candidate, nn_model_edge_at(p->model, i),
                                   error, cap);
    if (okay) {
        NNResourceRef *old_refs = p->packages; size_t old_count = p->package_count;
        NNCatalog *old_catalog = p->catalog; bool old_dirty = p->dirty;
        p->packages = refs; p->package_count++; p->catalog = candidate;
        p->dirty = true;
        okay = nn_project_save(p, error, cap);
        if (okay) { for (size_t i = 0; i < old_count; ++i) { free((char *)old_refs[i].id); free((char *)old_refs[i].version); free((char *)old_refs[i].path); } free(old_refs); nn_catalog_free(old_catalog); refs = NULL; candidate = NULL; }
        else { p->packages = old_refs; p->package_count--; p->catalog = old_catalog; p->dirty = old_dirty; }
    }
    if (refs) { for (size_t i = 0; i <= p->package_count; ++i) { free((char *)refs[i].id); free((char *)refs[i].version); free((char *)refs[i].path); } free(refs); }
    nn_catalog_free(candidate); if (!okay) { nn_project_remove_tree(dir); if (category_created) { char *base = nn_path_join(p->directory, "packages"); if (base) { rmdir(base); free(base); } } }
    free(rel); free(dir);
    if (!okay && error && cap && !error[0]) nn_errorf(error, cap, "stereotype creation failed");
    return okay;
}

bool nn_project_create_dataset(NNProject *p, const char *id, const char *version,
                               const char *definition_json, bool select,
                               char *error, size_t cap)
{
    if (error && cap) error[0] = '\0';
    if (!p || !nn_project_valid_id(id) || !nn_project_valid_semver(version) || !definition_json) { nn_errorf(error, cap, "invalid dataset identity or payload"); return false; }
    if (p->dataset_count >= 128) { nn_errorf(error, cap, "dataset catalog limit reached"); return false; }
    for (size_t i = 0; i < p->dataset_count; ++i) if (!strcmp(p->datasets[i].id, id) && !strcmp(p->datasets[i].version, version)) { nn_errorf(error, cap, "duplicate dataset identity"); return false; }
    char name[256]; if (snprintf(name, sizeof(name), "%s-%s", id, version) >= (int)sizeof(name)) { nn_errorf(error, cap, "resource path too long"); return false; }
    if (strlen(definition_json) > 16u * 1024u * 1024u) { nn_errorf(error, cap, "dataset JSON exceeds resource file limit"); return false; }
    char *rel = NULL, *dir = NULL, *definition = pretty_json(definition_json), *manifest = dataset_manifest(id, version); bool category_created = false;
    if (definition && strlen(definition) > 16u * 1024u * 1024u) {
        free(manifest); free(definition); nn_errorf(error, cap, "dataset JSON exceeds resource file limit"); return false;
    }
    if (!manifest || !definition || !make_resource_directory(p->directory, "datasets", name, &rel, &dir, &category_created)) { free(manifest); free(definition); free(rel); free(dir); nn_errorf(error, cap, "cannot create dataset resource directory"); return false; }
    char *mp = nn_path_join(dir, "manifest.json"), *dp = nn_path_join(dir, "dataset.json");
    char *py = nn_path_join(dir, "pyproject.toml"), *ds = nn_path_join(dir, "dataset.py");
    char *pyproject = python_pyproject(id, version);
    bool okay = mp && dp && py && ds && write_new_text(mp, manifest) && write_new_text(dp, definition) &&
        pyproject && write_new_text(py, pyproject) && write_new_text(ds, dataset_scaffold());
    free(manifest); free(definition); free(mp); free(dp); free(py); free(ds); free(pyproject);
    NNDataset candidate = { .id = nn_text_copy(id), .version = nn_text_copy(version), .path = nn_text_copy(rel) };
    if (okay) okay = candidate.id && candidate.version && candidate.path && nn_project_load_dataset(p, &candidate, error, cap);
    NNDataset *grown = okay ? malloc((p->dataset_count + 1) * sizeof(*grown)) : NULL;
    if (okay && !grown) okay = false;
    if (okay) {
        NNDataset *old = p->datasets; size_t old_count = p->dataset_count;
        if (old_count) memcpy(grown, old, old_count * sizeof(*grown));
        char *old_id = p->active_dataset_id, *old_version = p->active_dataset_version; bool old_dirty = p->dirty;
        p->datasets = grown; p->datasets[p->dataset_count++] = candidate; memset(&candidate, 0, sizeof(candidate));
        if (select) { p->active_dataset_id = nn_text_copy(id); p->active_dataset_version = nn_text_copy(version); if (!p->active_dataset_id || !p->active_dataset_version) okay = false; }
        p->dirty = true;
        if (okay) okay = nn_project_save(p, error, cap);
        if (okay) { free(old); if (select) { free(old_id); free(old_version); } }
        else {
            if (select) { free(p->active_dataset_id); free(p->active_dataset_version); p->active_dataset_id = old_id; p->active_dataset_version = old_version; }
            p->dataset_count = old_count; p->datasets = old; p->dirty = old_dirty;
            nn_project_dataset_dispose(&grown[old_count]);
            free(grown);
        }
    }
    nn_project_dataset_dispose(&candidate);
    if (!okay) { nn_project_remove_tree(dir); if (category_created) { char *base = nn_path_join(p->directory, "datasets"); if (base) { rmdir(base); free(base); } } }
    free(rel); free(dir);
    if (!okay && error && cap && !error[0]) nn_errorf(error, cap, "dataset creation failed");
    return okay;
}
