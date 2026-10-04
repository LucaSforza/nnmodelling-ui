#define _XOPEN_SOURCE 700
#include "catalog_internal.h"
#include "utils/utils.h"

#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static void package_dispose(Package *p) {
  size_t i;
  free((char *)p->pub.directory);
  free((char *)p->pub.id);
  free((char *)p->pub.version);
  free((char *)p->pub.name);
  free((char *)p->pub.description);
  free((char *)p->pub.kind);
  free((char *)p->pub.color);
  free((char *)p->pub.inference_file);
  free((char *)p->pub.visualization_file);
  for (i = 0; p->outputs && i < p->pub.output_count; ++i) {
    free((char *)p->outputs[i].id);
    free((char *)p->outputs[i].type);
  }
  free(p->outputs);
  for (i = 0; p->parameters && i < p->pub.parameter_count; ++i) {
    free((char *)p->parameters[i].key);
    free((char *)p->parameters[i].type);
    free((char *)p->parameters[i].kind);
    free((char *)p->parameters[i].position);
    if (p->parameters[i].has_default &&
        (p->parameters[i].default_value.type == NN_PARAMETER_STRING ||
         p->parameters[i].default_value.type == NN_PARAMETER_JSON))
      free((char *)p->parameters[i].default_value.as.string);
    if (p->choice_storage && p->choice_storage[i]) {
      for (size_t j = 0; p->choice_storage[i][j]; ++j)
        free(p->choice_storage[i][j]);
      free(p->choice_storage[i]);
    }
  }
  free(p->choice_storage);
  free(p->parameters);
  for (i = 0; p->dependencies && i < p->pub.dependency_count; ++i) {
    free((char *)p->dependencies[i].id);
    free((char *)p->dependencies[i].version_constraint);
  }
  free(p->dependencies);
  memset(p, 0, sizeof(*p));
}

void nn_catalog_free(NNCatalog *catalog) {
  size_t i;
  if (!catalog)
    return;
  for (i = 0; i < catalog->count; ++i)
    package_dispose(&catalog->items[i]);
  free(catalog->items);
  free(catalog);
}

static bool load_package(NNCatalog *cat, const char *root, const char *rel,
                         const char *expected_id, const char *expected_version,
                         char *err, size_t cap) {
  char dir[PATH_MAX], manifest_path[PATH_MAX], def_path[PATH_MAX],
      inference_path[PATH_MAX], visualization_path[PATH_MAX];
  char manifest_rel[PATH_MAX], def_rel[PATH_MAX], inf_rel[PATH_MAX], vis_rel[PATH_MAX];
  char *buf = NULL;
  size_t len;
  yyjson_doc *doc = NULL;
  Package item = {0};
  bool ok = false;
  if (!nn_catalog_checked_path(root, rel, true, dir) || !nn_catalog_under_path(root, dir)) {
    nn_errorf(err, cap, "invalid or linked package path: %s", rel);
    return false;
  }
  if (snprintf(manifest_rel, sizeof(manifest_rel), "%s/manifest.json", rel) >=
          (int)sizeof(manifest_rel) ||
      !nn_catalog_checked_path(root, manifest_rel, false, manifest_path) ||
      !nn_catalog_under_path(root, manifest_path)) {
    nn_errorf(err, cap, "missing or unsafe package manifest: %s", rel);
    return false;
  }
  buf = nn_catalog_read_file(manifest_path, &len);
  if (!buf || !(doc = yyjson_read(buf, len, 0))) {
    goto done;
  }
  ok = nn_catalog_parse_manifest(&item, yyjson_doc_get_root(doc), dir);
  if (!ok) {
    nn_errorf(err, cap, "invalid package manifest: %s", rel);
    goto done;
  }
  if ((expected_id && strcmp(expected_id, item.pub.id)) ||
      (expected_version && strcmp(expected_version, item.pub.version))) {
    ok = false;
    nn_errorf(err, cap, "package identity does not match reference: %s", rel);
    goto done;
  }
  {
    yyjson_val *entry = nn_catalog_get(yyjson_doc_get_root(doc), "entrypoints");
    const char *name = nn_catalog_strval(nn_catalog_get(entry, "definition"));
    const char *file = item.pub.inference_file;
    if (snprintf(def_rel, sizeof(def_rel), "%s/%s", rel, name) >=
            (int)sizeof(def_rel) ||
        snprintf(inf_rel, sizeof(inf_rel), "%s/%s", rel, file) >=
            (int)sizeof(inf_rel) ||
        !nn_catalog_checked_path(root, def_rel, false, def_path) ||
        !nn_catalog_under_path(root, def_path) ||
        !nn_catalog_checked_path(root, inf_rel, false, inference_path) ||
        !nn_catalog_under_path(root, inference_path)) {
      ok = false;
      nn_errorf(err, cap, "invalid package entrypoint path: %s", rel);
      goto done;
    }
    if (item.pub.visualization_file &&
        (snprintf(vis_rel, sizeof(vis_rel), "%s/%s", rel,
                  item.pub.visualization_file) >= (int)sizeof(vis_rel) ||
         !nn_catalog_checked_path(root, vis_rel, false, visualization_path) ||
         !nn_catalog_under_path(root, visualization_path))) {
      ok = false;
      nn_errorf(err, cap, "invalid visualization entrypoint path: %s", rel);
      goto done;
    }
    if (item.pub.visualization_file) {
      struct stat info;
      if (lstat(visualization_path, &info) || !S_ISREG(info.st_mode) ||
          info.st_size < 0 || (size_t)info.st_size > 1024u * 1024u) {
        ok = false;
        nn_errorf(err, cap, "invalid visualization Lua file: %s", rel);
        goto done;
      }
    }
  }
  {
    size_t dlen;
    char *definition = nn_catalog_read_file(def_path, &dlen);
    yyjson_doc *ddoc = definition ? yyjson_read(definition, dlen, 0) : NULL;
    if (!ddoc || !nn_catalog_parse_definition(&item, yyjson_doc_get_root(ddoc))) {
      if (ddoc)
        yyjson_doc_free(ddoc);
      free(definition);
      ok = false;
      nn_errorf(err, cap, "invalid package definition: %s", rel);
      goto done;
    }
    yyjson_doc_free(ddoc);
    free(definition);
  }
  if (item.pub.visualization_file && strcmp(item.pub.kind, "subflow")) {
    ok = false;
    nn_errorf(err, cap, "visualization entrypoint requires kind=subflow: %s", rel);
    goto done;
  }
  if (cat->count >= CATALOG_ITEM_LIMIT) {
    ok = false;
    nn_errorf(err, cap, "too many packages at %s", rel);
    goto done;
  }
  {
    Package *grown = realloc(cat->items, (cat->count + 1) * sizeof(*grown));
    if (!grown) {
      ok = false;
      goto done;
    }
    cat->items = grown;
    cat->items[cat->count++] = item;
    memset(&item, 0, sizeof(item));
  }
  ok = true;
done:
  if (!ok && err && cap && !err[0])
    nn_errorf(err, cap, "cannot load package: %s", rel);
  if (doc)
    yyjson_doc_free(doc);
  free(buf);
  package_dispose(&item);
  return ok;
}

NNCatalog *nn_catalog_load(const char *core_root, const char *project_root,
                           const NNResourceRef *custom, size_t custom_count,
                           char *err, size_t cap) {
  NNCatalog *cat = NULL;
  char core[PATH_MAX], project[PATH_MAX];
  DIR *dir = NULL;
  struct dirent *entry;
  struct stat st;
  if (err && cap)
    err[0] = '\0';
  if (!core_root || !project_root || (custom_count && !custom) ||
      custom_count > CATALOG_ITEM_LIMIT || !realpath(core_root, core) ||
      !realpath(project_root, project) || stat(core, &st) ||
      !S_ISDIR(st.st_mode) || stat(project, &st) || !S_ISDIR(st.st_mode)) {
    nn_error_set(err, cap, "invalid catalog roots");
    return NULL;
  }
  cat = calloc(1, sizeof(*cat));
  if (!cat) {
    nn_error_set(err, cap, "out of memory");
    return NULL;
  }
  dir = opendir(core);
  if (!dir) {
    nn_errorf(err, cap, "cannot open core package directory: %s", core);
    goto fail;
  }
  while ((entry = readdir(dir))) {
    char path[PATH_MAX], rel[PATH_MAX];
    if (entry->d_name[0] == '.')
      continue;
    if (snprintf(path, sizeof(path), "%s/%s", core, entry->d_name) >=
        (int)sizeof(path)) {
      nn_errorf(err, cap, "core package path too long: %s", entry->d_name);
      goto fail;
    }
    if (lstat(path, &st) != 0) {
      nn_errorf(err, cap, "cannot inspect core package: %s", entry->d_name);
      goto fail;
    }
    if (S_ISLNK(st.st_mode)) {
      nn_errorf(err, cap, "symlink in core package catalog: %s", entry->d_name);
      goto fail;
    }
    if (!S_ISDIR(st.st_mode))
      continue;
    strcpy(rel, entry->d_name);
    if (!load_package(cat, core, rel, NULL, NULL, err, cap))
      goto fail;
  }
  closedir(dir);
  dir = NULL;
  for (size_t i = 0; i < custom_count; ++i) {
    unsigned long a, b, c;
    if (!custom[i].id || !nn_catalog_valid_id(custom[i].id) ||
        !nn_catalog_semver(custom[i].version, &a, &b, &c) || !custom[i].path ||
        !nn_catalog_safe_rel(custom[i].path)) {
      nn_errorf(err, cap, "invalid custom package reference: %s",
                custom[i].id ? custom[i].id : "");
      goto fail;
    }
    if (!load_package(cat, project, custom[i].path, custom[i].id,
                      custom[i].version, err, cap))
      goto fail;
  }
  if (!nn_catalog_validate_dependencies(cat, err, cap))
    goto fail;
  return cat;
fail:
  if (dir)
    closedir(dir);
  nn_catalog_free(cat);
  return NULL;
}

size_t nn_catalog_count(const NNCatalog *catalog) {
  return catalog ? catalog->count : 0;
}

const NNPackage *nn_catalog_at(const NNCatalog *catalog, size_t index) {
  return catalog && index < catalog->count ? &catalog->items[index].pub : NULL;
}

const NNPackage *nn_catalog_find(const NNCatalog *catalog, const char *id,
                                 const char *version) {
  size_t i;
  if (!catalog || !id || !version)
    return NULL;
  for (i = 0; i < catalog->count; ++i)
    if (!strcmp(catalog->items[i].pub.id, id) &&
        !strcmp(catalog->items[i].pub.version, version))
      return &catalog->items[i].pub;
  return NULL;
}

const NNPackage *nn_catalog_resolve(const NNCatalog *catalog, const char *id,
                                   const char *version_constraint) {
  const NNPackage *match = NULL;
  if (!catalog || !id || !version_constraint)
    return NULL;
  for (size_t i = 0; i < catalog->count; ++i) {
    const NNPackage *candidate = &catalog->items[i].pub;
    if (strcmp(candidate->id, id) ||
        !nn_catalog_version_satisfies(candidate->version, version_constraint))
      continue;
    if (match)
      return NULL;
    match = candidate;
  }
  return match;
}
