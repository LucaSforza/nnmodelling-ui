#define _XOPEN_SOURCE 700
#include "catalog.h"
#include "utils.h"

#include "yyjson.h"

#include <dirent.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CATALOG_FILE_LIMIT (4u * 1024u * 1024u)
#define CATALOG_ITEM_LIMIT 4096u

typedef struct {
  NNPackage pub;
  NNParameterDef *parameters;
  char ***choice_storage;
  NNPackageDependency *dependencies;
} Package;

struct NNCatalog {
  Package *items;
  size_t count;
};

static bool valid_id(const char *s) {
  if (!s || !*s)
    return false;
  for (; *s; ++s)
    if (!((*s >= 'a' && *s <= 'z') || (*s >= 'A' && *s <= 'Z') ||
          (*s >= '0' && *s <= '9') || *s == '.' || *s == '_' || *s == '-'))
      return false;
  return true;
}

static bool semver(const char *s, unsigned long *a, unsigned long *b,
                   unsigned long *c) {
  unsigned long *parts[3] = {a, b, c};
  size_t i;
  if (!s || !*s)
    return false;
  for (i = 0; i < 3; ++i) {
    const char *start = s;
    unsigned long value = 0;
    if (*s < '0' || *s > '9')
      return false;
    if (*s == '0' && s[1] >= '0' && s[1] <= '9')
      return false;
    while (*s >= '0' && *s <= '9') {
      unsigned digit = (unsigned)(*s++ - '0');
      if (value > (ULONG_MAX - digit) / 10)
        return false;
      value = value * 10 + digit;
    }
    if (s == start)
      return false;
    *parts[i] = value;
    if (i < 2) {
      if (*s++ != '.')
        return false;
    } else if (*s)
      return false;
  }
  return true;
}

static bool safe_rel(const char *s) {
  const char *p;
  if (!s || !*s || *s == '/' || strchr(s, '\\'))
    return false;
  p = s;
  while (*p) {
    const char *start = p;
    while (*p && *p != '/')
      ++p;
    if (p == start || (p - start == 1 && start[0] == '.') ||
        (p - start == 2 && start[0] == '.' && start[1] == '.'))
      return false;
    if (*p)
      ++p;
  }
  return true;
}

/* Check each component with lstat so a symlink cannot hide an escape. */
static bool checked_path(const char *base, const char *rel, bool want_dir,
                         char out[PATH_MAX]) {
  char current[PATH_MAX];
  const char *p;
  struct stat st;
  if (!base || !safe_rel(rel) ||
      strlen(base) + strlen(rel) + 2 >= sizeof(current))
    return false;
  strcpy(current, base);
  p = rel;
  while (*p) {
    char *end = current + strlen(current);
    const char *start = p;
    size_t n;
    while (*p && *p != '/')
      ++p;
    n = (size_t)(p - start);
    *end++ = '/';
    memcpy(end, start, n);
    end[n] = '\0';
    if (lstat(current, &st) != 0 || S_ISLNK(st.st_mode))
      return false;
    if (*p && !S_ISDIR(st.st_mode))
      return false;
    if (*p)
      ++p;
  }
  if ((want_dir && !S_ISDIR(st.st_mode)) || (!want_dir && !S_ISREG(st.st_mode)))
    return false;
  if (!realpath(current, out))
    return false;
  return true;
}

static bool under_path(const char *root, const char *path) {
  size_t n = strlen(root);
  return strncmp(root, path, n) == 0 && path[n] == '/';
}

static char *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  long size;
  char *buf;
  if (!f)
    return NULL;
  if (fseek(f, 0, SEEK_END) || (size = ftell(f)) < 0 ||
      (unsigned long)size > CATALOG_FILE_LIMIT || fseek(f, 0, SEEK_SET)) {
    fclose(f);
    return NULL;
  }
  buf = malloc((size_t)size + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf);
    fclose(f);
    return NULL;
  }
  buf[size] = '\0';
  fclose(f);
  *len = (size_t)size;
  return buf;
}

static yyjson_val *get(yyjson_val *v, const char *key) {
  return yyjson_obj_get(v, key);
}
static const char *strval(yyjson_val *v) {
  return yyjson_is_str(v) ? yyjson_get_str(v) : NULL;
}
static bool number(yyjson_val *v, double *out) {
  if (yyjson_is_num(v)) {
    *out = yyjson_get_num(v);
    return isfinite(*out);
  }
  return false;
}

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
  for (i = 0; p->parameters && i < p->pub.parameter_count; ++i) {
    free((char *)p->parameters[i].key);
    free((char *)p->parameters[i].type);
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

static bool parse_definition(Package *p, yyjson_val *root) {
  yyjson_val *view, *params, *v;
  const char *s;
  double n;
  if (!yyjson_is_obj(root) || !(s = strval(get(root, "name"))) || !*s ||
      !(p->pub.name = nn_text_copy(s)) || !(s = strval(get(root, "kind"))) ||
      !*s || !(p->pub.kind = nn_text_copy(s)))
    return false;
  s = strval(get(root, "description"));
  if (s && !(p->pub.description = nn_text_copy(s)))
    return false;
  view = get(root, "view");
  if (!yyjson_is_obj(view))
    return false;
  s = strval(get(view, "color"));
  if (!s || !*s || !(p->pub.color = nn_text_copy(s)))
    return false;
  if (!(v = get(view, "width")) || !number(v, &n) || n <= 0)
    return false;
  p->pub.width = n;
  if (!(v = get(view, "height")) || !number(v, &n) || n <= 0)
    return false;
  p->pub.height = n;
  params = get(root, "parameters");
  if (!yyjson_is_obj(params))
    return false;
  p->pub.parameter_count = yyjson_obj_size(params);
  if (p->pub.parameter_count > CATALOG_ITEM_LIMIT)
    return false;
  p->parameters = calloc(p->pub.parameter_count ? p->pub.parameter_count : 1,
                         sizeof(*p->parameters));
  p->choice_storage = calloc(
      p->pub.parameter_count ? p->pub.parameter_count : 1, sizeof(char **));
  if (!p->parameters || !p->choice_storage)
    return false;
  p->pub.parameters = p->parameters;
  {
    yyjson_obj_iter it = yyjson_obj_iter_with(params);
    size_t i = 0;
    while ((v = yyjson_obj_iter_next(&it))) {
      yyjson_val *val = yyjson_obj_iter_get_val(v);
      NNParameterDef *d = &p->parameters[i];
      const char *key = yyjson_get_str(v);
      d->key = nn_text_copy(key);
      d->type = nn_text_copy(strval(get(val, "type")));
      if (!d->key || !*d->key || !d->type ||
          (strcmp(d->type, "boolean") && strcmp(d->type, "integer") &&
           strcmp(d->type, "number") && strcmp(d->type, "string") &&
           strcmp(d->type, "dtype") && strcmp(d->type, "json") &&
           strcmp(d->type, "stereotype")))
        return false;
      for (size_t previous = 0; previous < i; ++previous)
        if (!strcmp(p->parameters[previous].key, d->key))
          return false;
      {
        const char *position = strval(get(val, "position"));
        if (get(val, "position")) {
          if (!position ||
              (strcmp(position, "top") && strcmp(position, "bottom")))
            return false;
          d->position = nn_text_copy(position);
          if (!d->position)
            return false;
        }
      }
      if (get(val, "minimum")) {
        if (!number(get(val, "minimum"), &d->minimum) ||
            (strcmp(d->type, "integer") && strcmp(d->type, "number")))
          return false;
        d->has_minimum = true;
      }
      {
        yyjson_val *dv = get(val, "default");
        if (dv) {
          d->has_default = true;
          if (yyjson_is_bool(dv)) {
            d->default_value.type = NN_PARAMETER_BOOLEAN;
            d->default_value.as.boolean = yyjson_get_bool(dv);
          } else if (!strcmp(d->type, "number") && yyjson_is_num(dv)) {
            d->default_value.type = NN_PARAMETER_NUMBER;
            d->default_value.as.number = yyjson_get_num(dv);
          } else if (!strcmp(d->type, "integer") && yyjson_is_int(dv)) {
            d->default_value.type = NN_PARAMETER_INTEGER;
            d->default_value.as.integer = yyjson_get_sint(dv);
          } else if (yyjson_is_num(dv)) {
            d->default_value.type = NN_PARAMETER_NUMBER;
            d->default_value.as.number = yyjson_get_num(dv);
          } else if (yyjson_is_str(dv)) {
            d->default_value.type = NN_PARAMETER_STRING;
            d->default_value.as.string = nn_text_copy(yyjson_get_str(dv));
          } else if (yyjson_is_arr(dv) || yyjson_is_obj(dv)) {
            d->default_value.type = NN_PARAMETER_JSON;
            d->default_value.as.string =
                yyjson_val_write_opts(dv, 0, NULL, NULL, NULL);
          } else
            return false;
          if ((d->default_value.type == NN_PARAMETER_STRING ||
               d->default_value.type == NN_PARAMETER_JSON) &&
              !d->default_value.as.string)
            return false;
        }
      }
      {
        yyjson_val *choices = get(val, "choices");
        if (choices) {
          size_t j, count = yyjson_arr_size(choices);
          if (!yyjson_is_arr(choices) || count > CATALOG_ITEM_LIMIT ||
              (strcmp(d->type, "string") && strcmp(d->type, "dtype")))
            return false;
          p->choice_storage[i] = calloc(count + 1, sizeof(char *));
          if (!p->choice_storage[i])
            return false;
          for (j = 0; j < count; ++j) {
            const char *choice = strval(yyjson_arr_get(choices, j));
            if (!choice || !(p->choice_storage[i][j] = nn_text_copy(choice)))
              return false;
          }
          d->choices = (const char *const *)p->choice_storage[i];
          d->choice_count = count;
        }
      }
      ++i;
    }
  }
  return true;
}

static bool parse_manifest(Package *p, yyjson_val *root,
                           const char *directory) {
  yyjson_val *sv, *deps, *entry, *inf;
  const char *s;
  if (!yyjson_is_obj(root) ||
      !yyjson_is_uint(sv = get(root, "schemaVersion")) ||
      yyjson_get_uint(sv) != 1)
    return false;
  s = strval(get(root, "id"));
  if (!valid_id(s) || !(p->pub.id = nn_text_copy(s)))
    return false;
  s = strval(get(root, "version"));
  {
    unsigned long a, b, c;
    if (!semver(s, &a, &b, &c) || !(p->pub.version = nn_text_copy(s)))
      return false;
  }
  p->pub.directory = nn_text_copy(directory);
  if (!p->pub.directory)
    return false;
  deps = get(root, "dependencies");
  if (!yyjson_is_obj(deps))
    return false;
  p->pub.dependency_count = yyjson_obj_size(deps);
  if (p->pub.dependency_count > CATALOG_ITEM_LIMIT)
    return false;
  p->dependencies =
      calloc(p->pub.dependency_count ? p->pub.dependency_count : 1,
             sizeof(*p->dependencies));
  if (!p->dependencies)
    return false;
  p->pub.dependencies = p->dependencies;
  {
    yyjson_obj_iter it = yyjson_obj_iter_with(deps);
    size_t i = 0;
    yyjson_val *key;
    while ((key = yyjson_obj_iter_next(&it))) {
      const char *id = yyjson_get_str(key),
                 *constraint = strval(yyjson_obj_iter_get_val(key));
      unsigned long a, b, c;
      if (!valid_id(id) || !constraint || !*constraint ||
          !(constraint[0] == '^' ? semver(constraint + 1, &a, &b, &c)
                                 : semver(constraint, &a, &b, &c)))
        return false;
      p->dependencies[i].id = nn_text_copy(id);
      p->dependencies[i].version_constraint = nn_text_copy(constraint);
      if (!p->dependencies[i].id || !p->dependencies[i].version_constraint)
        return false;
      ++i;
    }
  }
  entry = get(root, "entrypoints");
  if (!yyjson_is_obj(entry))
    return false;
  s = strval(get(entry, "definition"));
  if (!s || !safe_rel(s))
    return false;
  inf = get(entry, "inference");
  if (!yyjson_is_obj(inf) ||
      strcmp(strval(get(inf, "language")) ? strval(get(inf, "language")) : "",
             "lua") != 0)
    return false;
  s = strval(get(inf, "file"));
  if (!s || !safe_rel(s) || !(p->pub.inference_file = nn_text_copy(s)))
    return false;
  return true;
}

static bool load_package(NNCatalog *cat, const char *root, const char *rel,
                         const char *expected_id, const char *expected_version,
                         char *err, size_t cap) {
  char dir[PATH_MAX], manifest_path[PATH_MAX], def_path[PATH_MAX],
      inference_path[PATH_MAX];
  char manifest_rel[PATH_MAX], def_rel[PATH_MAX], inf_rel[PATH_MAX];
  char *buf = NULL;
  size_t len;
  yyjson_doc *doc = NULL;
  Package item = {0};
  bool ok = false;
  if (!checked_path(root, rel, true, dir) || !under_path(root, dir)) {
    nn_errorf(err, cap, "invalid or linked package path: %s", rel);
    return false;
  }
  if (snprintf(manifest_rel, sizeof(manifest_rel), "%s/manifest.json", rel) >=
          (int)sizeof(manifest_rel) ||
      !checked_path(root, manifest_rel, false, manifest_path) ||
      !under_path(root, manifest_path)) {
    nn_errorf(err, cap, "missing or unsafe package manifest: %s", rel);
    return false;
  }
  buf = read_file(manifest_path, &len);
  if (!buf || !(doc = yyjson_read(buf, len, 0))) {
    goto done;
  }
  ok = parse_manifest(&item, yyjson_doc_get_root(doc), dir);
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
    yyjson_val *entry = get(yyjson_doc_get_root(doc), "entrypoints");
    const char *name = strval(get(entry, "definition"));
    const char *file = item.pub.inference_file;
    if (snprintf(def_rel, sizeof(def_rel), "%s/%s", rel, name) >=
            (int)sizeof(def_rel) ||
        snprintf(inf_rel, sizeof(inf_rel), "%s/%s", rel, file) >=
            (int)sizeof(inf_rel) ||
        !checked_path(root, def_rel, false, def_path) ||
        !under_path(root, def_path) ||
        !checked_path(root, inf_rel, false, inference_path) ||
        !under_path(root, inference_path)) {
      ok = false;
      nn_errorf(err, cap, "invalid package entrypoint path: %s", rel);
      goto done;
    }
  }
  {
    size_t dlen;
    char *definition = read_file(def_path, &dlen);
    yyjson_doc *ddoc = definition ? yyjson_read(definition, dlen, 0) : NULL;
    if (!ddoc || !parse_definition(&item, yyjson_doc_get_root(ddoc))) {
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

static bool version_satisfies(const char *version, const char *constraint) {
  unsigned long a, b, c, x, y, z;
  const char *wanted = constraint;
  bool caret = *wanted == '^';
  if (caret)
    ++wanted;
  if (!semver(version, &a, &b, &c) || !semver(wanted, &x, &y, &z))
    return false;
  if (!caret)
    return a == x && b == y && c == z;
  if (a != x)
    return false;
  if (a)
    return b > y || (b == y && c >= z);
  return b == y && c >= z;
}

static bool validate_dependencies(NNCatalog *cat, char *err, size_t cap) {
  size_t i, j;
  unsigned char *marks = calloc(cat->count ? cat->count : 1, 1);
  if (!marks)
    return nn_fail(err, cap, "out of memory validating dependencies");
  for (i = 0; i < cat->count; ++i)
    for (j = i + 1; j < cat->count; ++j)
      if (!strcmp(cat->items[i].pub.id, cat->items[j].pub.id)) {
        nn_errorf(err, cap, "duplicate package identity: %s",
                  cat->items[i].pub.id);
        free(marks);
        return false;
      }
  for (i = 0; i < cat->count; ++i)
    for (j = 0; j < cat->items[i].pub.dependency_count; ++j) {
      NNPackageDependency *dep = &cat->items[i].dependencies[j];
      size_t k, matches = 0;
      for (k = 0; k < cat->count; ++k)
        if (!strcmp(cat->items[k].pub.id, dep->id) &&
            version_satisfies(cat->items[k].pub.version,
                              dep->version_constraint))
          ++matches;
      if (matches != 1) {
        nn_errorf(err, cap, "dependency has no unique provider: %s", dep->id);
        free(marks);
        return false;
      }
    }
  /* Three-colour DFS, package count is bounded above. */
  for (i = 0; i < cat->count; ++i) {
    size_t stack[CATALOG_ITEM_LIMIT], edge[CATALOG_ITEM_LIMIT], depth = 0;
    if (marks[i])
      continue;
    stack[depth] = i;
    edge[depth++] = 0;
    marks[i] = 1;
    while (depth) {
      size_t at = stack[depth - 1];
      Package *p = &cat->items[at];
      if (edge[depth - 1] == p->pub.dependency_count) {
        marks[at] = 2;
        --depth;
        continue;
      }
      {
        const char *dep_id = p->dependencies[edge[depth - 1]++].id;
        size_t k;
        for (k = 0; k < cat->count && strcmp(cat->items[k].pub.id, dep_id);
             ++k) {
        }
        if (marks[k] == 1) {
          nn_errorf(err, cap, "dependency cycle at package: %s", dep_id);
          free(marks);
          return false;
        }
        if (!marks[k]) {
          marks[k] = 1;
          stack[depth] = k;
          edge[depth++] = 0;
        }
      }
    }
  }
  free(marks);
  return true;
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
    if (!custom[i].id || !valid_id(custom[i].id) ||
        !semver(custom[i].version, &a, &b, &c) || !custom[i].path ||
        !safe_rel(custom[i].path)) {
      nn_errorf(err, cap, "invalid custom package reference: %s",
                custom[i].id ? custom[i].id : "");
      goto fail;
    }
    if (!load_package(cat, project, custom[i].path, custom[i].id,
                      custom[i].version, err, cap))
      goto fail;
  }
  if (!validate_dependencies(cat, err, cap))
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
