#include "visualization_internal.h"

#include "catalog/catalog.h"
#include "model/model.h"
#include "project/project.h"
#include "utils/utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  SCENE_NODE_LIMIT = 20000,
  SCENE_EDGE_LIMIT = 60000,
  SCENE_DEPTH_LIMIT = 32
};

static char *copy_text(const char *s) { return nn_text_copy(s ? s : ""); }
static char *join_path(const char *parent, const char *child) {
  if (!parent || !*parent)
    return copy_text(child);
  size_t a = strlen(parent), b = strlen(child);
  if (a > SIZE_MAX - b - 2)
    return NULL;
  char *p = malloc(a + b + 2);
  if (!p)
    return NULL;
  memcpy(p, parent, a);
  p[a] = '/';
  memcpy(p + a + 1, child, b + 1);
  return p;
}
static const NNPackage *package_for(const NNCatalog *catalog,
                                    const NNNode *node) {
  return nn_catalog_find(catalog, node->package_id, node->package_version);
}
static bool is_subflow(const NNPackage *package) {
  return package && package->kind && strcmp(package->kind, "subflow") == 0;
}
static size_t node_index(const NN3DScene *scene, const char *path) {
  for (size_t i = 0; i < scene->node_count; i++)
    if (strcmp(scene->nodes[i].path, path) == 0)
      return i;
  return SIZE_MAX;
}
static bool reserve_nodes(NN3DScene *scene, size_t count) {
  if (count > SCENE_NODE_LIMIT - scene->node_count)
    return false;
  NN3DNode *p = realloc(scene->nodes, (scene->node_count + count) * sizeof(*p));
  if (!p && count)
    return false;
  scene->nodes = p;
  return true;
}
static bool reserve_groups(NN3DScene *scene, size_t count) {
  if (count > SCENE_NODE_LIMIT - scene->group_count)
    return false;
  NN3DGroup *p =
      realloc(scene->groups, (scene->group_count + count) * sizeof(*p));
  if (!p && count)
    return false;
  scene->groups = p;
  return true;
}
static bool reserve_edges(NN3DScene *scene, size_t count) {
  if (count > SCENE_EDGE_LIMIT - scene->edge_count)
    return false;
  NN3DEdge *p = realloc(scene->edges, (scene->edge_count + count) * sizeof(*p));
  if (!p && count)
    return false;
  scene->edges = p;
  return true;
}
typedef struct {
  size_t outputs[2];
  size_t *inputs;
  size_t input_count, input_capacity;
} SubflowMap;
static size_t find_local(const NNNode **nodes, size_t count, const char *id) {
  for (size_t i = 0; i < count; i++)
    if (!strcmp(nodes[i]->id, id))
      return i;
  return SIZE_MAX;
}
static size_t plan_node_index(const NN3DPlan *plan, const char *id) {
  for (size_t i = 0; i < plan->node_count; i++)
    if (!strcmp(plan->nodes[i].id, id))
      return i;
  return SIZE_MAX;
}
static bool add_scene_edge(NN3DScene *scene, size_t source, size_t target,
                           const char *source_handle, const char *target_handle,
                           bool loss) {
  if (source >= scene->node_count || target >= scene->node_count)
    return false;
  if (!reserve_edges(scene, 1))
    return false;
  NN3DEdge *edge = &scene->edges[scene->edge_count];
  memset(edge, 0, sizeof(*edge));
  edge->source = source;
  edge->target = target;
  edge->source_handle = copy_text(source_handle);
  edge->target_handle = copy_text(target_handle);
  edge->loss = loss;
  if (!edge->source_handle || !edge->target_handle) {
    free((char *)edge->source_handle);
    free((char *)edge->target_handle);
    return false;
  }
  scene->edge_count++;
  return true;
}
static bool subflow_add_input(SubflowMap *map, size_t index) {
  if (index == SIZE_MAX)
    return true;
  if (map->input_count == map->input_capacity) {
    size_t capacity = map->input_capacity ? map->input_capacity * 2 : 4;
    if (capacity < map->input_capacity || capacity > SCENE_NODE_LIMIT)
      return false;
    size_t *items = realloc(map->inputs, capacity * sizeof(*items));
    if (!items)
      return false;
    map->inputs = items;
    map->input_capacity = capacity;
  }
  map->inputs[map->input_count++] = index;
  return true;
}
static char *format_parameters(const NNParameter *parameters, size_t count,
                               const NNPackage *package);
static bool append_plan_default(const NNPackage *owner, NN3DPlan *plan) {
  plan->nodes = calloc(1, sizeof(*plan->nodes));
  plan->edges = calloc(1, sizeof(*plan->edges));
  plan->outputs = calloc(owner->output_count ? owner->output_count : 1,
                         sizeof(*plan->outputs));
  if (!plan->nodes || !plan->edges || (owner->output_count && !plan->outputs))
    return false;
  plan->node_count = 1;
  plan->edge_count = 1;
  plan->output_count = owner->output_count;
  plan->nodes[0].id = copy_text("body");
  plan->nodes[0].label = copy_text("Body");
  plan->nodes[0].body = true;
  plan->edges[0].source = copy_text("$input");
  plan->edges[0].source_handle = copy_text("out");
  plan->edges[0].target = copy_text("body");
  plan->edges[0].target_handle = copy_text("in");
  if (!plan->nodes[0].id || !plan->nodes[0].label || !plan->edges[0].source ||
      !plan->edges[0].source_handle || !plan->edges[0].target ||
      !plan->edges[0].target_handle)
    return false;
  for (size_t i = 0; i < owner->output_count; i++) {
    NN3DPlanOutput *out = &plan->outputs[i];
    out->id = copy_text(owner->outputs[i].id);
    out->node = copy_text("body");
    out->handle = copy_text(owner->outputs[i].id);
    if (!out->id || !out->node || !out->handle)
      return false;
  }
  return true;
}
static bool validate_plan(const NNCatalog *catalog, const NNPackage *owner,
                          NN3DPlan *plan, char *error, size_t cap) {
  if (!plan->node_count || plan->output_count != owner->output_count) {
    nn_error_set(
        error, cap,
        "Visualization recipe must define nodes and map every declared output");
    return false;
  }
  for (size_t i = 0; i < plan->node_count; i++) {
    NN3DPlanNode *node = &plan->nodes[i];
    if (node->body) {
      node->package = owner;
      continue;
    }
    node->package =
        nn_catalog_resolve(catalog, node->package_id, node->version);
    if (!node->package || is_subflow(node->package)) {
      nn_errorf(error, cap,
                "Visualization recipe has unknown or subflow package %s",
                node->package_id);
      return false;
    }
    NNParameter *effective = NULL;
    size_t effective_count = 0;
    if (!nn_catalog_parameters(catalog, node->package, node->parameters,
                               node->parameter_count, &effective,
                               &effective_count, error, cap))
      return false;
    nn_catalog_parameters_free(node->parameters, node->parameter_count);
    node->parameters = effective;
    node->parameter_count = effective_count;
  }
  for (size_t i = 0; i < plan->edge_count; i++) {
    NN3DPlanEdge *edge = &plan->edges[i];
    size_t target = plan_node_index(plan, edge->target);
    size_t source = strcmp(edge->source, "$input")
                        ? plan_node_index(plan, edge->source)
                        : SIZE_MAX;
    if (target == SIZE_MAX ||
        (strcmp(edge->source, "$input") && source == SIZE_MAX)) {
      nn_error_set(error, cap,
                   "Visualization recipe edge refers to an unknown local node");
      return false;
    }
    if (!strcmp(edge->source, "$input")) {
      if (strcmp(edge->source_handle, "out")) {
        nn_error_set(error, cap, "Visualization input anchor exposes only out");
        return false;
      }
    } else {
      NN3DPlanNode *from = &plan->nodes[source];
      bool found = false;
      const char *output_type = NULL;
      for (size_t j = 0; j < from->package->output_count; j++)
        if (!strcmp(from->package->outputs[j].id, edge->source_handle)) {
          found = true;
          output_type = from->package->outputs[j].type;
        }
      if (!found) {
        nn_errorf(error, cap, "Unknown visualization output handle %s",
                  edge->source_handle);
        return false;
      }
      if (output_type && !strcmp(output_type, "loss")) {
        nn_error_set(error, cap,
                     "Visualization loss outputs can only map to a declared "
                     "loss output");
        return false;
      }
    }
    NN3DPlanNode *to = &plan->nodes[target];
    bool target_ok = false;
    if (to->body)
      target_ok = !strcmp(edge->target_handle, "in");
    else if (to->package->kind && !strcmp(to->package->kind, "join")) {
      size_t order = 0;
      target_ok = nn_join_handle_order(edge->target_handle, &order);
    } else
      target_ok = !strcmp(edge->target_handle, "in");
    if (!target_ok) {
      nn_errorf(error, cap, "Invalid visualization input handle %s",
                edge->target_handle);
      return false;
    }
    if (to->package->kind && (!strcmp(to->package->kind, "input") ||
                              !strcmp(to->package->kind, "output") ||
                              !strcmp(to->package->kind, "loss-output"))) {
      nn_error_set(
          error, cap,
          "Visualization recipes cannot connect through boundary terminals");
      return false;
    }
    if (source != SIZE_MAX && plan->nodes[source].package->kind &&
        (!strcmp(plan->nodes[source].package->kind, "output") ||
         !strcmp(plan->nodes[source].package->kind, "loss-output"))) {
      nn_error_set(error, cap,
                   "Visualization terminal nodes cannot be edge sources");
      return false;
    }
    for (size_t j = 0; j < i; j++)
      if (!strcmp(plan->edges[j].target, edge->target) &&
          !strcmp(plan->edges[j].target_handle, edge->target_handle)) {
        nn_errorf(error, cap,
                  "Visualization input %s on %s is connected more than once",
                  edge->target_handle, edge->target);
        return false;
      }
  }
  for (size_t i = 0; i < plan->output_count; i++) {
    NN3DPlanOutput *o = &plan->outputs[i];
    size_t n = plan_node_index(plan, o->node);
    bool declared = false;
    for (size_t j = 0; j < owner->output_count; j++)
      if (!strcmp(owner->outputs[j].id, o->id))
        declared = true;
    if (n == SIZE_MAX || !declared) {
      nn_error_set(error, cap, "Visualization output mapping is unknown");
      return false;
    }
    for (size_t j = 0; j < i; j++)
      if (!strcmp(plan->outputs[j].id, o->id)) {
        nn_error_set(error, cap, "Duplicate visualization output mapping");
        return false;
      }
    NN3DPlanNode *target = &plan->nodes[n];
    bool found = false;
    const char *mapped_type = NULL;
    for (size_t j = 0; j < target->package->output_count; j++)
      if (!strcmp(target->package->outputs[j].id, o->handle)) {
        found = true;
        mapped_type = target->package->outputs[j].type;
      }
    if (!found) {
      nn_errorf(error, cap, "Unknown mapped visualization handle %s",
                o->handle);
      return false;
    }
    for (size_t j = 0; j < owner->output_count; j++)
      if (!strcmp(owner->outputs[j].id, o->id) &&
          strcmp(owner->outputs[j].type ? owner->outputs[j].type : "",
                 mapped_type ? mapped_type : "")) {
        nn_errorf(error, cap,
                  "Visualization output %s maps to an incompatible type",
                  o->id);
        return false;
      }
  }
  /* Any acyclic recipe has a topological ordering within at most node_count
   * passes. */
  size_t *rank = calloc(plan->node_count, sizeof(*rank));
  if (!rank)
    return nn_fail(error, cap, "Out of memory validating visualization recipe");
  for (size_t pass = 0; pass < plan->node_count; pass++) {
    bool changed = false;
    for (size_t e = 0; e < plan->edge_count; e++) {
      size_t from = plan_node_index(plan, plan->edges[e].source),
             to = plan_node_index(plan, plan->edges[e].target);
      if (from != SIZE_MAX && to != SIZE_MAX && rank[to] <= rank[from]) {
        rank[to] = rank[from] + 1;
        changed = true;
      }
    }
    if (!changed)
      break;
    if (pass + 1 == plan->node_count) {
      free(rank);
      nn_error_set(error, cap, "Visualization recipe contains a cycle");
      return false;
    }
  }
  free(rank);
  return true;
}
static bool validate_subflow_boundaries(const NNNode **nodes,
                                        const NNPackage **packages,
                                        size_t count, const NNPackage *owner,
                                        char *error, size_t cap) {
  size_t inputs = 0, terminals = 0;
  for (size_t i = 0; i < count; i++) {
    const NNPackage *package = packages[i];
    const char *kind = package->kind ? package->kind : "";
    if (!strcmp(kind, "input")) {
      inputs++;
      if (nodes[i]->boundary_handle_id && *nodes[i]->boundary_handle_id) {
        nn_errorf(error, cap, "Input %s cannot map a subflow output",
                  nodes[i]->id);
        return false;
      }
      continue;
    }
    if (strcmp(kind, "output") && strcmp(kind, "loss-output")) {
      if (nodes[i]->boundary_handle_id && *nodes[i]->boundary_handle_id) {
        nn_errorf(error, cap,
                  "Node %s has a boundary mapping but is not a terminal",
                  nodes[i]->id);
        return false;
      }
      continue;
    }
    terminals++;
    if (!nodes[i]->boundary_handle_id || !*nodes[i]->boundary_handle_id) {
      nn_errorf(error, cap, "Terminal %s has no subflow output mapping",
                nodes[i]->id);
      return false;
    }
    size_t output = SIZE_MAX;
    for (size_t j = 0; j < owner->output_count; j++)
      if (!strcmp(owner->outputs[j].id, nodes[i]->boundary_handle_id))
        output = j;
    if (output == SIZE_MAX) {
      nn_errorf(error, cap, "Terminal %s maps an unknown output", nodes[i]->id);
      return false;
    }
    const char *expected =
        !strcmp(owner->outputs[output].type, "loss") ? "loss-output" : "output";
    if (strcmp(kind, expected)) {
      nn_errorf(error, cap, "Terminal %s has the wrong output type",
                nodes[i]->id);
      return false;
    }
    for (size_t j = 0; j < i; j++)
      if (nodes[j]->boundary_handle_id &&
          !strcmp(nodes[j]->boundary_handle_id, nodes[i]->boundary_handle_id)) {
        nn_errorf(error, cap, "Subflow output %s is mapped more than once",
                  nodes[i]->boundary_handle_id);
        return false;
      }
  }
  if (inputs != 1) {
    nn_errorf(error, cap, "Subflow %s must contain exactly one Input",
              owner->id);
    return false;
  }
  if (terminals != owner->output_count) {
    nn_errorf(error, cap, "Subflow %s is missing mapped output terminals",
              owner->id);
    return false;
  }
  return true;
}
static bool add_synthetic_node(NN3DScene *scene, const NN3DPlanNode *plan_node,
                               const char *path, const char *owner_id,
                               size_t group, size_t depth, double x, double y) {
  if (!reserve_nodes(scene, 1))
    return false;
  NN3DNode *node = &scene->nodes[scene->node_count];
  memset(node, 0, sizeof(*node));
  node->path = copy_text(path);
  node->source_id = copy_text(owner_id);
  node->label = copy_text(plan_node->label);
  node->package_id = copy_text(plan_node->package->id);
  node->color = copy_text(plan_node->package->color);
  node->group = group;
  node->center = (NN3DVec){x, y, (double)depth * 150.0};
  node->size = (NN3DVec){
      plan_node->package->width > 0 ? plan_node->package->width : 100.0,
      plan_node->package->height > 0 ? plan_node->package->height : 70.0, 22.0};
  node->parameters = format_parameters(
      plan_node->parameters, plan_node->parameter_count, plan_node->package);
  if (!node->path || !node->source_id || !node->label || !node->package_id ||
      !node->color || !node->parameters) {
    free((char *)node->path);
    free((char *)node->source_id);
    free((char *)node->label);
    free((char *)node->package_id);
    free((char *)node->color);
    free((char *)node->parameters);
    return false;
  }
  scene->node_count++;
  return true;
}
static bool append_text(char **text, size_t *length, const char *value) {
  size_t add = strlen(value), maximum = 1024u * 1024u;
  if (add > maximum - *length - 1)
    return false;
  char *grown = realloc(*text, *length + add + 1);
  if (!grown)
    return false;
  memcpy(grown + *length, value, add + 1);
  *length += add;
  *text = grown;
  return true;
}
static const NNParameterDef *parameter_definition(const NNPackage *package,
                                                  const char *key) {
  for (size_t i = 0; i < package->parameter_count; i++)
    if (!strcmp(package->parameters[i].key, key))
      return &package->parameters[i];
  return NULL;
}
static char *format_parameters(const NNParameter *parameters, size_t count,
                               const NNPackage *package) {
  char *text = calloc(1, 1);
  size_t length = 0;
  bool first = true;
  if (!text)
    return NULL;
  for (size_t i = 0; i < count; i++) {
    const NNParameterDef *definition =
        parameter_definition(package, parameters[i].key);
    if (!definition || !definition->position)
      continue;
    const NNValue *value = &parameters[i].value;
    char number[64];
    const char *shown = NULL;
    switch (value->type) {
    case NN_VALUE_BOOL:
      shown = value->as.boolean ? "true" : "false";
      break;
    case NN_VALUE_INT:
      snprintf(number, sizeof(number), "%lld", value->as.integer);
      shown = number;
      break;
    case NN_VALUE_REAL:
      snprintf(number, sizeof(number), "%.6g", value->as.real);
      shown = number;
      break;
    case NN_VALUE_STRING:
      shown = value->as.string ? value->as.string : "";
      break;
    case NN_VALUE_OBJECT: {
      const char *id = NULL, *version = NULL;
      for (size_t j = 0; j < value->as.object.count; j++) {
        const NNParameter *entry = &value->as.object.items[j];
        if (!strcmp(entry->key, "id") && entry->value.type == NN_VALUE_STRING)
          id = entry->value.as.string;
        if (!strcmp(entry->key, "version") &&
            entry->value.type == NN_VALUE_STRING)
          version = entry->value.as.string;
      }
      if (id && version) {
        if (snprintf(number, sizeof(number), "%s@%s", id, version) >=
            (int)sizeof(number))
          shown = "reference";
        else
          shown = number;
      } else
        shown = "object";
      break;
    }
    case NN_VALUE_ARRAY:
      shown = "array";
      break;
    default:
      shown = "value";
      break;
    }
    if ((!first && !append_text(&text, &length, "\n")) ||
        !append_text(&text, &length, parameters[i].key) ||
        !append_text(&text, &length, "=") ||
        !append_text(&text, &length, shown)) {
      free(text);
      return NULL;
    }
    first = false;
  }
  return text;
}
static bool add_group(NN3DScene *scene, const char *path, const NNNode *owner,
                      size_t parent, size_t *index) {
  if (!reserve_groups(scene, 1))
    return false;
  NN3DGroup *g = &scene->groups[scene->group_count];
  memset(g, 0, sizeof(*g));
  g->path = copy_text(path);
  g->source_id = copy_text(owner->id);
  g->label = copy_text(owner->label);
  g->parent = parent;
  g->minimum = (NN3DVec){0, 0, 0};
  g->maximum = (NN3DVec){0, 0, 0};
  if (!g->path || !g->source_id || !g->label) {
    free((char *)g->path);
    free((char *)g->source_id);
    free((char *)g->label);
    return false;
  }
  *index = scene->group_count++;
  return true;
}
static bool add_model_node(NN3DScene *scene, const NNNode *node,
                           const NNPackage *package, const char *path,
                           size_t group, double x, double y, double z) {
  if (!reserve_nodes(scene, 1))
    return false;
  NN3DNode *n = &scene->nodes[scene->node_count];
  memset(n, 0, sizeof(*n));
  n->path = copy_text(path);
  n->source_id = copy_text(node->id);
  n->label = copy_text(node->label);
  n->package_id = copy_text(node->package_id);
  n->color = copy_text(package->color);
  n->parameters =
      format_parameters(node->parameters, node->parameter_count, package);
  n->group = group;
  n->center = (NN3DVec){x, y, z};
  double w =
      package->width > 0 && isfinite(package->width) ? package->width : 100.0;
  double h =
      package->height > 0 && isfinite(package->height) ? package->height : 70.0;
  n->size = (NN3DVec){w, h, 22.0};
  if (!n->path || !n->source_id || !n->label || !n->package_id || !n->color ||
      !n->parameters) {
    free((char *)n->path);
    free((char *)n->source_id);
    free((char *)n->label);
    free((char *)n->package_id);
    free((char *)n->color);
    free((char *)n->parameters);
    return false;
  }
  scene->node_count++;
  return true;
}
typedef struct {
  double min_x, max_x, min_y, max_y;
  bool present;
} SceneBounds;
static bool path_in_subtree(const char *path, const char *root) {
  size_t n = strlen(root);
  return !strncmp(path, root, n) && (path[n] == '\0' || path[n] == '/');
}
static SceneBounds subtree_bounds(const NN3DScene *scene, const char *path) {
  SceneBounds bounds = {0};
  for (size_t i = 0; i < scene->node_count; i++) {
    const NN3DNode *node = &scene->nodes[i];
    if (!path_in_subtree(node->path, path))
      continue;
    double min_x = node->center.x - node->size.x * 0.5,
           max_x = node->center.x + node->size.x * 0.5;
    double min_y = node->center.y - node->size.y * 0.5,
           max_y = node->center.y + node->size.y * 0.5;
    if (!bounds.present) {
      bounds = (SceneBounds){min_x, max_x, min_y, max_y, true};
    } else {
      bounds.min_x = fmin(bounds.min_x, min_x);
      bounds.max_x = fmax(bounds.max_x, max_x);
      bounds.min_y = fmin(bounds.min_y, min_y);
      bounds.max_y = fmax(bounds.max_y, max_y);
    }
  }
  return bounds;
}
static void move_subtree(NN3DScene *scene, const char *path, double dx,
                         double dy) {
  for (size_t i = 0; i < scene->node_count; i++)
    if (path_in_subtree(scene->nodes[i].path, path)) {
      scene->nodes[i].center.x += dx;
      scene->nodes[i].center.y += dy;
    }
}
static bool pack_paths(NN3DScene *scene, char *const *paths,
                       const size_t *ranks, size_t count, double origin_x,
                       double origin_y) {
  size_t max_rank = 0;
  for (size_t i = 0; i < count; i++)
    if (ranks[i] > max_rank)
      max_rank = ranks[i];
  if (count && max_rank > count)
    return false;
  size_t columns = count ? max_rank + 1 : 0;
  double *widths = calloc(columns ? columns : 1, sizeof(*widths));
  double *column_x = calloc(columns ? columns : 1, sizeof(*column_x));
  double *lanes = calloc(columns ? columns : 1, sizeof(*lanes));
  SceneBounds *bounds = calloc(count ? count : 1, sizeof(*bounds));
  if (!widths || !column_x || !lanes || !bounds) {
    free(widths);
    free(column_x);
    free(lanes);
    free(bounds);
    return false;
  }
  for (size_t i = 0; i < count; i++) {
    bounds[i] = subtree_bounds(scene, paths[i]);
    if (bounds[i].present)
      widths[ranks[i]] =
          fmax(widths[ranks[i]], bounds[i].max_x - bounds[i].min_x);
  }
  double x = origin_x;
  for (size_t rank = 0; rank < columns; rank++) {
    column_x[rank] = x;
    x += fmax(160.0, widths[rank]) + 140.0;
    lanes[rank] = origin_y;
  }
  for (size_t i = 0; i < count; i++)
    if (bounds[i].present) {
      size_t rank = ranks[i];
      move_subtree(scene, paths[i], column_x[rank] - bounds[i].min_x,
                   lanes[rank] - bounds[i].min_y);
      lanes[rank] += bounds[i].max_y - bounds[i].min_y + 120.0;
    }
  free(widths);
  free(column_x);
  free(lanes);
  free(bounds);
  return true;
}
static bool layout_scope(NN3DScene *scene, const NNModel *model,
                         const NNCatalog *catalog, const char *scope,
                         const char *prefix, size_t group, size_t depth,
                         double x_offset, double y_offset, double z_offset,
                         NN3DScopeBoundary *boundaries, char *error,
                         size_t cap) {
  size_t count = 0;
  size_t model_count = nn_model_node_count(model);
  for (size_t i = 0; i < model_count; i++) {
    const NNNode *n = nn_model_node_at(model, i);
    if (strcmp(n->scope_id ? n->scope_id : "", scope ? scope : "") == 0)
      count++;
  }
  if (count > SCENE_NODE_LIMIT - scene->node_count) {
    nn_error_set(error, cap, "3D scene node limit exceeded");
    return false;
  }
  const NNNode **nodes = count ? malloc(count * sizeof(*nodes)) : NULL;
  const NNPackage **packages = count ? malloc(count * sizeof(*packages)) : NULL;
  size_t *rank = count ? calloc(count, sizeof(*rank)) : NULL;
  size_t *indices = count ? malloc(count * sizeof(*indices)) : NULL;
  SubflowMap *maps = count ? calloc(count, sizeof(*maps)) : NULL;
  char **paths = count ? calloc(count, sizeof(*paths)) : NULL;
  if (count && (!nodes || !packages || !rank || !indices || !maps || !paths)) {
    free(nodes);
    free(packages);
    free(rank);
    free(indices);
    free(maps);
    free(paths);
    nn_error_set(error, cap, "Out of memory building 3D scene");
    return false;
  }
  for (size_t i = 0; i < count; i++) {
    indices[i] = SIZE_MAX;
    maps[i].outputs[0] = maps[i].outputs[1] = SIZE_MAX;
  }
  size_t used = 0;
  for (size_t i = 0; i < model_count; i++) {
    const NNNode *n = nn_model_node_at(model, i);
    if (strcmp(n->scope_id ? n->scope_id : "", scope ? scope : ""))
      continue;
    const NNPackage *p = package_for(catalog, n);
    if (!p) {
      nn_errorf(error, cap, "Unknown package %s@%s in 3D scene", n->package_id,
                n->package_version);
      goto fail;
    }
    nodes[used] = n;
    packages[used] = p;
    indices[used] = SIZE_MAX;
    used++;
  }
  for (size_t i = 0; i < count; i++) {
    paths[i] = join_path(prefix, nodes[i]->id);
    if (!paths[i])
      goto memory_fail;
  }
  if (scope && *scope) {
    const NNNode *owner_node = nn_model_find_node(model, scope);
    const NNPackage *owner_package =
        owner_node ? package_for(catalog, owner_node) : NULL;
    if (!owner_node || !is_subflow(owner_package)) {
      nn_errorf(error, cap, "Orphan or malformed node scope %s", scope);
      goto fail;
    }
    if (!validate_subflow_boundaries(nodes, packages, count, owner_package,
                                     error, cap))
      goto fail;
  } else {
    for (size_t i = 0; i < model_count; i++) {
      const NNNode *candidate = nn_model_node_at(model, i);
      if (!candidate->scope_id || !*candidate->scope_id)
        continue;
      const NNNode *owner_node = nn_model_find_node(model, candidate->scope_id);
      const NNPackage *owner_package =
          owner_node ? package_for(catalog, owner_node) : NULL;
      if (!owner_node || !is_subflow(owner_package)) {
        nn_errorf(error, cap, "Orphan or malformed node scope %s",
                  candidate->scope_id);
        goto fail;
      }
    }
    for (size_t i = 0; i < count; i++)
      if ((!strcmp(packages[i]->kind ? packages[i]->kind : "", "output") ||
           !strcmp(packages[i]->kind ? packages[i]->kind : "",
                   "loss-output")) &&
          nodes[i]->boundary_handle_id) {
        nn_errorf(error, cap, "Root terminal %s cannot have a subflow mapping",
                  nodes[i]->id);
        goto fail;
      }
  }
  /* A rank relaxation gives deterministic left-to-right DAG columns. */
  size_t edge_count = nn_model_edge_count(model);
  for (size_t pass = 0; pass < count; pass++) {
    bool changed = false;
    for (size_t e = 0; e < edge_count; e++) {
      const NNEdge *edge = nn_model_edge_at(model, e);
      if (strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : ""))
        continue;
      size_t a = SIZE_MAX, b = SIZE_MAX;
      for (size_t i = 0; i < count; i++) {
        if (!strcmp(nodes[i]->id, edge->source_id))
          a = i;
        if (!strcmp(nodes[i]->id, edge->target_id))
          b = i;
      }
      if (a != SIZE_MAX && b != SIZE_MAX && rank[b] <= rank[a]) {
        rank[b] = rank[a] + 1;
        changed = true;
      }
    }
    if (!changed)
      break;
    if (pass + 1 == count) {
      nn_error_set(error, cap, "Cycle found while laying out 3D scene");
      goto fail;
    }
  }
  for (size_t i = 0; i < count; i++) {
    if (is_subflow(packages[i]))
      continue;
    if (!add_model_node(scene, nodes[i], packages[i], paths[i], group,
                        x_offset + (double)rank[i] * 190.0,
                        y_offset + (double)i * 112.0,
                        z_offset + (double)depth * 150.0)) {
      nn_error_set(error, cap, "Out of memory expanding 3D scene");
      goto fail;
    }
    indices[i] = node_index(scene, paths[i]);
  }
  for (size_t i = 0; i < count; i++)
    if (is_subflow(packages[i])) {
      if (depth >= SCENE_DEPTH_LIMIT) {
        nn_error_set(error, cap, "3D subflow depth limit exceeded");
        goto fail;
      }
      char *path = paths[i];
      size_t child_group = group;
      if (!add_group(scene, path, nodes[i], group, &child_group))
        goto memory_fail;
      NN3DPlan plan = {0};
      if (!nn_3d_plan_load(catalog, packages[i], nodes[i], &plan, error, cap))
        goto fail;
      if (!plan.node_count && !append_plan_default(packages[i], &plan)) {
        nn_3d_plan_dispose(&plan);
        goto memory_fail;
      }
      if (!validate_plan(catalog, packages[i], &plan, error, cap)) {
        nn_3d_plan_dispose(&plan);
        goto fail;
      }
      size_t recipe_count = plan.node_count;
      size_t *recipe_indices = calloc(plan.node_count, sizeof(*recipe_indices));
      size_t (*recipe_outputs)[2] =
          calloc(plan.node_count, sizeof(*recipe_outputs));
      size_t *recipe_entries = calloc(plan.node_count, sizeof(*recipe_entries));
      char **recipe_paths = calloc(plan.node_count, sizeof(*recipe_paths));
      size_t *recipe_ranks = calloc(plan.node_count, sizeof(*recipe_ranks));
      if (!recipe_indices || !recipe_outputs || !recipe_entries ||
          !recipe_paths || !recipe_ranks) {
        free(recipe_indices);
        free(recipe_outputs);
        free(recipe_entries);
        free(recipe_paths);
        free(recipe_ranks);
        nn_3d_plan_dispose(&plan);
        goto memory_fail;
      }
      for (size_t r = 0; r < plan.node_count; r++) {
        recipe_indices[r] = recipe_entries[r] = SIZE_MAX;
        recipe_outputs[r][0] = recipe_outputs[r][1] = SIZE_MAX;
        NN3DPlanNode *recipe = &plan.nodes[r];
        recipe_paths[r] = join_path(path, recipe->id);
        if (!recipe_paths[r])
          goto recipe_memory_fail;
        if (recipe->body) {
          size_t body_group = SIZE_MAX;
          size_t owner_len =
              strlen(nodes[i]->label ? nodes[i]->label : nodes[i]->id);
          size_t instance_len = strlen(recipe->label);
          if (instance_len > SIZE_MAX - 5 ||
              owner_len > SIZE_MAX - instance_len - 5) {
            nn_error_set(error, cap, "3D body group label is too large");
            goto recipe_failure;
          }
          char *group_label = malloc(owner_len + instance_len + 5);
          if (!group_label) {
            nn_error_set(error, cap, "Out of memory labeling 3D body group");
            goto recipe_failure;
          }
          snprintf(group_label, owner_len + instance_len + 5, "%s · %s",
                   nodes[i]->label ? nodes[i]->label : nodes[i]->id,
                   recipe->label);
          NNNode owner_copy = *nodes[i];
          owner_copy.label = group_label;
          if (scene->body_instances >= 4096) {
            free(group_label);
            nn_error_set(error, cap, "3D body instance limit exceeded");
            goto recipe_failure;
          }
          scene->body_instances++;
          bool group_ok = add_group(scene, recipe_paths[r], &owner_copy,
                                    child_group, &body_group);
          free(group_label);
          if (!group_ok) {
            nn_error_set(error, cap, "Out of memory creating 3D body group");
            goto recipe_failure;
          }
          NN3DScopeBoundary body_boundary = {.input = SIZE_MAX,
                                             .outputs = {SIZE_MAX, SIZE_MAX}};
          if (!layout_scope(scene, model, catalog, nodes[i]->id,
                            recipe_paths[r], body_group, depth + 1,
                            x_offset + (double)r * 300.0, y_offset, z_offset,
                            &body_boundary, error, cap))
            goto recipe_failure;
          recipe_indices[r] = body_boundary.input;
          recipe_entries[r] = body_boundary.input;
          for (size_t out = 0; out < packages[i]->output_count && out < 2;
               out++)
            recipe_outputs[r][out] = body_boundary.outputs[out];
        } else {
          if (!add_synthetic_node(scene, recipe, recipe_paths[r], nodes[i]->id,
                                  child_group, depth + 1,
                                  x_offset + (double)r * 300.0,
                                  y_offset + (double)r * 80.0)) {
            goto recipe_memory_fail;
          }
          recipe_indices[r] = node_index(scene, recipe_paths[r]);
          recipe_entries[r] = recipe_indices[r];
          recipe_outputs[r][0] = recipe_outputs[r][1] = recipe_indices[r];
        }
      }
      for (size_t o = 0; o < plan.output_count; o++) {
        size_t r = plan_node_index(&plan, plan.outputs[o].node),
               owner_out = SIZE_MAX, inner_out = SIZE_MAX;
        if (r == SIZE_MAX)
          continue;
        for (size_t h = 0; h < packages[i]->output_count && h < 2; h++) {
          if (!strcmp(packages[i]->outputs[h].id, plan.outputs[o].id))
            owner_out = h;
          if (!strcmp(packages[i]->outputs[h].id, plan.outputs[o].handle))
            inner_out = h;
        }
        if (owner_out != SIZE_MAX)
          maps[i].outputs[owner_out] =
              plan.nodes[r].body
                  ? recipe_outputs[r][inner_out == SIZE_MAX ? owner_out
                                                            : inner_out]
                  : recipe_indices[r];
      }
      for (size_t e = 0; e < plan.edge_count; e++) {
        NN3DPlanEdge *edge = &plan.edges[e];
        size_t target = plan_node_index(&plan, edge->target);
        if (target == SIZE_MAX)
          continue;
        if (!strcmp(edge->source, "$input")) {
          if (!subflow_add_input(&maps[i], recipe_entries[target]))
            goto recipe_memory_fail;
          continue;
        }
        size_t source = plan_node_index(&plan, edge->source);
        if (source == SIZE_MAX)
          continue;
        size_t out = SIZE_MAX;
        for (size_t h = 0;
             h < plan.nodes[source].package->output_count && h < 2; h++)
          if (!strcmp(plan.nodes[source].package->outputs[h].id,
                      edge->source_handle))
            out = h;
        size_t from = plan.nodes[source].body
                          ? recipe_outputs[source][out == SIZE_MAX ? 0 : out]
                          : recipe_indices[source];
        bool loss =
            out != SIZE_MAX && plan.nodes[source].package->outputs[out].type &&
            !strcmp(plan.nodes[source].package->outputs[out].type, "loss");
        if (from != SIZE_MAX && recipe_entries[target] != SIZE_MAX &&
            !add_scene_edge(scene, from, recipe_entries[target],
                            edge->source_handle, edge->target_handle, loss))
          goto recipe_memory_fail;
      }
      for (size_t pass = 0; pass < plan.node_count; pass++) {
        bool changed = false;
        for (size_t e = 0; e < plan.edge_count; e++) {
          size_t from = plan_node_index(&plan, plan.edges[e].source),
                 to = plan_node_index(&plan, plan.edges[e].target);
          if (from != SIZE_MAX && to != SIZE_MAX &&
              recipe_ranks[to] <= recipe_ranks[from]) {
            recipe_ranks[to] = recipe_ranks[from] + 1;
            changed = true;
          }
        }
        if (!changed)
          break;
      }
      if (!pack_paths(scene, recipe_paths, recipe_ranks, plan.node_count,
                      x_offset, y_offset))
        goto recipe_memory_fail;
      for (size_t r = 0; r < plan.node_count; r++)
        if (recipe_indices[r] != SIZE_MAX && indices[i] == SIZE_MAX)
          indices[i] = recipe_indices[r];
      nn_3d_plan_dispose(&plan);
      free(recipe_indices);
      free(recipe_outputs);
      free(recipe_entries);
      for (size_t r = 0; r < recipe_count; r++)
        free(recipe_paths[r]);
      free(recipe_paths);
      free(recipe_ranks);
      continue;
    recipe_memory_fail:
      free(recipe_indices);
      free(recipe_outputs);
      free(recipe_entries);
      nn_3d_plan_dispose(&plan);
      for (size_t r = 0; r < recipe_count; r++)
        free(recipe_paths[r]);
      free(recipe_paths);
      free(recipe_ranks);
      goto memory_fail;
    recipe_failure:
      free(recipe_indices);
      free(recipe_outputs);
      free(recipe_entries);
      nn_3d_plan_dispose(&plan);
      for (size_t r = 0; r < recipe_count; r++)
        free(recipe_paths[r]);
      free(recipe_paths);
      free(recipe_ranks);
      goto fail;
    }
  for (size_t e = 0; e < edge_count; e++) {
    const NNEdge *edge = nn_model_edge_at(model, e);
    if (strcmp(edge->scope_id ? edge->scope_id : "", scope ? scope : ""))
      continue;
    size_t source = find_local(nodes, count, edge->source_id),
           target = find_local(nodes, count, edge->target_id);
    if (source == SIZE_MAX || target == SIZE_MAX) {
      nn_errorf(error, cap, "3D edge %s -> %s has endpoints outside scope %s",
                edge->source_id, edge->target_id, scope ? scope : "root");
      goto fail;
    }
    bool loss = false;
    for (size_t h = 0; h < packages[source]->output_count; h++)
      if (!strcmp(packages[source]->outputs[h].id, edge->source_handle_id))
        loss = packages[source]->outputs[h].type &&
               !strcmp(packages[source]->outputs[h].type, "loss");
    size_t source_node = indices[source];
    if (is_subflow(packages[source])) {
      size_t output = SIZE_MAX;
      for (size_t h = 0; h < packages[source]->output_count && h < 2; h++)
        if (!strcmp(packages[source]->outputs[h].id, edge->source_handle_id))
          output = h;
      source_node =
          output == SIZE_MAX ? SIZE_MAX : maps[source].outputs[output];
    }
    if (source_node == SIZE_MAX ||
        (!is_subflow(packages[target]) && indices[target] == SIZE_MAX)) {
      nn_errorf(error, cap,
                "3D edge %s -> %s cannot map its source or target handle",
                edge->source_id, edge->target_id);
      goto fail;
    }
    if (is_subflow(packages[target])) {
      for (size_t t = 0; t < maps[target].input_count; t++)
        if (source_node != SIZE_MAX &&
            !add_scene_edge(scene, source_node, maps[target].inputs[t],
                            edge->source_handle_id, edge->target_handle_id,
                            loss))
          goto memory_fail;
    } else if (source_node != SIZE_MAX &&
               !add_scene_edge(scene, source_node, indices[target],
                               edge->source_handle_id, edge->target_handle_id,
                               loss))
      goto memory_fail;
  }
  if (!pack_paths(scene, paths, rank, count, x_offset, y_offset))
    goto memory_fail;
  if (boundaries) {
    boundaries->input = SIZE_MAX;
    boundaries->outputs[0] = boundaries->outputs[1] = SIZE_MAX;
    boundaries->output_count = 0;
    for (size_t i = 0; i < count; i++) {
      const char *kind = packages[i]->kind ? packages[i]->kind : "";
      if (!strcmp(kind, "input"))
        boundaries->input = indices[i];
      if (!strcmp(kind, "output") || !strcmp(kind, "loss-output")) {
        const NNNode *owner_node =
            scope && *scope ? nn_model_find_node(model, scope) : NULL;
        const NNPackage *owner_package =
            owner_node ? package_for(catalog, owner_node) : NULL;
        if (owner_package && nodes[i]->boundary_handle_id) {
          for (size_t h = 0; h < owner_package->output_count && h < 2; h++)
            if (!strcmp(owner_package->outputs[h].id,
                        nodes[i]->boundary_handle_id)) {
              boundaries->outputs[h] = indices[i];
              boundaries->output_count = owner_package->output_count;
            }
        }
      }
    }
  }
  for (size_t i = 0; i < count; i++)
    free(maps[i].inputs);
  for (size_t i = 0; i < count; i++)
    free(paths[i]);
  free(paths);
  free(maps);
  free(nodes);
  free(packages);
  free(rank);
  free(indices);
  return true;
memory_fail:
  nn_error_set(error, cap, "Out of memory expanding 3D scene");
fail:
  for (size_t i = 0; i < count; i++)
    free(maps[i].inputs);
  for (size_t i = 0; i < count; i++)
    free(paths[i]);
  free(paths);
  free(maps);
  free(nodes);
  free(packages);
  free(rank);
  free(indices);
  return false;
}

void nn_3d_free(NN3DScene *scene) {
  if (!scene)
    return;
  for (size_t i = 0; i < scene->node_count; i++) {
    NN3DNode *n = &scene->nodes[i];
    free((char *)n->path);
    free((char *)n->source_id);
    free((char *)n->label);
    free((char *)n->parameters);
    free((char *)n->package_id);
    free((char *)n->color);
  }
  for (size_t i = 0; i < scene->edge_count; i++) {
    free((char *)scene->edges[i].source_handle);
    free((char *)scene->edges[i].target_handle);
  }
  for (size_t i = 0; i < scene->group_count; i++) {
    NN3DGroup *g = &scene->groups[i];
    free((char *)g->path);
    free((char *)g->source_id);
    free((char *)g->label);
  }
  free(scene->nodes);
  free(scene->edges);
  free(scene->groups);
  free(scene);
}
size_t nn_3d_node_count(const NN3DScene *s) { return s ? s->node_count : 0; }
size_t nn_3d_edge_count(const NN3DScene *s) { return s ? s->edge_count : 0; }
size_t nn_3d_group_count(const NN3DScene *s) { return s ? s->group_count : 0; }
const NN3DNode *nn_3d_node_at(const NN3DScene *s, size_t i) {
  return s && i < s->node_count ? &s->nodes[i] : NULL;
}
const NN3DEdge *nn_3d_edge_at(const NN3DScene *s, size_t i) {
  return s && i < s->edge_count ? &s->edges[i] : NULL;
}
const NN3DGroup *nn_3d_group_at(const NN3DScene *s, size_t i) {
  return s && i < s->group_count ? &s->groups[i] : NULL;
}

NN3DScene *nn_3d_build(const NNProject *project, char *error, size_t cap) {
  if (error && cap)
    error[0] = '\0';
  if (!project) {
    nn_error_set(error, cap, "No project is open");
    return NULL;
  }
  const NNModel *model = nn_project_model((NNProject *)project);
  const NNCatalog *catalog = nn_project_catalog(project);
  if (!model || !catalog) {
    nn_error_set(error, cap, "Project has no model or package catalog");
    return NULL;
  }
  NN3DScene *scene = calloc(1, sizeof(*scene));
  if (!scene) {
    nn_error_set(error, cap, "Out of memory building 3D scene");
    return NULL;
  }
  if (!layout_scope(scene, model, catalog, "", "", SIZE_MAX, 0, 0, 0, 0, NULL,
                    error, cap)) {
    nn_3d_free(scene);
    return NULL;
  }
  bool *has_bounds = scene->group_count
                         ? calloc(scene->group_count, sizeof(*has_bounds))
                         : NULL;
  if (scene->group_count && !has_bounds) {
    nn_3d_free(scene);
    nn_error_set(error, cap, "Out of memory measuring 3D groups");
    return NULL;
  }
  for (size_t i = 0; i < scene->node_count; i++) {
    const NN3DNode *node = &scene->nodes[i];
    NN3DVec half = {node->size.x * 0.5, node->size.y * 0.5, node->size.z * 0.5};
    NN3DVec low = {node->center.x - half.x, node->center.y - half.y,
                   node->center.z - half.z};
    NN3DVec high = {node->center.x + half.x, node->center.y + half.y,
                    node->center.z + half.z};
    size_t group = node->group;
    while (group != SIZE_MAX && group < scene->group_count) {
      NN3DGroup *current = &scene->groups[group];
      if (!has_bounds[group]) {
        current->minimum = low;
        current->maximum = high;
        has_bounds[group] = true;
      } else {
        current->minimum.x = fmin(current->minimum.x, low.x);
        current->minimum.y = fmin(current->minimum.y, low.y);
        current->minimum.z = fmin(current->minimum.z, low.z);
        current->maximum.x = fmax(current->maximum.x, high.x);
        current->maximum.y = fmax(current->maximum.y, high.y);
        current->maximum.z = fmax(current->maximum.z, high.z);
      }
      group = current->parent;
    }
  }
  free(has_bounds);
  return scene;
}
