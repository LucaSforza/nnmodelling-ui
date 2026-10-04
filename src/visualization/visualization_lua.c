#define _POSIX_C_SOURCE 200809L
#include "visualization_internal.h"

#include "utils/utils.h"

#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif

enum {
  LUA_SOURCE_LIMIT = 1024 * 1024,
  LUA_MEMORY_LIMIT = 8 * 1024 * 1024,
  LUA_INSTRUCTION_LIMIT = 1000000,
  PLAN_NODE_LIMIT = 4096,
  PLAN_EDGE_LIMIT = 60000,
  VALUE_ITEM_LIMIT = 8192,
  VALUE_DEPTH_LIMIT = 64
};

typedef struct {
  size_t used;
  int remaining;
  const char *source;
  size_t source_length;
  const NNNode *owner_node;
  NN3DPlan *plan;
} LuaBudget;
static void *budget_alloc(void *user, void *ptr, size_t old_size,
                          size_t new_size) {
  LuaBudget *budget = user;
  if (!new_size) {
    if (ptr)
      budget->used -= old_size;
    free(ptr);
    return NULL;
  }
  size_t old = ptr ? old_size : 0;
  if (budget->used > LUA_MEMORY_LIMIT ||
      (new_size > old && new_size - old > LUA_MEMORY_LIMIT - budget->used))
    return NULL;
  void *result = realloc(ptr, new_size);
  if (result)
    budget->used = budget->used - old + new_size;
  return result;
}
static void instruction_hook(lua_State *state, lua_Debug *debug) {
  (void)debug;
  LuaBudget *budget = *(LuaBudget **)lua_getextraspace(state);
  budget->remaining -= 1000;
  if (budget->remaining <= 0)
    luaL_error(state, "visualization instruction limit exceeded");
}
static char *lua_string(lua_State *state, int index) {
  if (lua_type(state, index) != LUA_TSTRING)
    return NULL;
  size_t length = 0;
  const char *value = lua_tolstring(state, index, &length);
  if (!value || memchr(value, '\0', length))
    return NULL;
  char *copy = malloc(length + 1);
  if (copy) {
    memcpy(copy, value, length);
    copy[length] = '\0';
  }
  return copy;
}
static bool dense_sequence(lua_State *state, int index, size_t *length) {
  index = lua_absindex(state, index);
  size_t expected = lua_rawlen(state, index), count = 0;
  lua_pushnil(state);
  while (lua_next(state, index)) {
    int valid = 0;
    lua_Integer key = lua_tointegerx(state, -2, &valid);
    if (!valid || key < 1 || (uint64_t)key > expected) {
      lua_pop(state, 2);
      return false;
    }
    count++;
    lua_pop(state, 1);
  }
  if (count != expected)
    return false;
  *length = expected;
  return true;
}
static bool safe_local_id(const char *text) {
  if (!text || !*text || strlen(text) > 128 || !strcmp(text, "$input"))
    return false;
  for (const unsigned char *p = (const unsigned char *)text; *p; p++)
    if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
          (*p >= '0' && *p <= '9') || *p == '_' || *p == '-' || *p == '.'))
      return false;
  return true;
}
static int raw_field(lua_State *state, int index, const char *key) {
  index = lua_absindex(state, index);
  lua_pushstring(state, key);
  return lua_rawget(state, index);
}
static bool push_value(lua_State *state, const NNValue *value, size_t depth) {
  if (depth > VALUE_DEPTH_LIMIT)
    luaL_error(state, "visualization parameter depth limit exceeded");
  if (!lua_checkstack(state, 8))
    luaL_error(state, "visualization parameter stack limit exceeded");
  switch (value->type) {
  case NN_VALUE_BOOL:
    lua_pushboolean(state, value->as.boolean);
    break;
  case NN_VALUE_INT:
    lua_pushinteger(state, (lua_Integer)value->as.integer);
    break;
  case NN_VALUE_REAL:
    lua_pushnumber(state, value->as.real);
    break;
  case NN_VALUE_STRING:
    lua_pushstring(state, value->as.string ? value->as.string : "");
    break;
  case NN_VALUE_ARRAY:
    lua_createtable(state, (int)value->as.array.count, 0);
    for (size_t i = 0; i < value->as.array.count; i++) {
      push_value(state, &value->as.array.items[i], depth + 1);
      lua_rawseti(state, -2, (lua_Integer)i + 1);
    }
    break;
  case NN_VALUE_OBJECT:
    lua_createtable(state, 0, (int)value->as.object.count);
    for (size_t i = 0; i < value->as.object.count; i++) {
      lua_pushstring(state, value->as.object.items[i].key);
      push_value(state, &value->as.object.items[i].value, depth + 1);
      lua_rawset(state, -3);
    }
    break;
  }
  return true;
}
static bool parse_value(lua_State *state, int index, NNValue *value,
                        size_t depth, size_t *items) {
  memset(value, 0, sizeof(*value));
  if (depth > VALUE_DEPTH_LIMIT || ++*items > VALUE_ITEM_LIMIT)
    return false;
  if (!lua_checkstack(state, 8))
    luaL_error(state, "visualization parameter stack limit exceeded");
  index = lua_absindex(state, index);
  switch (lua_type(state, index)) {
  case LUA_TBOOLEAN:
    value->type = NN_VALUE_BOOL;
    value->as.boolean = lua_toboolean(state, index) != 0;
    return true;
  case LUA_TNUMBER: {
    if (lua_isinteger(state, index)) {
      value->type = NN_VALUE_INT;
      value->as.integer = (long long)lua_tointeger(state, index);
      return true;
    }
    int ok = 0;
    double d = lua_tonumberx(state, index, &ok);
    if (!ok || !isfinite(d))
      return false;
    value->type = NN_VALUE_REAL;
    value->as.real = d;
    return true;
  }
  case LUA_TSTRING:
    value->type = NN_VALUE_STRING;
    value->as.string = lua_string(state, index);
    return value->as.string != NULL;
  case LUA_TTABLE: {
    size_t count = lua_rawlen(state, index);
    if (count) {
      if (!dense_sequence(state, index, &count))
        return false;
      if (count > VALUE_ITEM_LIMIT - *items)
        return false;
      value->type = NN_VALUE_ARRAY;
      value->as.array.items = calloc(count, sizeof(NNValue));
      value->as.array.count = count;
      if (!value->as.array.items)
        return false;
      for (size_t i = 0; i < count; i++) {
        lua_rawgeti(state, index, (lua_Integer)i + 1);
        bool ok =
            parse_value(state, -1, &value->as.array.items[i], depth + 1, items);
        lua_pop(state, 1);
        if (!ok)
          return false;
      }
      return true;
    }
    size_t fields = 0;
    lua_pushnil(state);
    while (lua_next(state, index)) {
      if (lua_type(state, -2) != LUA_TSTRING) {
        lua_pop(state, 2);
        return false;
      }
      fields++;
      lua_pop(state, 1);
    }
    if (fields > VALUE_ITEM_LIMIT - *items)
      return false;
    value->type = NN_VALUE_OBJECT;
    value->as.object.items = calloc(fields ? fields : 1, sizeof(NNParameter));
    value->as.object.count = fields;
    if (fields && !value->as.object.items)
      return false;
    size_t i = 0;
    lua_pushnil(state);
    while (lua_next(state, index)) {
      value->as.object.items[i].key = lua_string(state, -2);
      bool ok = value->as.object.items[i].key &&
                parse_value(state, -1, &value->as.object.items[i].value,
                            depth + 1, items);
      lua_pop(state, 1);
      if (!ok) {
        lua_pop(state, 1);
        return false;
      }
      i++;
    }
    return true;
  }
  default:
    return false;
  }
}
static void dispose_parameters(NNParameter *parameters, size_t count) {
  if (!parameters)
    return;
  for (size_t i = 0; i < count; i++) {
    free(parameters[i].key);
    nn_value_dispose(&parameters[i].value);
  }
  free(parameters);
}
static bool parse_parameters(lua_State *state, int table, NN3DPlanNode *node) {
  if (lua_type(state, table) != LUA_TTABLE)
    return false;
  table = lua_absindex(state, table);
  size_t n = 0;
  lua_pushnil(state);
  while (lua_next(state, table)) {
    if (lua_type(state, -2) != LUA_TSTRING) {
      lua_pop(state, 2);
      return false;
    }
    n++;
    lua_pop(state, 1);
  }
  if (n > VALUE_ITEM_LIMIT)
    return false;
  NNParameter *values = calloc(n ? n : 1, sizeof(*values));
  if (n && !values)
    return false;
  // Keep partially decoded values owned by the plan if Lua raises on OOM.
  node->parameters = values;
  node->parameter_count = n;
  size_t i = 0, items = 0;
  lua_pushnil(state);
  while (lua_next(state, table)) {
    values[i].key = lua_string(state, -2);
    bool ok =
        values[i].key && parse_value(state, -1, &values[i].value, 0, &items);
    lua_pop(state, 1);
    if (!ok) {
      lua_pop(state, 1);
      return false;
    }
    i++;
  }
  return true;
}
static bool owner_parameters(lua_State *state, const NNNode *node) {
  lua_createtable(state, 0, (int)node->parameter_count);
  for (size_t i = 0; i < node->parameter_count; i++) {
    lua_pushstring(state, node->parameters[i].key);
    push_value(state, &node->parameters[i].value, 0);
    lua_rawset(state, -3);
  }
  return true;
}
static bool read_plan(lua_State *state, NN3DPlan *plan) {
  if (lua_type(state, -1) != LUA_TTABLE)
    return false;
  raw_field(state, -1, "nodes");
  if (lua_type(state, -1) != LUA_TTABLE)
    return false;
  size_t nodes = 0;
  if (!dense_sequence(state, -1, &nodes) || nodes > PLAN_NODE_LIMIT)
    return false;
  plan->nodes = calloc(nodes ? nodes : 1, sizeof(*plan->nodes));
  if (nodes && !plan->nodes)
    return false;
  plan->node_count = nodes;
  int node_table = lua_absindex(state, -1);
  for (size_t i = 0; i < nodes; i++) {
    lua_rawgeti(state, node_table, (lua_Integer)i + 1);
    if (lua_type(state, -1) != LUA_TTABLE)
      return false;
    NN3DPlanNode *n = &plan->nodes[i];
    raw_field(state, -1, "id");
    n->id = lua_string(state, -1);
    lua_pop(state, 1);
    if (!safe_local_id(n->id))
      return false;
    for (size_t j = 0; j < i; j++)
      if (!strcmp(plan->nodes[j].id, n->id))
        return false;
    raw_field(state, -1, "label");
    if (lua_type(state, -1) == LUA_TSTRING)
      n->label = lua_string(state, -1);
    else if (!lua_isnil(state, -1)) {
      lua_pop(state, 2);
      return false;
    }
    lua_pop(state, 1);
    if (!n->label)
      n->label = nn_text_copy(n->id);
    if (!n->label || !*n->label)
      return false;
    raw_field(state, -1, "body");
    if (!lua_isnil(state, -1) && lua_type(state, -1) != LUA_TBOOLEAN) {
      lua_pop(state, 2);
      return false;
    }
    n->body = lua_toboolean(state, -1) != 0;
    lua_pop(state, 1);
    raw_field(state, -1, "package");
    if (!lua_isnil(state, -1) && lua_type(state, -1) != LUA_TTABLE) {
      lua_pop(state, 2);
      return false;
    }
    if (lua_type(state, -1) == LUA_TTABLE) {
      raw_field(state, -1, "id");
      n->package_id = lua_string(state, -1);
      lua_pop(state, 1);
      raw_field(state, -1, "version");
      n->version = lua_string(state, -1);
      lua_pop(state, 1);
      if (!n->package_id || !n->version) {
        lua_pop(state, 2);
        return false;
      }
      raw_field(state, -2, "parameters");
      if (!lua_isnil(state, -1) && lua_type(state, -1) != LUA_TTABLE) {
        lua_pop(state, 3);
        return false;
      }
      if (lua_type(state, -1) == LUA_TTABLE &&
          !parse_parameters(state, -1, n)) {
        lua_pop(state, 3);
        return false;
      }
      lua_pop(state, 1);
    }
    lua_pop(state, 1);
    if (n->body && (n->package_id || n->parameters)) {
      lua_pop(state, 1);
      return false;
    }
    if (!n->body && !n->package_id) {
      lua_pop(state, 1);
      return false;
    }
    lua_pop(state, 1);
  }
  lua_pop(state, 1);
  raw_field(state, -1, "edges");
  if (lua_type(state, -1) != LUA_TTABLE)
    return false;
  size_t edges = 0;
  if (!dense_sequence(state, -1, &edges) || edges > PLAN_EDGE_LIMIT)
    return false;
  plan->edges = calloc(edges ? edges : 1, sizeof(*plan->edges));
  if (edges && !plan->edges)
    return false;
  plan->edge_count = edges;
  int edge_table = lua_absindex(state, -1);
  for (size_t i = 0; i < edges; i++) {
    lua_rawgeti(state, edge_table, (lua_Integer)i + 1);
    if (lua_type(state, -1) != LUA_TTABLE)
      return false;
    NN3DPlanEdge *e = &plan->edges[i];
    raw_field(state, -1, "source");
    e->source = lua_string(state, -1);
    lua_pop(state, 1);
    raw_field(state, -1, "sourceHandle");
    e->source_handle = lua_string(state, -1);
    lua_pop(state, 1);
    raw_field(state, -1, "target");
    e->target = lua_string(state, -1);
    lua_pop(state, 1);
    raw_field(state, -1, "targetHandle");
    e->target_handle = lua_string(state, -1);
    lua_pop(state, 1);
    if (!e->source || !e->source_handle || !e->target || !e->target_handle) {
      lua_pop(state, 1);
      return false;
    }
    lua_pop(state, 1);
  }
  lua_pop(state, 1);
  raw_field(state, -1, "outputs");
  if (lua_type(state, -1) != LUA_TTABLE)
    return false;
  size_t outputs = 0;
  lua_pushnil(state);
  while (lua_next(state, -2)) {
    if (lua_type(state, -2) != LUA_TSTRING) {
      lua_pop(state, 2);
      return false;
    }
    outputs++;
    lua_pop(state, 1);
  }
  if (outputs > 2)
    return false;
  plan->outputs = calloc(outputs ? outputs : 1, sizeof(*plan->outputs));
  if (outputs && !plan->outputs)
    return false;
  plan->output_count = outputs;
  int output_table = lua_absindex(state, -1);
  size_t i = 0;
  lua_pushnil(state);
  while (lua_next(state, output_table)) {
    NN3DPlanOutput *o = &plan->outputs[i++];
    o->id = lua_string(state, -2);
    if (lua_type(state, -1) != LUA_TTABLE) {
      lua_pop(state, 2);
      return false;
    }
    raw_field(state, -1, "node");
    o->node = lua_string(state, -1);
    lua_pop(state, 1);
    raw_field(state, -1, "handle");
    o->handle = lua_string(state, -1);
    lua_pop(state, 1);
    if (!o->id || !o->node || !o->handle) {
      lua_pop(state, 2);
      return false;
    }
    lua_pop(state, 1);
  }
  lua_pop(state, 1);
  return true;
}

static int initialize_safe_lua(lua_State *state) {
  luaL_requiref(state, LUA_GNAME, luaopen_base, 1);
  lua_pop(state, 1);
  lua_pushnil(state);
  lua_setglobal(state, "dofile");
  lua_pushnil(state);
  lua_setglobal(state, "loadfile");
  lua_pushnil(state);
  lua_setglobal(state, "load");
  lua_pushnil(state);
  lua_setglobal(state, "print");
  lua_pushnil(state);
  lua_setglobal(state, "warn");
  lua_pushnil(state);
  lua_setglobal(state, "collectgarbage");
  luaL_requiref(state, LUA_MATHLIBNAME, luaopen_math, 1);
  lua_pop(state, 1);
  luaL_requiref(state, LUA_TABLIBNAME, luaopen_table, 1);
  lua_pop(state, 1);
  luaL_requiref(state, LUA_STRLIBNAME, luaopen_string, 1);
  lua_pop(state, 1);
  return 0;
}
static int run_plan(lua_State *state) {
  LuaBudget *budget = *(LuaBudget **)lua_getextraspace(state);
  int status = luaL_loadbufferx(state, budget->source, budget->source_length,
                                "visualization", "t");
  if (status != LUA_OK)
    return lua_error(state);
  lua_call(state, 0, 1);
  if (lua_type(state, -1) != LUA_TFUNCTION)
    return luaL_error(state, "visualization source must return a function");
  owner_parameters(state, budget->owner_node);
  lua_call(state, 1, 1);
  if (!read_plan(state, budget->plan))
    return luaL_error(
        state, "visualization function returned an invalid declarative plan");
  return 0;
}

void nn_3d_plan_dispose(NN3DPlan *plan) {
  if (!plan)
    return;
  for (size_t i = 0; i < plan->node_count; i++) {
    NN3DPlanNode *n = &plan->nodes[i];
    free(n->id);
    free(n->label);
    free(n->package_id);
    free(n->version);
    dispose_parameters(n->parameters, n->parameter_count);
  }
  for (size_t i = 0; i < plan->edge_count; i++) {
    free(plan->edges[i].source);
    free(plan->edges[i].source_handle);
    free(plan->edges[i].target);
    free(plan->edges[i].target_handle);
  }
  for (size_t i = 0; i < plan->output_count; i++) {
    free(plan->outputs[i].id);
    free(plan->outputs[i].node);
    free(plan->outputs[i].handle);
  }
  free(plan->nodes);
  free(plan->edges);
  free(plan->outputs);
  memset(plan, 0, sizeof(*plan));
}

bool nn_3d_plan_load(const NNCatalog *catalog, const NNPackage *owner,
                     const NNNode *node, NN3DPlan *plan, char *error,
                     size_t cap) {
  if (!catalog || !owner || !node || !plan)
    return nn_fail(error, cap, "Invalid visualization recipe request");
  memset(plan, 0, sizeof(*plan));
  if (!owner->visualization_file)
    return true;
  char *path = nn_path_join(owner->directory, owner->visualization_file);
  if (!path)
    return nn_fail(error, cap, "Out of memory locating visualization recipe");
  int fd = open(path, O_RDONLY | O_NOFOLLOW);
  struct stat info;
  if (fd < 0 || fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) ||
      info.st_size < 0 || info.st_size > LUA_SOURCE_LIMIT) {
    if (fd >= 0)
      close(fd);
    free(path);
    return nn_fail(error, cap,
                   "Visualization recipe is missing, not a regular file, or "
                   "exceeds 1 MiB");
  }
  size_t length = (size_t)info.st_size;
  char *source = malloc(length + 1);
  if (!source) {
    close(fd);
    free(path);
    return nn_fail(error, cap, "Out of memory reading visualization recipe");
  }
  size_t used = 0;
  while (used < length) {
    ssize_t n = read(fd, source + used, length - used);
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0)
      break;
    used += (size_t)n;
  }
  close(fd);
  free(path);
  if (used != length) {
    free(source);
    return nn_fail(error, cap, "Unable to read visualization recipe");
  }
  source[length] = '\0';
  LuaBudget budget = {0};
  budget.remaining = LUA_INSTRUCTION_LIMIT;
  budget.source = source;
  budget.source_length = length;
  budget.owner_node = node;
  budget.plan = plan;
  lua_State *state = lua_newstate(budget_alloc, &budget, 0);
  if (!state) {
    free(source);
    return nn_fail(error, cap, "Unable to allocate visualization Lua state");
  }
  *(LuaBudget **)lua_getextraspace(state) = &budget;
  lua_pushcfunction(state, initialize_safe_lua);
  int status = lua_pcall(state, 0, 0, 0);
  if (status == LUA_OK) {
    lua_sethook(state, instruction_hook, LUA_MASKCOUNT, 1000);
    lua_pushcfunction(state, run_plan);
    status = lua_pcall(state, 0, 0, 0);
  }
  free(source);
  if (status != LUA_OK) {
    const char *message = lua_tostring(state, -1);
    nn_errorf(error, cap, "Invalid visualization recipe for %s: %s", owner->id,
              message ? message : "expected a declarative plan");
    nn_3d_plan_dispose(plan);
    lua_close(state);
    return false;
  }
  lua_close(state);
  return true;
}
