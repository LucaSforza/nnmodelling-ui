#include "editor.h"

#include "project.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum { FIELD_NONE, FIELD_OPEN, FIELD_PARENT, FIELD_ID, FIELD_NAME,
       FIELD_NODE_NAME, FIELD_PARAMETER };

typedef struct {
    NNPlatform *platform;
    NNProject *project;
    const char *core_root;
    int width, height;
    char open_path[1024], parent[1024], project_id[128], project_name[256];
    char edit[512], selected[128], selected_edge[128], draft_source[128];
    char status[512];
    int focus, parameter_index, palette_scroll;
    int resource_scroll;
    bool close_prompt, drag_node, drag_pan, drag_edge, resources_view;
    bool quit_pending, exit_requested;
    float pointer_x, pointer_y, down_x, down_y;
    double camera_x, camera_y, zoom, node_x, node_y, pan_x, pan_y;
    unsigned long sequence;
} Editor;

static const NNPlatformColor background = {15, 20, 31, 255};
static const NNPlatformColor panel = {23, 31, 45, 255};
static const NNPlatformColor surface = {32, 42, 59, 255};
static const NNPlatformColor line_color = {69, 84, 104, 255};
static const NNPlatformColor text_color = {226, 234, 244, 255};
static const NNPlatformColor muted = {150, 166, 185, 255};
static const NNPlatformColor accent = {80, 194, 153, 255};
static const NNPlatformColor warning_color = {245, 174, 94, 255};

static NNPlatformRect rect(float x, float y, float w, float h)
{
    return (NNPlatformRect){x, y, w, h};
}

static bool inside(NNPlatformRect box, float x, float y)
{
    return x >= box.x && x < box.x + box.width &&
           y >= box.y && y < box.y + box.height;
}

static void copy_text(char *target, size_t capacity, const char *source)
{
    if (capacity) snprintf(target, capacity, "%s", source ? source : "");
}

static void label(Editor *e, float x, float y, const char *value, NNPlatformColor color)
{
    platform_draw_text(e->platform, x, y + 17, value, color);
}

static void box(Editor *e, NNPlatformRect area, NNPlatformColor color)
{
    platform_fill_rect(e->platform, area, color);
}

static void button(Editor *e, NNPlatformRect area, const char *caption, bool highlighted)
{
    box(e, area, highlighted ? accent : surface);
    platform_outline_rect(e->platform, area, highlighted ? accent : line_color);
    label(e, area.x + 12, area.y + (area.height - 18) / 2,
          caption, highlighted ? background : text_color);
}

static void field(Editor *e, NNPlatformRect area, const char *caption,
                  const char *value, bool focused)
{
    label(e, area.x, area.y - 22, caption, muted);
    box(e, area, background);
    platform_outline_rect(e->platform, area, focused ? accent : line_color);
    platform_push_clip(e->platform, area);
    label(e, area.x + 10, area.y + 9, value, text_color);
    platform_pop_clip(e->platform);
}

static NNPlatformRect viewport(const Editor *e)
{
    return rect(244, 62, e->width - 244 - 320,
                e->height - 62 - 48);
}

static NNPlatformPoint to_screen(const Editor *e, double x, double y)
{
    NNPlatformRect view = viewport(e);
    return (NNPlatformPoint){view.x + (float)((x - e->camera_x) * e->zoom),
                             view.y + (float)((y - e->camera_y) * e->zoom)};
}

static NNPlatformPoint to_world(const Editor *e, float x, float y)
{
    NNPlatformRect view = viewport(e);
    return (NNPlatformPoint){(float)(e->camera_x + (x - view.x) / e->zoom),
                             (float)(e->camera_y + (y - view.y) / e->zoom)};
}

static const NNPackage *node_package(const Editor *e, const NNNode *node)
{
    return nn_catalog_find(nn_project_catalog(e->project),
                           node->package_id, node->package_version);
}

static NNPlatformRect node_rect(const Editor *e, const NNNode *node)
{
    NNPlatformPoint p = to_screen(e, node->x, node->y);
    if (e->drag_node && !strcmp(e->selected, node->id)) {
        p.x += e->pointer_x - e->down_x;
        p.y += e->pointer_y - e->down_y;
    }
    const NNPackage *package = node_package(e, node);
    float width = package ? (float)package->width : 180;
    float height = package ? (float)package->height : 100;
    return rect(p.x, p.y, width * (float)e->zoom, height * (float)e->zoom);
}

static NNPlatformPoint handle_point(const Editor *e, const NNNode *node, bool output)
{
    NNPlatformRect box = node_rect(e, node);
    return (NNPlatformPoint){output ? box.x + box.width : box.x,
                             box.y + box.height * 0.5f};
}

static bool is_input_kind(const NNPackage *package)
{
    return package && package->kind && !strcmp(package->kind, "input");
}

static bool is_output_kind(const NNPackage *package)
{
    return package && package->kind && !strcmp(package->kind, "output");
}

static float distance2(float x1, float y1, float x2, float y2)
{
    float dx = x1 - x2, dy = y1 - y2;
    return dx * dx + dy * dy;
}

static NNPlatformColor package_color(const char *hex)
{
    unsigned int r, g, b;
    if (hex && sscanf(hex, "#%2x%2x%2x", &r, &g, &b) == 3)
        return (NNPlatformColor){(unsigned char)r, (unsigned char)g,
                                 (unsigned char)b, 255};
    return accent;
}

static void set_status(Editor *e, const char *message)
{
    copy_text(e->status, sizeof(e->status), message);
}

static void fit_view(Editor *e)
{
    if (!e->project) return;
    NNModel *model = nn_project_model(e->project);
    size_t count = nn_model_node_count(model);
    if (!count) { e->camera_x = e->camera_y = 0; e->zoom = 1; return; }
    double left = INFINITY, top = INFINITY, right = -INFINITY, bottom = -INFINITY;
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        const NNPackage *package = node_package(e, node);
        double width = package ? package->width : 180;
        double height = package ? package->height : 100;
        if (node->x < left) left = node->x;
        if (node->y < top) top = node->y;
        if (node->x + width > right) right = node->x + width;
        if (node->y + height > bottom) bottom = node->y + height;
    }
    NNPlatformRect view = viewport(e);
    double usable_width = view.width - 80, usable_height = view.height - 80;
    if (usable_width <= 0 || usable_height <= 0) return;
    double sx = usable_width / fmax(1, right - left), sy = usable_height / fmax(1, bottom - top);
    e->zoom = fmin(1.4, fmax(0.3, fmin(sx, sy)));
    e->camera_x = (left + right) / 2 - view.width / (2 * e->zoom);
    e->camera_y = (top + bottom) / 2 - view.height / (2 * e->zoom);
}

static void activate_project(Editor *e, NNProject *project)
{
    NNProject *old = e->project;
    e->project = project;
    nn_project_close(old);
    e->selected[0] = e->selected_edge[0] = '\0';
    e->palette_scroll = 0;
    e->close_prompt = false;
    e->focus = FIELD_NONE;
    fit_view(e);
    set_status(e, "Project ready. Drag nodes, connect ports, inspect parameters, save.");
}

static void perform_open(Editor *e)
{
    char error[512] = "";
    NNProject *project = nn_project_open(e->open_path, e->core_root,
                                         error, sizeof(error));
    if (!project) { set_status(e, error); return; }
    activate_project(e, project);
}

static void perform_create(Editor *e, bool mnist)
{
    char error[512] = "";
    NNProject *project = nn_project_create(e->parent, e->project_id,
                                            e->project_name, mnist,
                                            e->core_root, error, sizeof(error));
    if (!project) { set_status(e, error); return; }
    activate_project(e, project);
}

static void perform_save(Editor *e)
{
    char error[512] = "";
    if (nn_project_save(e->project, error, sizeof(error)))
        set_status(e, "Saved model.json atomically.");
    else set_status(e, error);
}

static void apply_default(Editor *e, const char *node_id,
                          const NNParameterDef *definition)
{
    NNValue value = {0};
    if (definition->has_default) {
        switch (definition->default_value.type) {
        case NN_PARAMETER_BOOLEAN:
            value.type = NN_VALUE_BOOL;
            value.as.boolean = definition->default_value.as.boolean; break;
        case NN_PARAMETER_INTEGER:
            value.type = NN_VALUE_INT;
            value.as.integer = definition->default_value.as.integer; break;
        case NN_PARAMETER_NUMBER:
            value.type = NN_VALUE_REAL;
            value.as.real = definition->default_value.as.number; break;
        case NN_PARAMETER_STRING:
            value.type = NN_VALUE_STRING;
            value.as.string = (char *)definition->default_value.as.string; break;
        default: return;
        }
    } else if (!strcmp(definition->type, "integer")) {
        value.type = NN_VALUE_INT;
        value.as.integer = definition->has_minimum
            ? (long long)fmax(1, definition->minimum) : 1;
    } else if (!strcmp(definition->type, "number")) {
        value.type = NN_VALUE_REAL;
        value.as.real = definition->has_minimum ? definition->minimum : 0;
    } else if (!strcmp(definition->type, "boolean")) {
        value.type = NN_VALUE_BOOL;
        value.as.boolean = false;
    } else {
        value.type = NN_VALUE_STRING;
        const NNDataset *dataset = nn_project_active_dataset(e->project);
        if (!strcmp(definition->key, "binding") && dataset && dataset->input_count)
            value.as.string = dataset->inputs[0].name;
        else if (definition->choice_count)
            value.as.string = (char *)definition->choices[0];
        else value.as.string = "";
    }
    char error[256] = "";
    nn_model_set_parameter(nn_project_model(e->project), node_id,
                           definition->key, &value, error, sizeof(error));
}

static void add_package_node(Editor *e, const NNPackage *package)
{
    if (!package) return;
    char id[96], error[512] = "";
    snprintf(id, sizeof(id), "node-%lu", ++e->sequence);
    NNPlatformRect view = viewport(e);
    NNPlatformPoint center = to_world(e, view.x + view.width / 2,
                                      view.y + view.height / 2);
    NNModel *model = nn_project_model(e->project);
    if (!nn_model_add_node(model, id, package->name, package->id,
                           package->version, "", center.x, center.y,
                           error, sizeof(error))) {
        set_status(e, error); return;
    }
    for (size_t i = 0; i < package->parameter_count; ++i)
        apply_default(e, id, &package->parameters[i]);
    nn_project_mark_dirty(e->project);
    copy_text(e->selected, sizeof(e->selected), id);
    e->selected_edge[0] = '\0';
    set_status(e, "Node added. Select parameter in inspector to edit.");
}

static const NNParameter *find_parameter(const NNNode *node, const char *key)
{
    for (size_t i = 0; i < node->parameter_count; ++i)
        if (!strcmp(node->parameters[i].key, key)) return &node->parameters[i];
    return NULL;
}

static void format_value(const NNValue *value, char *buffer, size_t size)
{
    if (!value) { copy_text(buffer, size, ""); return; }
    switch (value->type) {
    case NN_VALUE_BOOL: copy_text(buffer, size, value->as.boolean ? "true" : "false"); break;
    case NN_VALUE_INT: snprintf(buffer, size, "%lld", value->as.integer); break;
    case NN_VALUE_REAL: snprintf(buffer, size, "%g", value->as.real); break;
    case NN_VALUE_STRING: copy_text(buffer, size, value->as.string); break;
    case NN_VALUE_ARRAY: copy_text(buffer, size, "[array]"); break;
    }
}

static void begin_edit(Editor *e, int focus, int parameter_index)
{
    e->focus = focus;
    e->parameter_index = parameter_index;
    const NNNode *node = nn_model_find_node(nn_project_model(e->project), e->selected);
    if (!node) return;
    if (focus == FIELD_NODE_NAME) copy_text(e->edit, sizeof(e->edit), node->label);
    if (focus == FIELD_PARAMETER) {
        const NNPackage *package = node_package(e, node);
        if (!package || parameter_index < 0 ||
            (size_t)parameter_index >= package->parameter_count) return;
        const NNParameter *parameter = find_parameter(node, package->parameters[parameter_index].key);
        format_value(parameter ? &parameter->value : NULL, e->edit, sizeof(e->edit));
    }
}

static void commit_edit(Editor *e)
{
    if (!e->project || (e->focus != FIELD_NODE_NAME && e->focus != FIELD_PARAMETER)) return;
    NNModel *model = nn_project_model(e->project);
    const NNNode *node = nn_model_find_node(model, e->selected);
    if (!node) { e->focus = FIELD_NONE; return; }
    char error[512] = "";
    bool okay = false;
    if (e->focus == FIELD_NODE_NAME) {
        okay = nn_model_rename_node(model, node->id, e->edit, error, sizeof(error));
    } else {
        const NNPackage *package = node_package(e, node);
        if (!package || e->parameter_index < 0 ||
            (size_t)e->parameter_index >= package->parameter_count) return;
        const NNParameterDef *definition = &package->parameters[e->parameter_index];
        NNValue value = {0};
        if (!strcmp(definition->type, "boolean")) {
            value.type = NN_VALUE_BOOL;
            if (strcmp(e->edit, "true") && strcmp(e->edit, "false")) {
                set_status(e, "Boolean must be true or false."); return;
            }
            value.as.boolean = !strcmp(e->edit, "true");
        } else if (!strcmp(definition->type, "integer")) {
            char *end;
            value.type = NN_VALUE_INT;
            value.as.integer = strtoll(e->edit, &end, 10);
            if (!e->edit[0] || *end ||
                (definition->has_minimum && value.as.integer < definition->minimum)) {
                set_status(e, "Invalid integer or below minimum."); return;
            }
        } else if (!strcmp(definition->type, "number")) {
            char *end;
            value.type = NN_VALUE_REAL;
            value.as.real = strtod(e->edit, &end);
            if (!e->edit[0] || *end || !isfinite(value.as.real) ||
                (definition->has_minimum && value.as.real < definition->minimum)) {
                set_status(e, "Invalid number or below minimum."); return;
            }
        } else {
            value.type = NN_VALUE_STRING;
            value.as.string = e->edit;
            if (definition->choice_count) {
                bool found = false;
                for (size_t i = 0; i < definition->choice_count; ++i)
                    if (!strcmp(e->edit, definition->choices[i])) found = true;
                if (!found) { set_status(e, "Value not in allowed choices."); return; }
            }
        }
        okay = nn_model_set_parameter(model, node->id, definition->key,
                                       &value, error, sizeof(error));
    }
    if (okay) { nn_project_mark_dirty(e->project); set_status(e, "Inspector change applied."); }
    else set_status(e, error);
    e->focus = FIELD_NONE;
}

static void draw_chooser(Editor *e)
{
    float card_x = (e->width - 700) / 2.0f, card_y = (e->height - 650) / 2.0f;
    if (card_y < 60) card_y = 60;
    box(e, rect(card_x, card_y, 700, 650), panel);
    platform_outline_rect(e->platform, rect(card_x, card_y, 700, 650), line_color);
    label(e, card_x + 36, card_y + 24, "NNMODELLING", accent);
    label(e, card_x + 36, card_y + 64, "Native project editor", text_color);
    label(e, card_x + 36, card_y + 103,
          "Create or open a local project. Model, datasets and packages stay together.", muted);
    field(e, rect(card_x + 36, card_y + 170, 628, 42), "Open project folder",
          e->open_path, e->focus == FIELD_OPEN);
    button(e, rect(card_x + 36, card_y + 228, 154, 40), "Open project", false);
    label(e, card_x + 36, card_y + 293, "NEW PROJECT", accent);
    field(e, rect(card_x + 36, card_y + 342, 628, 42), "Parent folder",
          e->parent, e->focus == FIELD_PARENT);
    field(e, rect(card_x + 36, card_y + 425, 288, 42), "Project ID",
          e->project_id, e->focus == FIELD_ID);
    field(e, rect(card_x + 340, card_y + 425, 324, 42), "Display name",
          e->project_name, e->focus == FIELD_NAME);
    button(e, rect(card_x + 36, card_y + 510, 160, 44), "Create blank", false);
    button(e, rect(card_x + 212, card_y + 510, 222, 44), "Create MNIST MLP", true);
    label(e, card_x + 36, card_y + 580,
          "MNIST template: 28x28 image, two hidden layers, 10 logits.", muted);
}

static void draw_graph(Editor *e)
{
    NNPlatformRect view = viewport(e);
    box(e, view, background);
    platform_push_clip(e->platform, view);
    float spacing = (float)(80 * e->zoom);
    if (spacing >= 24) {
        float start_x = view.x + (float)fmod(-e->camera_x * e->zoom, spacing);
        float start_y = view.y + (float)fmod(-e->camera_y * e->zoom, spacing);
        for (float x = start_x; x < view.x + view.width; x += spacing)
            platform_line(e->platform, (NNPlatformPoint){x, view.y},
                          (NNPlatformPoint){x, view.y + view.height},
                          (NNPlatformColor){28, 38, 53, 255});
        for (float y = start_y; y < view.y + view.height; y += spacing)
            platform_line(e->platform, (NNPlatformPoint){view.x, y},
                          (NNPlatformPoint){view.x + view.width, y},
                          (NNPlatformColor){28, 38, 53, 255});
    }
    NNModel *model = nn_project_model(e->project);
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        const NNNode *source = nn_model_find_node(model, edge->source_id);
        const NNNode *target = nn_model_find_node(model, edge->target_id);
        if (!source || !target) continue;
        NNPlatformPoint a = handle_point(e, source, true), b = handle_point(e, target, false);
        NNPlatformColor color = !strcmp(e->selected_edge, edge->id) ? warning_color : line_color;
        platform_line(e->platform, a, b, color);
        platform_circle(e->platform, b, 3, color, true);
    }
    if (e->drag_edge) {
        const NNNode *source = nn_model_find_node(model, e->draft_source);
        if (source) platform_line(e->platform, handle_point(e, source, true),
                                  (NNPlatformPoint){e->pointer_x, e->pointer_y}, accent);
    }
    for (size_t i = 0; i < nn_model_node_count(model); ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        const NNPackage *package = node_package(e, node);
        NNPlatformRect area = node_rect(e, node);
        if (area.x + area.width < view.x || area.x > view.x + view.width ||
            area.y + area.height < view.y || area.y > view.y + view.height) continue;
        box(e, area, surface);
        box(e, rect(area.x, area.y, area.width, 6), package_color(package ? package->color : NULL));
        platform_outline_rect(e->platform, area,
                              !strcmp(e->selected, node->id) ? accent : line_color);
        if (e->zoom >= 0.55) {
            label(e, area.x + 12, area.y + 17, node->label, text_color);
            label(e, area.x + 12, area.y + 49,
                  package ? package->name : node->package_id, muted);
        } else label(e, area.x + 2, area.y + area.height + 3,
                     node->id, text_color);
        if (!is_input_kind(package)) platform_circle(e->platform, handle_point(e, node, false),
                                                     7, accent, true);
        if (!is_output_kind(package)) platform_circle(e->platform, handle_point(e, node, true),
                                                      7, accent, true);
    }
    platform_pop_clip(e->platform);
    char zoom[64];
    snprintf(zoom, sizeof(zoom), "%.0f%%", e->zoom * 100);
    label(e, view.x + 16, view.y + view.height - 32, zoom, muted);
}

static void draw_palette(Editor *e)
{
    NNPlatformRect area = rect(0, 62, 244, e->height - 110);
    box(e, area, panel);
    label(e, 20, 76, "STEREOTYPES", accent);
    label(e, 20, 101, "Click to add to canvas", muted);
    platform_push_clip(e->platform, rect(0, 130, 244, area.height - 68));
    const NNCatalog *catalog = nn_project_catalog(e->project);
    for (size_t i = (size_t)e->palette_scroll; i < nn_catalog_count(catalog); ++i) {
        float y = 132 + (float)(i - (size_t)e->palette_scroll) * 35;
        if (y > e->height - 80) break;
        const NNPackage *package = nn_catalog_at(catalog, i);
        box(e, rect(12, y, 220, 31), surface);
        box(e, rect(12, y, 5, 31), package_color(package->color));
        label(e, 25, y + 5, package->name, text_color);
    }
    platform_pop_clip(e->platform);
}

static void format_shape(const NNValue *shape, char *output, size_t capacity)
{
    copy_text(output, capacity, "[");
    if (!shape || shape->type != NN_VALUE_ARRAY) return;
    for (size_t i = 0; i < shape->as.array.count; ++i) {
        char dimension[64];
        format_value(&shape->as.array.items[i], dimension, sizeof(dimension));
        size_t used = strlen(output);
        if (used + strlen(dimension) + 3 >= capacity) break;
        if (i) strcat(output, ",");
        strcat(output, dimension);
    }
    if (strlen(output) + 2 < capacity) strcat(output, "]");
}

static void draw_resources(Editor *e)
{
    float x = (float)e->width - 320;
    label(e, x + 20, 132, "PROJECT DATASETS", accent);
    size_t count = nn_project_dataset_count(e->project);
    float y = 171;
    for (size_t i = 0; i < count; ++i) {
        const NNDataset *dataset = nn_project_dataset_at(e->project, i);
        char info[280];
        snprintf(info, sizeof(info), "%s @ %s", dataset->id, dataset->version);
        label(e, x + 20, y, info, text_color); y += 31;
        for (size_t slot = 0; slot < dataset->input_count && slot < 3; ++slot) {
            char shape[96];
            format_shape(&dataset->inputs[slot].shape, shape, sizeof(shape));
            snprintf(info, sizeof(info), "in %s %s %s", dataset->inputs[slot].name,
                     dataset->inputs[slot].dtype, shape);
            label(e, x + 30, y, info, muted); y += 27;
        }
        for (size_t slot = 0; slot < dataset->target_count && slot < 2; ++slot) {
            char shape[96];
            format_shape(&dataset->targets[slot].shape, shape, sizeof(shape));
            snprintf(info, sizeof(info), "target %s %s %s", dataset->targets[slot].name,
                     dataset->targets[slot].dtype, shape);
            label(e, x + 30, y, info, muted); y += 27;
        }
        y += 8;
    }
    if (!count) { label(e, x + 20, y, "No project dataset", muted); y += 35; }
    label(e, x + 20, y + 12, "ACTIVE STEREOTYPES", accent);
    float top = y + 48;
    platform_push_clip(e->platform, rect(x + 12, top, 296, e->height - top - 60));
    const NNCatalog *catalog = nn_project_catalog(e->project);
    for (size_t i = (size_t)e->resource_scroll; i < nn_catalog_count(catalog); ++i) {
        float row = top + (float)(i - (size_t)e->resource_scroll) * 52;
        if (row > e->height - 65) break;
        const NNPackage *package = nn_catalog_at(catalog, i);
        char info[280];
        snprintf(info, sizeof(info), "%s @ %s", package->id, package->version);
        label(e, x + 20, row, info, text_color);
        if (package->dependency_count) {
            snprintf(info, sizeof(info), "requires %s %s", package->dependencies[0].id,
                     package->dependencies[0].version_constraint);
            label(e, x + 30, row + 22, info, muted);
        }
    }
    platform_pop_clip(e->platform);
}

static void draw_inspector(Editor *e)
{
    float x = (float)e->width - 320;
    box(e, rect(x, 62, 320, e->height - 110), panel);
    button(e, rect(x + 12, 72, 128, 38), "Inspector", !e->resources_view);
    button(e, rect(x + 148, 72, 160, 38), "Resources", e->resources_view);
    if (e->resources_view) { draw_resources(e); return; }
    NNModel *model = nn_project_model(e->project);
    const NNNode *node = nn_model_find_node(model, e->selected);
    if (node) {
        const NNPackage *package = node_package(e, node);
        label(e, x + 20, 105, package ? package->name : node->package_id, text_color);
        field(e, rect(x + 20, 157, 280, 38), "Node name",
              e->focus == FIELD_NODE_NAME ? e->edit : node->label,
              e->focus == FIELD_NODE_NAME);
        label(e, x + 20, 213, "PARAMETERS", accent);
        if (package) {
            platform_push_clip(e->platform, rect(x + 12, 242, 296, 198));
            for (size_t i = 0; i < package->parameter_count; ++i) {
                float y = 270 + (float)i * 57;
                const NNParameterDef *definition = &package->parameters[i];
                const NNParameter *parameter = find_parameter(node, definition->key);
                char value[512];
                format_value(parameter ? &parameter->value : NULL, value, sizeof(value));
                field(e, rect(x + 20, y, 280, 34), definition->key,
                      e->focus == FIELD_PARAMETER && e->parameter_index == (int)i
                          ? e->edit : value,
                      e->focus == FIELD_PARAMETER && e->parameter_index == (int)i);
            }
            platform_pop_clip(e->platform);
        }
        label(e, x + 20, 448, "Delete: remove selected node", muted);
    } else if (e->selected_edge[0]) {
        label(e, x + 20, 112, "Connection selected", text_color);
        label(e, x + 20, 146, e->selected_edge, muted);
        label(e, x + 20, 186, "Delete: disconnect", muted);
    } else label(e, x + 20, 112, "Select a graph node", muted);
    float resource_y = 492;
    box(e, rect(x + 12, resource_y, 296, e->height - resource_y - 60), background);
    label(e, x + 20, resource_y + 12, "PROJECT RESOURCES", accent);
    const NNDataset *dataset = nn_project_active_dataset(e->project);
    char info[256];
    snprintf(info, sizeof(info), "Datasets %zu  |  Packages %zu",
             nn_project_dataset_count(e->project),
             nn_catalog_count(nn_project_catalog(e->project)));
    label(e, x + 20, resource_y + 41, info, muted);
    if (dataset) {
        snprintf(info, sizeof(info), "%s @ %s", dataset->name, dataset->version);
        label(e, x + 20, resource_y + 74, info, text_color);
        for (size_t i = 0; i < dataset->input_count && i < 3; ++i) {
            snprintf(info, sizeof(info), "input %s : %s", dataset->inputs[i].name,
                     dataset->inputs[i].dtype);
            label(e, x + 20, resource_y + 105 + (float)i * 24, info, muted);
        }
        for (size_t i = 0; i < dataset->target_count && i < 2; ++i) {
            snprintf(info, sizeof(info), "target %s : %s", dataset->targets[i].name,
                     dataset->targets[i].dtype);
            label(e, x + 20, resource_y + 185 + (float)i * 24, info, muted);
        }
    } else label(e, x + 20, resource_y + 74, "No active dataset", muted);
}

static void draw_editor(Editor *e)
{
    box(e, rect(0, 0, e->width, 62), panel);
    label(e, 20, 17, "NNMODELLING", accent);
    button(e, rect(186, 10, 88, 42), "Save", false);
    button(e, rect(284, 10, 88, 42), "Close", false);
    button(e, rect(382, 10, 76, 42), "Fit", false);
    button(e, rect(468, 10, 106, 42), "Arrange", false);
    char title[512];
    snprintf(title, sizeof(title), "%s%s", nn_project_name(e->project),
             nn_project_dirty(e->project) ? "  *" : "");
    label(e, 596, 18, title, text_color);
    draw_palette(e);
    draw_graph(e);
    draw_inspector(e);
    box(e, rect(0, e->height - 48, e->width, 48), panel);
    label(e, 18, e->height - 34, e->status, warning_color);
    if (e->close_prompt) {
        box(e, rect(0, 0, e->width, e->height), (NNPlatformColor){0, 0, 0, 180});
        float x = e->width / 2.0f - 235, y = e->height / 2.0f - 95;
        box(e, rect(x, y, 470, 190), panel);
        label(e, x + 24, y + 22, "Unsaved project changes", text_color);
        label(e, x + 24, y + 53, "Save before closing?", muted);
        button(e, rect(x + 22, y + 112, 115, 42), "Save", true);
        button(e, rect(x + 154, y + 112, 126, 42), "Discard", false);
        button(e, rect(x + 297, y + 112, 130, 42), "Cancel", false);
    }
}

static void render(Editor *e)
{
    if (!platform_begin_frame(e->platform, background)) return;
    if (e->project) draw_editor(e);
    else draw_chooser(e);
    static bool captured;
    const char *capture = getenv("NN_UI_QA_CAPTURE");
    if (capture && *capture && !captured) {
        if (!platform_capture_bmp(e->platform, capture))
            fprintf(stderr, "QA capture failed: %s\n", platform_error());
        captured = true;
    }
    platform_end_frame(e->platform);
}

static char *focus_buffer(Editor *e, size_t *capacity)
{
    switch (e->focus) {
    case FIELD_OPEN: *capacity = sizeof(e->open_path); return e->open_path;
    case FIELD_PARENT: *capacity = sizeof(e->parent); return e->parent;
    case FIELD_ID: *capacity = sizeof(e->project_id); return e->project_id;
    case FIELD_NAME: *capacity = sizeof(e->project_name); return e->project_name;
    case FIELD_NODE_NAME:
    case FIELD_PARAMETER: *capacity = sizeof(e->edit); return e->edit;
    default: *capacity = 0; return NULL;
    }
}

static void leave_focus(Editor *e)
{
    if (e->focus == FIELD_NODE_NAME || e->focus == FIELD_PARAMETER) commit_edit(e);
    e->focus = FIELD_NONE;
}

static void handle_chooser_click(Editor *e, float x, float y)
{
    float card_x = (e->width - 700) / 2.0f, card_y = (e->height - 650) / 2.0f;
    if (card_y < 60) card_y = 60;
    e->focus = FIELD_NONE;
    if (inside(rect(card_x + 36, card_y + 170, 628, 42), x, y)) e->focus = FIELD_OPEN;
    else if (inside(rect(card_x + 36, card_y + 342, 628, 42), x, y)) e->focus = FIELD_PARENT;
    else if (inside(rect(card_x + 36, card_y + 425, 288, 42), x, y)) e->focus = FIELD_ID;
    else if (inside(rect(card_x + 340, card_y + 425, 324, 42), x, y)) e->focus = FIELD_NAME;
    else if (inside(rect(card_x + 36, card_y + 228, 154, 40), x, y)) perform_open(e);
    else if (inside(rect(card_x + 36, card_y + 510, 160, 44), x, y)) perform_create(e, false);
    else if (inside(rect(card_x + 212, card_y + 510, 222, 44), x, y)) perform_create(e, true);
}

static void arrange_graph(Editor *e)
{
    NNModel *model = nn_project_model(e->project);
    size_t count = nn_model_node_count(model);
    char error[256] = "";
    for (size_t i = 0; i < count; ++i) {
        const NNNode *node = nn_model_node_at(model, i);
        char id[128];
        copy_text(id, sizeof(id), node->id);
        if (!nn_model_move_node(model, id, (double)i * 260, 100,
                                error, sizeof(error))) {
            set_status(e, error); return;
        }
    }
    nn_project_mark_dirty(e->project);
    fit_view(e);
    set_status(e, "Nodes arranged left to right.");
}

static void handle_toolbar_click(Editor *e, float x, float y)
{
    if (y >= 62) return;
    leave_focus(e);
    if (inside(rect(186, 10, 88, 42), x, y)) perform_save(e);
    else if (inside(rect(284, 10, 88, 42), x, y)) {
        if (nn_project_dirty(e->project)) e->close_prompt = true;
        else {
            nn_project_close(e->project);
            e->project = NULL;
            set_status(e, "Project closed.");
        }
    } else if (inside(rect(382, 10, 76, 42), x, y)) fit_view(e);
    else if (inside(rect(468, 10, 106, 42), x, y)) arrange_graph(e);
}

static void handle_close_prompt(Editor *e, float x, float y)
{
    float left = e->width / 2.0f - 235, top = e->height / 2.0f - 95;
    if (inside(rect(left + 22, top + 112, 115, 42), x, y)) {
        char error[512] = "";
        if (!nn_project_save(e->project, error, sizeof(error))) {
            set_status(e, error); return;
        }
        nn_project_close(e->project); e->project = NULL;
        e->close_prompt = false;
        e->exit_requested = e->quit_pending;
        e->quit_pending = false;
    } else if (inside(rect(left + 154, top + 112, 126, 42), x, y)) {
        nn_project_close(e->project); e->project = NULL;
        e->close_prompt = false;
        e->exit_requested = e->quit_pending;
        e->quit_pending = false;
    } else if (inside(rect(left + 297, top + 112, 130, 42), x, y)) {
        e->close_prompt = false;
        e->quit_pending = false;
    }
}

static float segment_distance2(NNPlatformPoint p, NNPlatformPoint a, NNPlatformPoint b)
{
    float dx = b.x - a.x, dy = b.y - a.y;
    float denominator = dx * dx + dy * dy;
    float t = denominator > 0 ? ((p.x - a.x) * dx + (p.y - a.y) * dy) / denominator : 0;
    if (t < 0) t = 0;
    if (t > 1) t = 1;
    return distance2(p.x, p.y, a.x + t * dx, a.y + t * dy);
}

static void select_canvas_hit(Editor *e, float x, float y)
{
    NNModel *model = nn_project_model(e->project);
    for (size_t i = nn_model_node_count(model); i-- > 0;) {
        const NNNode *node = nn_model_node_at(model, i);
        const NNPackage *package = node_package(e, node);
        NNPlatformPoint output = handle_point(e, node, true);
        if (!is_output_kind(package) && distance2(x, y, output.x, output.y) < 196) {
            copy_text(e->draft_source, sizeof(e->draft_source), node->id);
            e->drag_edge = true;
            e->selected_edge[0] = '\0';
            return;
        }
        NNPlatformRect area = node_rect(e, node);
        if (inside(area, x, y)) {
            copy_text(e->selected, sizeof(e->selected), node->id);
            e->selected_edge[0] = '\0';
            e->drag_node = true;
            e->node_x = node->x;
            e->node_y = node->y;
            e->down_x = x;
            e->down_y = y;
            return;
        }
    }
    NNPlatformPoint pointer = {x, y};
    for (size_t i = 0; i < nn_model_edge_count(model); ++i) {
        const NNEdge *edge = nn_model_edge_at(model, i);
        const NNNode *source = nn_model_find_node(model, edge->source_id);
        const NNNode *target = nn_model_find_node(model, edge->target_id);
        if (!source || !target) continue;
        if (segment_distance2(pointer, handle_point(e, source, true),
                              handle_point(e, target, false)) < 64) {
            copy_text(e->selected_edge, sizeof(e->selected_edge), edge->id);
            e->selected[0] = '\0';
            return;
        }
    }
    e->selected[0] = e->selected_edge[0] = '\0';
    e->drag_pan = true;
    e->down_x = x; e->down_y = y;
    e->pan_x = e->camera_x; e->pan_y = e->camera_y;
}

static void handle_inspector_click(Editor *e, float x, float y)
{
    float right = (float)e->width - 320;
    if (x < right) return;
    if (inside(rect(right + 12, 72, 128, 38), x, y)) {
        leave_focus(e); e->resources_view = false; return;
    }
    if (inside(rect(right + 148, 72, 160, 38), x, y)) {
        leave_focus(e); e->resources_view = true; return;
    }
    if (e->resources_view) return;
    const NNNode *node = nn_model_find_node(nn_project_model(e->project), e->selected);
    if (!node) return;
    if (inside(rect(right + 20, 157, 280, 38), x, y)) {
        leave_focus(e);
        begin_edit(e, FIELD_NODE_NAME, -1);
        return;
    }
    const NNPackage *package = node_package(e, node);
    if (!package || y < 242 || y > 440) return;
    for (size_t i = 0; i < package->parameter_count; ++i) {
        if (inside(rect(right + 20, 270 + (float)i * 57, 280, 34), x, y)) {
            leave_focus(e);
            begin_edit(e, FIELD_PARAMETER, (int)i);
            return;
        }
    }
}

static void handle_palette_click(Editor *e, float x, float y)
{
    if (x >= 244 || y < 132) return;
    int row = (int)((y - 132) / 35) + e->palette_scroll;
    if (row < 0 || (size_t)row >= nn_catalog_count(nn_project_catalog(e->project))) return;
    leave_focus(e);
    add_package_node(e, nn_catalog_at(nn_project_catalog(e->project), (size_t)row));
}

static void finish_connection(Editor *e, float x, float y)
{
    if (!e->drag_edge) return;
    e->drag_edge = false;
    NNModel *model = nn_project_model(e->project);
    for (size_t i = nn_model_node_count(model); i-- > 0;) {
        const NNNode *node = nn_model_node_at(model, i);
        const NNPackage *package = node_package(e, node);
        if (is_input_kind(package)) continue;
        NNPlatformPoint input = handle_point(e, node, false);
        if (distance2(x, y, input.x, input.y) >= 196) continue;
        char edge_id[96], error[512] = "";
        snprintf(edge_id, sizeof(edge_id), "edge-%lu", ++e->sequence);
        const char *target_handle = package && package->kind && !strcmp(package->kind, "join")
            ? "in-1" : "in";
        if (nn_model_connect(model, edge_id, e->draft_source, "out",
                             node->id, target_handle, error, sizeof(error))) {
            nn_project_mark_dirty(e->project);
            set_status(e, "Connected nodes.");
        } else set_status(e, error);
        return;
    }
    set_status(e, "Connection cancelled.");
}

static void handle_mouse_up(Editor *e, float x, float y)
{
    if (e->drag_edge) finish_connection(e, x, y);
    if (e->drag_node) {
        e->drag_node = false;
        double new_x = e->node_x + (x - e->down_x) / e->zoom;
        double new_y = e->node_y + (y - e->down_y) / e->zoom;
        if (new_x != e->node_x || new_y != e->node_y) {
            char error[256] = "";
            if (nn_model_move_node(nn_project_model(e->project), e->selected,
                                   new_x, new_y, error, sizeof(error)))
                nn_project_mark_dirty(e->project);
            else set_status(e, error);
        }
    }
    e->drag_pan = false;
}

static void handle_key(Editor *e, const NNPlatformEvent *event)
{
    if (event->key == 27) {
        e->close_prompt = false;
        e->quit_pending = false;
        e->drag_edge = e->drag_node = e->drag_pan = false;
        leave_focus(e);
        return;
    }
    size_t capacity = 0;
    char *buffer = focus_buffer(e, &capacity);
    if (buffer) {
        size_t length = strlen(buffer);
        if (event->key == 8 && length) buffer[length - 1] = '\0';
        else if (event->key == 13) {
            if (e->focus == FIELD_OPEN) { e->focus = FIELD_NONE; perform_open(e); }
            else leave_focus(e);
        }
        return;
    }
    if (!e->project) return;
    if (event->key == 127 || event->key == 8) {
        char error[512] = "";
        NNModel *model = nn_project_model(e->project);
        if (e->selected[0] && nn_model_remove_node(model, e->selected,
                                                   error, sizeof(error))) {
            e->selected[0] = '\0';
            nn_project_mark_dirty(e->project);
            set_status(e, "Node deleted.");
        } else if (e->selected_edge[0] &&
                   nn_model_disconnect(model, e->selected_edge, error, sizeof(error))) {
            e->selected_edge[0] = '\0';
            nn_project_mark_dirty(e->project);
            set_status(e, "Connection deleted.");
        } else if (error[0]) set_status(e, error);
    }
}

static void handle_event(Editor *e, const NNPlatformEvent *event, bool *quit)
{
    if (event->type == NN_PLATFORM_EVENT_QUIT) {
        if (e->project && nn_project_dirty(e->project)) {
            e->close_prompt = true;
            e->quit_pending = true;
        } else *quit = true;
        return;
    }
    if (event->type == NN_PLATFORM_EVENT_RESIZE && event->width > 0 && event->height > 0) {
        e->width = event->width; e->height = event->height;
    }
    if (event->type == NN_PLATFORM_EVENT_TEXT) {
        size_t capacity = 0;
        char *buffer = focus_buffer(e, &capacity);
        if (buffer && strlen(buffer) + strlen(event->text) + 1 < capacity)
            strcat(buffer, event->text);
    }
    if (event->type == NN_PLATFORM_EVENT_KEY_DOWN) handle_key(e, event);
    if (event->type == NN_PLATFORM_EVENT_MOUSE_MOVE) {
        e->pointer_x = event->x; e->pointer_y = event->y;
        if (e->drag_pan) {
            e->camera_x = e->pan_x - (event->x - e->down_x) / e->zoom;
            e->camera_y = e->pan_y - (event->y - e->down_y) / e->zoom;
        }
    }
    if (event->type == NN_PLATFORM_EVENT_WHEEL && e->project) {
        if (e->pointer_x < 244) {
            int count = (int)nn_catalog_count(nn_project_catalog(e->project));
            e->palette_scroll -= (int)event->delta_y;
            if (e->palette_scroll < 0) e->palette_scroll = 0;
            if (e->palette_scroll > count - 1) e->palette_scroll = count > 0 ? count - 1 : 0;
        } else if (e->pointer_x >= e->width - 320 && e->resources_view) {
            int count = (int)nn_catalog_count(nn_project_catalog(e->project));
            e->resource_scroll -= (int)event->delta_y;
            if (e->resource_scroll < 0) e->resource_scroll = 0;
            if (e->resource_scroll > count - 1)
                e->resource_scroll = count > 0 ? count - 1 : 0;
        } else if (inside(viewport(e), e->pointer_x, e->pointer_y)) {
            NNPlatformPoint before = to_world(e, e->pointer_x, e->pointer_y);
            e->zoom *= event->delta_y > 0 ? 1.12 : 1.0 / 1.12;
            if (e->zoom < 0.25) e->zoom = 0.25;
            if (e->zoom > 3.0) e->zoom = 3.0;
            NNPlatformPoint after = to_world(e, e->pointer_x, e->pointer_y);
            e->camera_x += before.x - after.x;
            e->camera_y += before.y - after.y;
        }
    }
    if (event->type == NN_PLATFORM_EVENT_MOUSE_DOWN && event->button == 1) {
        float x = event->x, y = event->y;
        e->pointer_x = x; e->pointer_y = y;
        if (!e->project) handle_chooser_click(e, x, y);
        else if (e->close_prompt) handle_close_prompt(e, x, y);
        else if (y < 62) handle_toolbar_click(e, x, y);
        else if (x < 244) handle_palette_click(e, x, y);
        else if (x >= e->width - 320) handle_inspector_click(e, x, y);
        else if (inside(viewport(e), x, y)) {
            leave_focus(e);
            select_canvas_hit(e, x, y);
        }
    }
    if (event->type == NN_PLATFORM_EVENT_MOUSE_UP && event->button == 1 && e->project)
        handle_mouse_up(e, event->x, event->y);
    if (e->exit_requested) *quit = true;
}

int nn_editor_run(NNPlatform *platform, const char *core_root,
                  const char *initial_project_path)
{
    if (!platform || !core_root) return 1;
    Editor editor = {0};
    editor.platform = platform;
    editor.core_root = core_root;
    editor.width = 1400;
    editor.height = 900;
    editor.zoom = 1;
    editor.sequence = (unsigned long)time(NULL);
    if (!getcwd(editor.parent, sizeof(editor.parent)))
        copy_text(editor.parent, sizeof(editor.parent), ".");
    copy_text(editor.project_id, sizeof(editor.project_id), "mnist-mlp");
    copy_text(editor.project_name, sizeof(editor.project_name), "MNIST MLP");
    set_status(&editor, "Choose a project folder or create a new model.");
    if (initial_project_path && *initial_project_path) {
        copy_text(editor.open_path, sizeof(editor.open_path), initial_project_path);
        perform_open(&editor);
    }
    render(&editor);
    if (getenv("NN_UI_QA_CAPTURE") && getenv("NN_UI_QA_EXIT_AFTER_CAPTURE")) {
        nn_project_close(editor.project);
        return 0;
    }
    bool quit = false;
    while (!quit) {
        NNPlatformEvent event;
        if (!platform_wait_event(platform, &event)) {
            fprintf(stderr, "UI event error: %s\n", platform_error());
            nn_project_close(editor.project);
            return 1;
        }
        handle_event(&editor, &event, &quit);
        if (!quit) render(&editor);
    }
    nn_project_close(editor.project);
    return 0;
}
