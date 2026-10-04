#include "application_internal.h"
#include "utils/utils.h"

#include <string.h>

static void clear_entries(NNHistoryEntry *entries, size_t *count)
{
    for (size_t i = 0; i < *count; ++i) nn_model_free(entries[i].model);
    *count = 0;
}

static void set_dirty(NNApplication *app)
{
    if (app && app->project)
        nn_project_set_dirty(app->project,
                             app->history.current_revision != app->history.saved_revision);
}

bool nn_app_history_fail(NNApplication *app, char *error, size_t capacity,
                         const char *message)
{
    if (app && app->history.group_active) app->history.group_failed = true;
    return nn_fail(error, capacity, message);
}

static void push_entry(NNHistoryEntry *entries, size_t *count, NNHistoryEntry entry)
{
    if (*count == NN_APP_HISTORY_LIMIT) {
        nn_model_free(entries[0].model);
        memmove(entries, entries + 1, (NN_APP_HISTORY_LIMIT - 1) * sizeof(*entries));
        --*count;
    }
    entries[(*count)++] = entry;
}

void nn_app_history_reset(NNApplication *app)
{
    if (!app) return;
    NNEditHistory *history = &app->history;
    clear_entries(history->undo, &history->undo_count);
    clear_entries(history->redo, &history->redo_count);
    nn_model_free(history->pending);
    nn_model_free(history->group_start);
    memset(history, 0, sizeof(*history));
    history->next_revision = 1;
}

void nn_app_history_barrier(NNApplication *app, bool saved)
{
    if (!app) return;
    NNEditHistory *history = &app->history;
    clear_entries(history->undo, &history->undo_count);
    clear_entries(history->redo, &history->redo_count);
    nn_model_free(history->pending);
    history->pending = NULL;
    nn_model_free(history->group_start);
    history->group_start = NULL;
    history->group_active = false;
    history->group_failed = false;
    if (saved) history->saved_revision = history->current_revision;
    else history->current_revision = history->next_revision++;
    set_dirty(app);
}

void nn_app_history_mark_saved(NNApplication *app)
{
    if (!app || !app->project) return;
    app->history.saved_revision = app->history.current_revision;
    set_dirty(app);
}

bool nn_app_history_prepare(NNApplication *app, char *error, size_t capacity)
{
    if (!app || !app->project) return nn_fail(error, capacity, "no active project");
    NNEditHistory *history = &app->history;
    if (history->group_active) {
        if (history->group_failed)
            return nn_fail(error, capacity, "edit group has failed and must be rolled back");
        if (!history->group_start) {
            history->group_start = nn_model_copy(nn_project_model(app->project));
            if (!history->group_start) {
                history->group_failed = true;
                return nn_fail(error, capacity, "out of memory recording graph edit group");
            }
            history->group_revision = history->current_revision;
        }
        return true;
    }
    if (history->pending)
        return nn_fail(error, capacity, "graph edit is already in progress");
    history->pending = nn_model_copy(nn_project_model(app->project));
    if (!history->pending)
        return nn_fail(error, capacity, "out of memory recording graph edit");
    history->pending_revision = history->current_revision;
    return true;
}

void nn_app_history_finish(NNApplication *app, bool success)
{
    if (!app || !app->project) return;
    NNEditHistory *history = &app->history;
    NNModel *model = nn_project_model(app->project);
    if (history->group_active) {
        if (!success) history->group_failed = true;
        return;
    }
    if (!history->pending) return;
    const bool changed = !nn_model_equal(model, history->pending);
    if (!success && changed) nn_model_swap(model, history->pending);
    if (success && changed) {
        push_entry(history->undo, &history->undo_count,
                   (NNHistoryEntry){history->pending, history->pending_revision});
        history->pending = NULL;
        clear_entries(history->redo, &history->redo_count);
        history->current_revision = history->next_revision++;
    } else {
        nn_model_free(history->pending);
        history->pending = NULL;
    }
    set_dirty(app);
}

bool nn_app_can_undo(const NNApplication *app)
{
    return app && app->project && !app->history.group_active && app->history.undo_count;
}

bool nn_app_can_redo(const NNApplication *app)
{
    return app && app->project && !app->history.group_active && app->history.redo_count;
}

bool nn_app_undo(NNApplication *app, char *error, size_t capacity)
{
    if (!app || !app->project) return nn_fail(error, capacity, "no active project");
    if (app->history.group_active)
        return nn_fail(error, capacity, "cannot undo during an edit group");
    if (!app->history.undo_count) return nn_fail(error, capacity, "nothing to undo");
    NNHistoryEntry entry = app->history.undo[--app->history.undo_count];
    const uint64_t current = app->history.current_revision;
    nn_model_swap(nn_project_model(app->project), entry.model);
    const uint64_t restored = entry.revision;
    entry.revision = current;
    push_entry(app->history.redo, &app->history.redo_count, entry);
    app->history.current_revision = restored;
    set_dirty(app);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, capacity, "");
    return true;
}

bool nn_app_redo(NNApplication *app, char *error, size_t capacity)
{
    if (!app || !app->project) return nn_fail(error, capacity, "no active project");
    if (app->history.group_active)
        return nn_fail(error, capacity, "cannot redo during an edit group");
    if (!app->history.redo_count) return nn_fail(error, capacity, "nothing to redo");
    NNHistoryEntry entry = app->history.redo[--app->history.redo_count];
    const uint64_t current = app->history.current_revision;
    nn_model_swap(nn_project_model(app->project), entry.model);
    const uint64_t restored = entry.revision;
    entry.revision = current;
    push_entry(app->history.undo, &app->history.undo_count, entry);
    app->history.current_revision = restored;
    set_dirty(app);
    nn_app_invalidate_analysis(app);
    nn_error_set(error, capacity, "");
    return true;
}

bool nn_app_begin_edit(NNApplication *app, char *error, size_t capacity)
{
    if (!app || !app->project) return nn_fail(error, capacity, "no active project");
    if (app->history.group_active || app->history.pending)
        return nn_fail(error, capacity, "an edit group is already active");
    app->history.group_active = true;
    app->history.group_failed = false;
    app->history.group_revision = app->history.current_revision;
    nn_error_set(error, capacity, "");
    return true;
}

static void rollback_group(NNApplication *app)
{
    NNEditHistory *history = &app->history;
    bool changed = false;
    if (history->group_start) {
        NNModel *model = nn_project_model(app->project);
        changed = !nn_model_equal(model, history->group_start);
        if (changed) nn_model_swap(model, history->group_start);
        nn_model_free(history->group_start);
        history->group_start = NULL;
    }
    history->current_revision = history->group_revision;
    history->group_active = false;
    history->group_failed = false;
    set_dirty(app);
    if (changed) nn_app_invalidate_analysis(app);
}

bool nn_app_end_edit(NNApplication *app, bool commit, char *error, size_t capacity)
{
    if (!app || !app->project) return nn_fail(error, capacity, "no active project");
    NNEditHistory *history = &app->history;
    if (!history->group_active) return nn_fail(error, capacity, "no edit group is active");
    if (!commit || history->group_failed) {
        const bool failed = history->group_failed;
        rollback_group(app);
        if (failed && commit) return nn_fail(error, capacity, "edit group failed and was rolled back");
        nn_error_set(error, capacity, "");
        return true;
    }
    if (history->group_start &&
        !nn_model_equal(nn_project_model(app->project), history->group_start)) {
        push_entry(history->undo, &history->undo_count,
                   (NNHistoryEntry){history->group_start, history->group_revision});
        history->group_start = NULL;
        clear_entries(history->redo, &history->redo_count);
        history->current_revision = history->next_revision++;
    }
    nn_model_free(history->group_start);
    history->group_start = NULL;
    history->group_active = false;
    history->group_failed = false;
    set_dirty(app);
    nn_error_set(error, capacity, "");
    return true;
}
