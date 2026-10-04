#ifndef NN_APPLICATION_HISTORY_INTERNAL_H
#define NN_APPLICATION_HISTORY_INTERNAL_H

#include "model/model.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum { NN_APP_HISTORY_LIMIT = 100 };

typedef struct {
    NNModel *model;
    uint64_t revision;
} NNHistoryEntry;

typedef struct {
    NNHistoryEntry undo[NN_APP_HISTORY_LIMIT];
    NNHistoryEntry redo[NN_APP_HISTORY_LIMIT];
    size_t undo_count;
    size_t redo_count;
    NNModel *pending;
    uint64_t pending_revision;
    NNModel *group_start;
    uint64_t group_revision;
    uint64_t current_revision;
    uint64_t saved_revision;
    uint64_t next_revision;
    bool group_active;
    bool group_failed;
} NNEditHistory;

#endif
