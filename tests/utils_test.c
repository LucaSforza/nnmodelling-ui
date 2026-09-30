#include "utils.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    nn_error_set(NULL, 0, "ignored");
    nn_errorf(NULL, 0, "%s", "ignored");
    assert(!nn_fail(NULL, 0, "failure"));

    char tiny[2] = {'x', 'x'};
    nn_error_set(tiny, sizeof(tiny), "long message");
    assert(tiny[1] == '\0');
    nn_errorf(tiny, sizeof(tiny), "%s", "long message");
    assert(tiny[1] == '\0');
    assert(nn_fail(tiny, sizeof(tiny), "x") == false && !strcmp(tiny, "x"));

    char *copy = nn_text_copy("text");
    assert(copy && !strcmp(copy, "text"));
    free(copy);
    assert(!nn_text_copy(NULL));
    char *path = nn_path_join("left", "right");
    assert(path && !strcmp(path, "left/right"));
    free(path);
    assert(!nn_path_join(NULL, "right"));

    char *boundary = malloc(1024u * 1024u + 1);
    assert(boundary);
    memset(boundary, 'a', 1024u * 1024u - 1);
    boundary[1024u * 1024u - 1] = '\0';
    path = nn_path_join(boundary, "");
    assert(path && strlen(path) == 1024u * 1024u);
    free(path);
    boundary[1024u * 1024u - 1] = 'a';
    boundary[1024u * 1024u] = '\0';
    assert(!nn_path_join(boundary, ""));
    free(boundary);

    size_t order = 0;
    assert(nn_join_handle_order("in-12", &order) && order == 12);
    assert(!nn_join_handle_order("in-01", &order));
    assert(!nn_join_handle_order("in-0", &order));
    return 0;
}
