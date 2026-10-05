#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L
#include "project_internal.h"
#include "utils/utils.h"

#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static bool copy_file(const char *source, const char *destination)
{
    struct stat st;
    if (lstat(source, &st) || !S_ISREG(st.st_mode)) return false;
    int input = open(source, O_RDONLY | O_NOFOLLOW);
    if (input < 0) return false;
    int output = open(destination, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, st.st_mode & 0777);
    bool okay = output >= 0;
    char buffer[16384];
    while (okay) {
        ssize_t count = read(input, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) { okay = false; break; }
        if (!count) break;
        okay = nn_project_write_all(output, buffer, (size_t)count);
    }
    if (output >= 0 && close(output)) okay = false;
    close(input);
    return okay;
}

static bool copy_tree(const char *source, const char *destination);

static bool copy_tree_contents(const char *source, const char *destination)
{
    struct stat st;
    if (lstat(source, &st) || !S_ISDIR(st.st_mode)) return false;
    DIR *dir = opendir(source);
    if (!dir) return false;
    bool okay = true;
    struct dirent *entry;
    while (okay && (entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char *from = nn_path_join(source, entry->d_name), *to = nn_path_join(destination, entry->d_name);
        if (!from || !to || lstat(from, &st)) okay = false;
        else if (S_ISDIR(st.st_mode)) okay = copy_tree(from, to);
        else if (S_ISREG(st.st_mode)) okay = copy_file(from, to);
        else okay = false;
        free(from); free(to);
    }
    closedir(dir);
    return okay;
}

static bool copy_tree(const char *source, const char *destination)
{
    struct stat st;
    if (lstat(source, &st) || !S_ISDIR(st.st_mode) ||
        mkdir(destination, st.st_mode & 0777)) return false;
    return copy_tree_contents(source, destination);
}

static char *repo_root_from_core(const char *core_root)
{
    char *path = realpath(core_root, NULL);
    if (!path) return NULL;
    for (int i = 0; i < 2; ++i) {
        char *slash = strrchr(path, '/');
        if (!slash || slash == path) { free(path); return NULL; }
        *slash = '\0';
    }
    return path;
}

NNProject *nn_project_create(const char *parent, const char *id, const char *name,
                             bool mnist_template, const char *core_root,
                             char *error, size_t capacity)
{
    if (error && capacity) error[0] = '\0';
    if (!parent || !nn_project_valid_id(id) || !name || !*name || !core_root) {
        nn_errorf(error, capacity, "invalid project identity or parent"); return NULL;
    }
    struct stat parent_stat;
    if (stat(parent, &parent_stat) || !S_ISDIR(parent_stat.st_mode)) {
        nn_errorf(error, capacity, "project parent directory unavailable"); return NULL;
    }
    char *directory = nn_path_join(parent, id);
    if (!directory) { nn_errorf(error, capacity, "out of memory"); return NULL; }
    if (mkdir(directory, 0755)) {
        nn_errorf(error, capacity, "cannot create project directory: %s", strerror(errno));
        free(directory); return NULL;
    }
    bool okay = true;
    if (mnist_template) {
        char *repo = repo_root_from_core(core_root);
        char *template_root = repo ? nn_path_join(repo, "examples/models/mnist-mlp") : NULL;
        char *source_model = template_root ? nn_path_join(template_root, "model.json") : NULL;
        char *destination_model = nn_path_join(directory, "model.json");
        char *source_dataset = template_root ? nn_path_join(template_root, "datasets/mnist") : NULL;
        char *destination_dataset = nn_path_join(directory, "datasets/mnist");
        char *destination_datasets = nn_path_join(directory, "datasets");
        okay = source_model && destination_model && source_dataset && destination_dataset &&
            destination_datasets && copy_file(source_model, destination_model) &&
            mkdir(destination_datasets, 0755) == 0 && copy_tree(source_dataset, destination_dataset);
        free(repo); free(template_root); free(source_model); free(destination_model);
        free(source_dataset); free(destination_dataset); free(destination_datasets);
    } else {
        NNProject seed = { .directory = directory, .id = (char *)id, .version = "0.1.0",
            .name = (char *)name, .description = "", .layout_direction = "horizontal",
            .model = nn_model_new() };
        okay = seed.model != NULL;
        char seed_error[128];
        if (okay) okay = nn_model_add_node(seed.model, "output", "Output",
            "core.output", "0.1.0", "", 320, 120, seed_error, sizeof(seed_error));
        if (okay) okay = nn_model_add_node(seed.model, "loss-output", "Loss Output",
            "core.loss-output", "0.1.0", "", 320, 240, seed_error, sizeof(seed_error));
        char *json = NULL; size_t length = 0;
        if (okay) okay = nn_project_write_project_document(&seed, &json, &length);
        char *path = nn_path_join(directory, "model.json");
        int fd = path && okay ? open(path, O_WRONLY | O_CREAT | O_EXCL, 0644) : -1;
        if (fd < 0) okay = false;
        if (okay) okay = nn_project_write_all(fd, json, length) && fsync(fd) == 0;
        if (fd >= 0 && close(fd)) okay = false;
        free(path); free(json); nn_model_free(seed.model);
    }
    if (!okay) {
        nn_errorf(error, capacity, "cannot create project from requested template");
        nn_project_remove_tree(directory); free(directory); return NULL;
    }
    NNProject *project = nn_project_open(directory, core_root, error, capacity);
    free(directory);
    if (!project) {
        /* Creation is one transaction: a bad template or package scope leaves no child. */
        char *failed = nn_path_join(parent, id);
        if (failed) { nn_project_remove_tree(failed); free(failed); }
        return NULL;
    }
    if (strcmp(project->id, id) || strcmp(project->name, name)) {
        char *new_id = nn_text_copy(id), *new_name = nn_text_copy(name);
        if (!new_id || !new_name) {
            free(new_id); free(new_name); nn_project_close(project);
            char *failed = nn_path_join(parent, id);
            if (failed) { nn_project_remove_tree(failed); free(failed); }
            nn_errorf(error, capacity, "out of memory"); return NULL;
        }
        free(project->id); free(project->name);
        project->id = new_id; project->name = new_name;
        project->dirty = true;
        if (!nn_project_save(project, error, capacity)) {
            nn_project_close(project);
            char *failed = nn_path_join(parent, id);
            if (failed) { nn_project_remove_tree(failed); free(failed); }
            return NULL;
        }
    }
    return project;
}

bool nn_project_create_vae(const char *parent, const char *id, const char *name,
                           const char *core_root, NNProject **result,
                           char *error, size_t cap)
{
    if (!result) { nn_errorf(error, cap, "invalid project result"); return false; }
    *result = NULL;
    if (!parent || !nn_project_valid_id(id) || !name || !*name || !core_root) { nn_errorf(error, cap, "invalid project identity"); return false; }
    struct stat st; if (lstat(parent, &st) || !S_ISDIR(st.st_mode)) { nn_errorf(error, cap, "project parent unavailable or symlinked"); return false; }
    char *root = repo_root_from_core(core_root), *template = root ? nn_path_join(root, "examples/models/mnist-vae") : NULL;
    char *destination = nn_path_join(parent, id);
    bool destination_owned = template && destination && mkdir(destination, 0755) == 0;
    bool okay = destination_owned && copy_tree_contents(template, destination);
    free(root); free(template);
    if (!okay) { if (destination_owned) nn_project_remove_tree(destination); free(destination); nn_errorf(error, cap, "cannot copy MNIST VAE template"); return false; }
    NNProject *project = nn_project_open(destination, core_root, error, cap);
    free(destination);
    if (!project) { char *failed = nn_path_join(parent, id); if (failed) { nn_project_remove_tree(failed); free(failed); } return false; }
    char *new_name = nn_text_copy(name), *new_id = nn_text_copy(id);
    if (!new_name || !new_id) { free(new_name); free(new_id); nn_project_close(project); char *failed = nn_path_join(parent, id); if (failed) { nn_project_remove_tree(failed); free(failed); } nn_errorf(error, cap, "out of memory"); return false; }
    free(project->name); free(project->id); project->name = new_name; project->id = new_id; project->dirty = true;
    if (!nn_project_save(project, error, cap)) { nn_project_close(project); char *failed = nn_path_join(parent, id); if (failed) { nn_project_remove_tree(failed); free(failed); } return false; }
    *result = project; return true;
}
