#define _POSIX_C_SOURCE 200809L
#include "automation.h"

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

static void dispatch(NNApplication *app, const char *request, bool success)
{
    char *response = nn_automation_dispatch(app, request, NULL, NULL);
    assert(response && strstr(response, success ? "\"ok\":true" : "\"ok\":false"));
    assert(response[strlen(response)-1] == '\n'); free(response);
}

static int connect_socket(const char *path)
{
    int fd = socket(AF_UNIX, SOCK_STREAM, 0); assert(fd >= 0);
    struct sockaddr_un address = { .sun_family = AF_UNIX };
    assert(strlen(path) < sizeof(address.sun_path)); strcpy(address.sun_path, path);
    assert(connect(fd, (struct sockaddr *)&address, sizeof(address)) == 0);
    assert(fcntl(fd, F_SETFL, O_NONBLOCK) == 0); return fd;
}

int main(void)
{
    NNApplication *app = nn_app_new("stereotype-packages/core"); assert(app);
    dispatch(app, "{bad", false);
    char *invalid = nn_automation_dispatch(app, "{bad", NULL, NULL);
    assert(invalid && strstr(invalid, "\"ok\":false") && strstr(invalid, "\"error\":\""));
    free(invalid);
    char *no_project = nn_automation_dispatch(app,
        "{\"operation\":\"analysis.diagnostics\",\"args\":{}}", NULL, NULL);
    assert(no_project && strstr(no_project, "\"ok\":false") &&
           strstr(no_project, "no active project"));
    free(no_project);
    dispatch(app, "{\"operation\":\"project.snapshot\",\"operation\":\"project.close\",\"args\":{}}", false);
    dispatch(app, "{\"operation\":\"project.snapshot\\u0000\",\"args\":{}}", false);
    dispatch(app, "{\"operation\":\"unknown\",\"args\":{}}", false);
    dispatch(app, "{\"operation\":\"project.snapshot\",\"args\":{}}", true);
    dispatch(app, "{\"operation\":\"ui.inspect\",\"args\":{}}", false);
    char root[] = "/tmp/opencode/nn-automation-XXXXXX"; assert(mkdtemp(root));
    char request[4096];
    snprintf(request, sizeof(request), "{\"operation\":\"project.create\",\"args\":{\"parent\":\"%s\",\"id\":\"test\",\"name\":\"Test\"}}", root);
    dispatch(app, request, true);
    char *before_analysis = nn_automation_dispatch(app,
        "{\"operation\":\"project.snapshot\",\"args\":{}}", NULL, NULL);
    char *report = nn_automation_dispatch(app,
        "{\"operation\":\"analysis.diagnostics\",\"args\":{}}", NULL, NULL);
    char *after_analysis = nn_automation_dispatch(app,
        "{\"operation\":\"project.snapshot\",\"args\":{}}", NULL, NULL);
    assert(before_analysis && report && after_analysis &&
           strstr(report, "\"available\":true") &&
           strstr(report, "\"problems\":[") && strstr(report, "\"tensors\":[") &&
           strstr(before_analysis, "\"dirty\":false") &&
           strstr(after_analysis, "\"dirty\":false"));
    free(before_analysis); free(report); free(after_analysis);
    dispatch(app, "{\"operation\":\"node.add\",\"args\":{\"id\":\"input\",\"package\":\"core.input\",\"version\":\"0.1.0\"}}", true);
    dispatch(app, "{\"operation\":\"node.add\",\"args\":{\"id\":\"relu\",\"package\":\"core.relu\",\"version\":\"0.1.0\"}}", true);
    dispatch(app, "{\"operation\":\"node.move\",\"args\":{\"id\":\"relu\",\"x\":\"bad\",\"y\":2}}", false);
    assert(nn_model_find_node(nn_app_model(app), "relu")->x == 0);
    dispatch(app, "{\"operation\":\"project.open\",\"args\":{\"path\":\"examples/mnist-vae\"}}", false);
    assert(nn_model_find_node(nn_app_model(app), "relu"));
    dispatch(app, "{\"operation\":\"edge.connect\",\"args\":{\"id\":\"edge\",\"source\":\"input\",\"sourceHandle\":\"out\",\"target\":\"relu\",\"targetHandle\":\"in\"}}", true);
    dispatch(app, "{\"operation\":\"node.parameter\",\"args\":{\"id\":\"input\",\"key\":\"binding\",\"value\":\"image\"}}", true);
    dispatch(app, "{\"operation\":\"node.parameter\",\"args\":{\"id\":\"input\",\"key\":\"binding\",\"value\":false}}", false);
    dispatch(app, "{\"operation\":\"project.close\",\"args\":{\"discard\":\"false\"}}", false);
    assert(nn_app_project(app));
    char *snapshot = nn_automation_dispatch(app, "{\"operation\":\"project.snapshot\",\"args\":{}}", NULL, NULL);
    assert(snapshot && strstr(snapshot, "\"parameters\":{\"binding\":\"image\"}") && strstr(snapshot, "\"edges\":[{")); free(snapshot);

    char socket_path[4096]; snprintf(socket_path, sizeof(socket_path), "%s/service.sock", root);
    char error[512] = "";
    NNAutomation *service = nn_automation_start(app, socket_path, NULL, NULL, error, sizeof(error));
    assert(service);
    struct stat info; assert(lstat(socket_path, &info) == 0 && (info.st_mode & 0777) == 0600);
    assert(!nn_automation_start(app, socket_path, NULL, NULL, error, sizeof(error)));
    int idle = connect_socket(socket_path), client = connect_socket(socket_path);
    const char *wire = "{\"operation\":\"project.snapshot\",\"args\":{}}\n";
    assert(write(client, wire, strlen(wire)) == (ssize_t)strlen(wire));
    char response[65536]; ssize_t count = -1;
    for (size_t i = 0; i < 100 && count < 0; ++i) {
        nn_automation_poll(service);
        count = read(client, response, sizeof(response)-1);
    }
    assert(count > 0); response[count] = '\0'; assert(strstr(response, "\"ok\":true"));
    close(client); close(idle);
    nn_automation_stop(service); assert(access(socket_path, F_OK) != 0);

    /* Never remove a path we did not create. */
    FILE *file = fopen(socket_path, "wb"); assert(file && fclose(file) == 0);
    assert(!nn_automation_start(app, socket_path, NULL, NULL, error, sizeof(error)));
    assert(lstat(socket_path, &info) == 0 && S_ISREG(info.st_mode)); assert(unlink(socket_path) == 0);
    dispatch(app, "{\"operation\":\"project.close\",\"args\":{\"discard\":true}}", true);
    char model[4096], project[4096];
    snprintf(model, sizeof(model), "%s/test/model.json", root); assert(unlink(model) == 0);
    snprintf(project, sizeof(project), "%s/test", root); assert(rmdir(project) == 0 && rmdir(root) == 0);
    nn_app_free(app); puts("local command dispatch and socket: ok"); return 0;
}
