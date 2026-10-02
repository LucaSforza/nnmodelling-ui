#include "cli_internal.h"
#include "yyjson.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void nn_cli_help(void)
{
    puts("Usage: nnmodelctl [--socket PATH] OPERATION [JSON_OBJECT|-]\n"
         "Socket defaults to NNMODELLING_SOCKET. Start UI with --socket PATH.\n"
         "Use - to read the argument object from stdin. Exit 0 success, 1 rejected, 2 transport/usage.\n"
         "project.open {path}; project.create {parent,id,name,template:blank|mnist-mlp|mnist-vae}\n"
         "project.save {}; project.close {discard:false}; project.snapshot {}\n"
         "analysis.diagnostics {} returns model/Lua problems and successful tensors\n"
         "stereotype.create {id,version,definition:{name,kind,view:{color,width,height},parameters:{},\n"
         "                   outputs:[{id:prediction,type:output},{id:objective,type:loss}]},\n"
         "                   inference:LuaSource,dependencies:{packageId:versionConstraint}}\n"
         "dataset.create {id,version,definition:{name,batch:{inputs:{slot:{dtype,shape}},targets:{}}},select:false}\n"
         "dataset.select {id,version}\n"
         "node.add {id,package,version,scope:\"\",x:0,y:0}; node.remove {id}; node.move {id,x,y}\n"
         "node.rename {id,name}; node.parameter {id,key,value:text}; node.boundary {id,handle}\n"
         "edge.connect {id,source,sourceHandle,target,targetHandle}; edge.disconnect {id}\n"
         "ui.inspect {}; ui.scope {id}; ui.reveal {id}; ui.arrange {}; ui.screenshot {path}\n"
         "See docs/knowledge/contracts/automation.md for complete payloads and transaction semantics.");
}

static char *stdin_json(void)
{
    char *text = malloc(NN_CLI_REQUEST_LIMIT + 1);
    if (!text) return NULL;
    size_t length = fread(text, 1, NN_CLI_REQUEST_LIMIT, stdin);
    if (ferror(stdin) || (length == NN_CLI_REQUEST_LIMIT && fgetc(stdin) != EOF) || memchr(text, '\0', length)) {
        free(text); return NULL;
    }
    text[length] = '\0';
    return text;
}

char *nn_cli_request(const char *operation, const char *payload, size_t *length)
{
    char *owned_payload = !strcmp(payload, "-") ? stdin_json() : NULL;
    if (!strcmp(payload, "-") && !owned_payload) { fputs("invalid or oversized stdin\n", stderr); return NULL; }
    if (owned_payload) payload = owned_payload;
    yyjson_doc *args = strlen(payload) <= NN_CLI_REQUEST_LIMIT ? yyjson_read(payload, strlen(payload), 0) : NULL;
    yyjson_mut_doc *request = yyjson_mut_doc_new(NULL);
    if (!args || !yyjson_is_obj(yyjson_doc_get_root(args)) || !request) {
        fputs("arguments must be a JSON object\n", stderr);
        yyjson_doc_free(args); yyjson_mut_doc_free(request); free(owned_payload); return NULL;
    }
    yyjson_mut_val *root = yyjson_mut_obj(request);
    yyjson_mut_val *arguments = yyjson_val_mut_copy(request, yyjson_doc_get_root(args));
    if (!root || !arguments || !yyjson_mut_obj_add_strcpy(request, root, "operation", operation) ||
        !yyjson_mut_obj_add_val(request, root, "args", arguments)) {
        fputs("out of memory building request\n", stderr);
        yyjson_doc_free(args); yyjson_mut_doc_free(request); free(owned_payload); return NULL;
    }
    yyjson_mut_doc_set_root(request, root);
    char *text = yyjson_mut_write(request, YYJSON_WRITE_NEWLINE_AT_END, length);
    yyjson_doc_free(args); yyjson_mut_doc_free(request); free(owned_payload);
    if (!text || *length > NN_CLI_REQUEST_LIMIT) { free(text); fputs("request too large\n", stderr); return NULL; }
    return text;
}
