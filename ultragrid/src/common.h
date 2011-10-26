#define error_with_code_msg(code, msg) {\
        fprintf(stderr, msg "\n");\
        exit(code);\
}

