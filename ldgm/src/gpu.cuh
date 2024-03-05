#include <stdio.h>

// CUDA check error
#define cuda_check_error(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[LDGM GPU] [Error] %s (line %i): %s: %s.\n", \
                __FILE__, __LINE__, msg, cudaGetErrorString( err) ); \
            exit(-1); \
        } \
    } \

#ifdef __cplusplus
extern "C" {
#endif

void gpu_encode_upgrade (char* source_data,int *OUTBUF, int * PCM,int param_k,int param_m,int w_f,int packet_size ,int buf_size);

void gpu_decode_upgrade(char *data, int * PCM,int* SYNC_VEC,int* ERROR_VEC, int not_done, int *frame_size,int *, int*,int M,int K,int w_f,int buf_size,int packet_size);

#ifdef __cplusplus
}
#endif

