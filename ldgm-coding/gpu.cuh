struct coding_params {
    int num_lost;
    int k;
    int m;
    int packet_size;
    int max_row_weight;
};

void gpu_encode ( char* source_data,int* pc_matrix, struct coding_params * );

void gpu_encode_upgrade (char* source_data,int *OUTBUF, int * PCM,int param_k,int param_m,int w_f,int packet_size ,int buf_size);

void gpu_decode (char * received,int * pcm,struct coding_params * params,int * error_vec,int * sync_vec,int undecoded,int * frame_size);

void gpu_decode_upgrade(char *data, int * PCM,int* SYNC_VEC,int* ERROR_VEC, int not_done, int *frame_size,int *, int*,int M,int K,int w_f,int buf_size,int packet_size);

__global__ void frame_encode(char * data,int * pcm,struct coding_params * params);

__global__ void frame_encode_int_big(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size);

__global__ void frame_encode_staircase(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size);

__global__ void frame_decode(char * received, int * pcm, int * error_vec,int * sync_vec,int packet_size,int max_row_weight,int K);

__global__ void frame_encode_int(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size);

__global__ void frame_decode_int(int * received, int * pcm, int * error_vec,int * sync_vec,int packet_size,int max_row_weight,int K);
