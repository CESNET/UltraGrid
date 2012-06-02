
void test(void);


struct coding_params {
    int num_lost;
    int k;
    int m;
    int packet_size;
    int max_row_weight;
};

void gpu_encode ( char*, char*, int*, struct coding_params );

void gpu_decode ( char*, int*, char*, struct coding_params );

