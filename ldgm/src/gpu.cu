/*
 * =====================================================================================
 *
 *       Filename:  hello.cu
 *
 *    Description:  CUDA test
 *
 *        Version:  1.0
 *        Created:  02/06/2012 03:54:42 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *        Authors:  Milan Kabat (kabat@ics.muni.cz), Vojtech David (374572@mail.muni.cz)
 *        Company:  FI MUNI
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <signal.h>
#include <emmintrin.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include "timer-util.h"
#include "gpu.cuh"

struct coding_params {
        int num_lost;
        int k;
        int m;
        int packet_size;
        int max_row_weight;
};

__global__ void frame_encode(char * data,int * pcm,struct coding_params * params);

__global__ void frame_encode_int_big(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size);

__global__ void frame_encode_staircase(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size);

__global__ void frame_decode(char * received, int * pcm, int * error_vec,int * sync_vec,int packet_size,int max_row_weight,int K);

__global__ void frame_encode_int(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size);

__global__ void frame_decode_int(int * received, int * pcm, int * error_vec,int * sync_vec,int packet_size,int max_row_weight,int K);

void gpu_encode ( char* source_data,int* pc_matrix, struct coding_params * );

void gpu_decode (char * received,int * pcm,struct coding_params * params,int * error_vec,int * sync_vec,int undecoded,int * frame_size);

__device__ unsigned int count = 0;
__device__ unsigned int count_M = 0;
char *xor_using_sse2 (char *source, char *dest, int packet_size)
{
    //First, do as many 128-bit XORs as possible
    int iter_bytes_16 = 0;
    int iter_bytes_4 = 0;
    int iter_bytes_1 = 0;

    iter_bytes_16 = (packet_size / 16) * 16;

    if ( iter_bytes_16 > 0)
    {

        //    printf ( "iter_bytes: %d\n", iter_bytes );
        __m128i *wrd_ptr = (__m128i *) source;
        __m128i *wrd_end = (__m128i *) (source + iter_bytes_16);
        __m128i *dst_ptr = (__m128i *) dest;

        //    printf ( "wrd_ptr address: %p\n", wrd_ptr );
        do
        {
            __m128i xmm1 = _mm_loadu_si128(wrd_ptr);
            __m128i xmm2 = _mm_loadu_si128(dst_ptr);

            xmm1 = _mm_xor_si128(xmm1, xmm2);     //  XOR  4 32-bit words
            _mm_storeu_si128(dst_ptr, xmm1);
            ++wrd_ptr;
            ++dst_ptr;

        }
        while (wrd_ptr < wrd_end);
    }
    //Check, whether further XORing is necessary
    if ( iter_bytes_16 < packet_size )
    {
        char *mark_source = source + iter_bytes_16;
        char *mark_dest = dest + iter_bytes_16;

        iter_bytes_4 = ((packet_size - iter_bytes_16) / 4) * 4;

        for ( int i = 0; i < (packet_size - iter_bytes_16) / 4; i++)
        {
            int *s = ((int *) mark_source) + i;
            int *d = ((int *) mark_dest) + i;
            *d ^= *s;
        }

        mark_source += iter_bytes_4;
        mark_dest += iter_bytes_4;

        iter_bytes_1 = packet_size - iter_bytes_16 - iter_bytes_4;

        for ( int i = 0; i < iter_bytes_1; i++)
        {
            *(mark_dest + i) ^= *(mark_source + i);
        }
    }

    return dest;
}

CUDA_DLL_API void gpu_encode_upgrade (char * source_data,int *OUTBUF, int * PCM,int param_k,int param_m,int w_f,int packet_size ,int buf_size)
{

    // cudaError_t error;
    int blocksize = packet_size/sizeof(int);
    // printf("blocksize: %d\npacket size: %d\n",blocksize,packet_size );
    if(blocksize>256){
        if(blocksize>1024)  blocksize=1024;
        // puts("big one");
        frame_encode_int_big <<< param_m, blocksize, packet_size >>> (OUTBUF,PCM, param_k, param_m, w_f, packet_size);
        cuda_check_error("frame_encode_int_big");

        frame_encode_staircase<<< 1, blocksize, packet_size >>> (OUTBUF, PCM, param_k, param_m, w_f, packet_size);
        cuda_check_error("frame_encode_staircase");

        cudaMemcpy(source_data + param_k*packet_size,OUTBUF + (param_k*packet_size)/4, param_m*packet_size,cudaMemcpyDeviceToHost );
        // // cudaMemcpy(source_data,OUTBUF, buf_size,cudaMemcpyDeviceToHost );
        cuda_check_error("memcpy out_buf");


        //     gettimeofday(&t0, 0);
        // for ( int m = 1; m < param_m; ++m)
        // {
        //     char *prev_parity = (char *) source_data + (param_k + m - 1) * packet_size;
        //     char *parity_packet = (char *) source_data + (param_k + m) * packet_size;
        //     xor_using_sse2(prev_parity, parity_packet, packet_size);

        // }
        // gettimeofday(&t1, 0);
        // long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
        // printf("time staircase: %f\n",elapsed/1000.0 );

    }
    else{
        // puts("chudy soused");
        frame_encode_int <<< param_m, blocksize, packet_size >>> (OUTBUF,PCM, param_k, param_m, w_f, packet_size);
        cuda_check_error("frame_encode_int");

        cudaMemcpy(source_data + param_k*packet_size,OUTBUF + (param_k*packet_size)/4, param_m*packet_size,cudaMemcpyDeviceToHost );
        // cudaMemcpy(source_data,OUTBUF, buf_size,cudaMemcpyDeviceToHost );
        cuda_check_error("memcpyu out_buf");

        for ( int m = 1; m < param_m; ++m)
        {
            char *prev_parity = (char *) source_data + (param_k + m - 1) * packet_size;
            char *parity_packet = (char *) source_data + (param_k + m) * packet_size;
            xor_using_sse2(prev_parity, parity_packet, packet_size);

        }

    }



    // cudaEvent_t start, stop;
    // float time;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start, 0);

    // cudaStream_t pStream;
    // cudaStreamCreate(&pStream);
    // frame_encode_staircase<<< 1, blocksize, packet_size >>> (OUTBUF, PCM, param_k, param_m, w_f, packet_size);
    // cuda_check_error("frame_encode_staircase");


    
    // cudaStreamSynchronize();
    // cudaDeviceSynchronize();

    return;
}

#define CHECK_CUDA(cmd) do { \
        cudaError_t err = cmd; \
        if (err != cudaSuccess) {\
                fprintf(stderr, "[LDGM GPU] %s: %s\n", #cmd, cudaGetErrorString(err)); \
        } \
} while(0)

CUDA_DLL_API void gpu_decode_upgrade(char *data, int * PCM,int* SYNC_VEC,int* ERROR_VEC, int not_done, int *frame_size,int * error_vec,int * sync_vec,int M,int K,int w_f,int buf_size,int packet_size)
{


    cudaError_t error;
    int *received_d;


    // int M = params->m;
    // int K = params->k;
    // int w_f = params->max_row_weight + 2;
    // int buf_size = params->buf_size;

    int* received = (int*) data;
    // printf("K: %d, M: %d, max_row_weight: %d, buf_size: %d,\n packet_size: %d\n",K,M,w_f,buf_size,packet_size );
    // printf("NOT DONE: %d\n",not_done );


    error = cudaHostRegister(received, buf_size, cudaHostRegisterMapped);
    if (error != cudaSuccess) printf("1 %s\n", cudaGetErrorString(error));

    error = cudaHostGetDevicePointer((void **) & (received_d), (void *)received, 0);
    if (error != cudaSuccess) printf("2 %s\n", cudaGetErrorString(error));

    // error = cudaMalloc(&received_d, buf_size);
    // if (error != cudaSuccess) printf("1 %s\n", cudaGetErrorString(error));

    // error = cudaMemcpy(received_d, received, buf_size, cudaMemcpyHostToDevice);
    // if (error != cudaSuccess) printf("2 %s\n", cudaGetErrorString(error));  

    // error = cudaMalloc(&pcm_d, w_f * M * sizeof(int));
    // if (error != cudaSuccess) printf("3 %s\n", cudaGetErrorString(error));

    // error = cudaMemcpy(pcm_d, PCM, w_f * M * sizeof(int), cudaMemcpyHostToDevice);
    // if (error != cudaSuccess) printf("4 %s\n", cudaGetErrorString(error));


    // error = cudaMalloc(&error_vec_d, (K + M) * sizeof(int));
    // if (error != cudaSuccess)printf("5 %s\n", cudaGetErrorString(error));

    // error = cudaMemcpy(error_vec_d, error_vec, (K + M) * sizeof(int), cudaMemcpyHostToDevice);
    // if (error != cudaSuccess) printf("6 %s\n", cudaGetErrorString(error));

    // error = cudaMalloc(&sync_vec_d, (K + M) * sizeof(int));
    // if (error != cudaSuccess) printf("7 %s\n", cudaGetErrorString(error));

    // error = cudaMemcpy(sync_vec_d, sync_vec, (K + M) * sizeof(int), cudaMemcpyHostToDevice);
    // if (error != cudaSuccess) printf("8 %s\n", cudaGetErrorString(error));

    int ps = packet_size/sizeof(int);

    int blocksize = packet_size/sizeof(int) +1;
    // printf("blocksize: %d\npacket size: %d\n",blocksize,packet_size );
    if(blocksize>512) blocksize=512;

    int not_done_source=0;
    for (int i = 0; i < K; i++)
    {
        if (error_vec[i] == 1) not_done_source++;
    }
    // printf("not_done %d\n",not_done );
    // printf("not_done_source %d\n",not_done_source);


    unsigned int count_host = 0;
    unsigned int count_host_M = 0;

    CHECK_CUDA(cudaMemcpyToSymbol  ( count, (void *)(&count_host), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol  ( count_M, (void *)(&count_host_M), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));


    int i = 0;
    for (i = 1; i < 30; ++i)
    {
        //__global__ void frame_decode_int(int * received, int * pcm, int * error_vec,int * sync_vec,int packet_size,int max_row_weight,int K);
        frame_decode_int <<< M, blocksize , packet_size >>> (received_d, PCM, ERROR_VEC, SYNC_VEC, ps, w_f-2, K);
        error = cudaGetLastError();
        if (error != cudaSuccess) printf("3 %s\n", cudaGetErrorString(error));

        // cudaDeviceSynchronize();
        //error = cudaMemcpyFromSymbol((void *)(&count_host), count, sizeof(int), 0, cudaMemcpyDeviceToHost);
        count_host=0;
        error = cudaMemcpyFromSymbol((void*)(&count_host), count, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) printf("10 %s\n", cudaGetErrorString(error));
        // printf("count host %d\n",count_host );

        if (count_host == not_done_source)
        {
            break;
        }


        CHECK_CUDA(cudaMemcpyFromSymbol((void*)(&count_host_M), count_M, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));
        // printf("count host_M %d\n",count_host_M );
        if (count_host_M == M)
        {
            break;
        }

        count_host_M = 0;
        CHECK_CUDA(cudaMemcpyToSymbol  ( count_M, (void *)(&count_host_M), sizeof(unsigned int), 0, cudaMemcpyHostToDevice));


    }
    // printf("iterace: %d\n",i);


    // cudaDeviceSynchronize();
    //cudaThreadSynchronize();

    CHECK_CUDA(cudaMemcpy(error_vec, ERROR_VEC, (K + M) * sizeof(int), cudaMemcpyDeviceToHost));


    int a = 0;
    int fs = 0;
    for (int i = 0; i < K; i++)
    {
        if (error_vec[i] == 1) a++;
    }

    // printf("UNDECODED: %d  NOT DONE: %d DEKODOVANO: %d\n",a,not_done,not_done-a);
    if (a != 0)
    {
        *frame_size = 0;

    }
    else
    {
        memcpy(&fs, received, 4);
        // printf("received size %d\n",fs );
        *frame_size = fs;
    }
    // printf("undecoded: %d, frame_size: %d, undecoded subtract: %d\n",a,fs,not_done-a );

    CHECK_CUDA(cudaHostUnregister(received));
    // cudaFree(received_d);
    // cudaFree(pcm_d);
    // cudaFree(error_vec_d);
    // cudaFree(sync_vec_d);
    // cudaFree(params_d);


    // puts("END");
    return;

}
__global__ void frame_encode_int_big(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size)
{
    int ps = packet_size/sizeof(int);

    int bx = blockIdx.x;
    int x  = threadIdx.x;
    int offset;


    // printf("K: %d M: %d max_row_weight: %d packet_size: %d\n",param_k,param_m,max_row_weight,ps);
    extern __shared__ int parity_packet[];
    // int *parity_packet = data + (param_k + bx) * ps;

    // if(x==0)printf("bx %d has parity packet at: %d,%d\n",bx,param_k*ps + bx*ps,param_k+bx );


    offset = x;
    while (offset < ps)
    {
        parity_packet[offset]=0;
        offset += blockDim.x;
    }
   // __syncthreads();

    for ( int i = 0; i < w_f; i++)
    {
        int idx = pcm[bx * w_f + i];
        //printf ( "adept: %d\n", idx );

        // if(x==0) printf ("block %d xor packet: %d\n",bx,idx);
        if (idx > -1 && idx < param_k)
        {

            //xoring parity_packet ^ idx
            offset = x;
            while (offset < ps)
            {
                parity_packet[offset]^=data[idx*ps + offset];
                offset += blockDim.x;
            }

        }
    }
    // __syncthreads();
    offset = x;
    while (offset < ps)
    {
        data[(param_k + bx) * ps + offset]= parity_packet[offset];
        offset += blockDim.x;
    }


}

__global__ void frame_encode_int(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size)
{
    int ps = packet_size/sizeof(int);

    int bx = blockIdx.x;
    int offset  = threadIdx.x;



    // printf("K: %d M: %d max_row_weight: %d packet_size: %d\n",param_k,param_m,max_row_weight,ps);
    extern __shared__ int parity_packet[];
    // int *parity_packet = data + (param_k + bx) * ps;

    // if(x==0)printf("bx %d has parity packet at: %d,%d\n",bx,param_k*ps + bx*ps,param_k+bx );


    // while (offset < ps)
    // {
    //     parity_packet[offset]=0;
    //     offset += blockDim.x;
    // }
    parity_packet[offset]=0;
   // __syncthreads();

    for ( int i = 0; i < w_f; i++)
    {
        int idx = pcm[bx * w_f + i];
        //printf ( "adept: %d\n", idx );

        // if(x==0) printf ("block %d xor packet: %d\n",bx,idx);
        if (idx > -1 && idx < param_k)
        {

            //xoring parity_packet ^ idx
          //   offset = x;
          //   while (offset < ps)
          //   {
          //       parity_packet[offset]^=data[idx*ps + offset];
		        // offset += blockDim.x;
          //   }
            parity_packet[offset]^=data[idx*ps + offset];

        }
    }
    // __syncthreads();
    // offset = x;
    // while (offset < ps)
    // {
    //     data[(param_k + bx) * ps + offset]= parity_packet[offset];
    //     offset += blockDim.x;
    // }

    data[(param_k + bx) * ps + offset]= parity_packet[offset];
    // __syncthreads();

}


__global__ void frame_decode_int(int *received, int *pcm, int *error_vec, int *sync_vec, int packet_size, int max_row_weight, int K)
{
    //TITAN

    __shared__ int undecoded;
    __shared__ int undecoded_index;
    __shared__ int ret;


    extern __shared__ int shared_parity_packet[];
    int w_f = max_row_weight + 2;
    int ps = packet_size;

    int bx = blockIdx.x;
    int x = threadIdx.x;

    int offset = 0;

    if (x == 0)
    {
        ret = 0;
        undecoded = 0;
        undecoded_index = -1;
        for (int j = 0; j < w_f; j++)
        {
            int p = pcm[bx * w_f + j];
            //printf("%d %d %d\n",p, error_vec[p],x);
            if (p != -1 && error_vec[p] == 1)
            {
                undecoded++;
                undecoded_index = p;
            }
        }
        if (undecoded == 1)
        {
            ret = atomicCAS(sync_vec + undecoded_index, 1, 0);
        }

    }
    __syncthreads();
    if (ret == 1)
    {

        // if(x==0) printf("decoding %7d, bx %7d\n",undecoded_index,bx );
        offset = x;

        while (offset < ps)
        {
            shared_parity_packet[offset]=0x0;
            offset += blockDim.x;
        }
        /*int zbyva = ps - offset;
        if (x < zbyva)
        {
            shared_parity_packet[x + offset] = 0;
        }*/

        __syncthreads();
        // if(x==0) printf("decoding [%d]\n",undecoded_index);
        for (int j = 0; j < w_f; j++)
        {
            int index = pcm[bx * w_f + j];
            if (index != undecoded_index && index != -1)
            {
                offset = x;
                while ( offset < ps)
                {
                    shared_parity_packet[offset] ^= received[index*ps + offset];
                    offset += blockDim.x;
                }/*
                int zbyva = ps - offset;
                if (x < zbyva)
                {
                    shared_parity_packet[x + offset] ^= received[(index * ps) + x + offset];
                }*/


            }

        }
        __syncthreads();
        offset = x; 

        while ( offset < ps)
        {
            // *((int *)(received + (undecoded_index * ps) + 4*x + a)) = *((int *)(shared_parity_packet + a + 4 * x));
            received[(undecoded_index * ps) + offset] = shared_parity_packet[offset];
            offset += blockDim.x;
        }
/*
        zbyva = ps - offset;
        if (x < zbyva)
        {
            received[(undecoded_index * ps) + x + offset] = shared_parity_packet[x + offset];
        }*/


    }
    if (x == 0 && ret == 1)
    {
        //error_vec[undecoded_index]=0;
        atomicCAS(error_vec + undecoded_index, 1, 0);
        // printf("node %d %d done\n",undecoded_index);
    }
    if (x == 0 && ret==1 && undecoded_index<K)
    {
        atomicAdd(&count, 1);
    }
    if (x == 0 && undecoded!=1 )
    {
        atomicAdd(&count_M, 1);
    }




}


__global__ void frame_encode_staircase(int *data, int *pcm,int param_k,int param_m,int w_f,int packet_size)
{
    int ps = packet_size/sizeof(int);

    int x  = threadIdx.x;

    for (int index = param_k; index < param_k + param_m-1; index++)
    {

        int offset = x;

        while (offset < ps)
        {
            // *((int *)(data + (index+1)*ps + offset + intSize * x)) ^= *((int *)(data + index * ps + intSize * x + offset));
            data[(index+1)*ps + offset] ^= data[index*ps + offset];
            offset += blockDim.x;
        }



    }

 
}
