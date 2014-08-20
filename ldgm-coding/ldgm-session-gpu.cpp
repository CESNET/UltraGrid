/*
 * =====================================================================================
 *
 *       Filename:  ldgm-session-gpu.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  04/20/2012 02:21:11 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *        Authors:  Milan Kabat (kabat@ics.muni.cz), Vojtech David (374572@mail.muni.cz)
 *   Organization:  FI MUNI
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <time.h>
#include <set>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <emmintrin.h>
#include <iostream>
#include <fstream>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>

#include "ldgm-session-gpu.h"
#include "ldgm-session-cpu.h"
#include "gpu.cuh"
#include "timer-util.h"
#include "tanner.h"

#include "crypto/crc.h"

// static  int * error_vec;
// static  int * sync_vec;

// static int * ERROR_VEC;
// static int * SYNC_VEC;


// static int * PCM;


void *LDGM_session_gpu::alloc_buf (int buf_size)
{
    // printf("cudaHostAlloc %d %d\n",this->indexMemoryPool,index);
    // printf("buf_size %d, bufferSize %d\n",buf_size,this->memoryPoolSize[index] );
    while (!freeBuffers.empty()) {
            char *buf = freeBuffers.front();
            freeBuffers.pop();

            if (buf_size <= (int) bufferSizes[buf]) {
                    return buf;
            } else {
                    cudaFreeHost(buf);
                    cuda_check_error("cudaFreeHost");
                    bufferSizes.erase(buf);
            }
    }

    char *buf;
    cudaHostAlloc(&buf, buf_size, cudaHostAllocDefault);
    // cudaHostAlloc(&this->memoryPool[index], buf_size, cudaHostAllocMapped);
    cuda_check_error("cudaHostAlloc");
    bufferSizes[buf] = buf_size;

    return buf;
}

void LDGM_session_gpu::free_out_buf ( char *buf)
{
    if ( buf != NULL ) {
         freeBuffers.push(buf);
    }
}

void LDGM_session_gpu::encode ( char *source_data, char *parity )
{
    assert(parity == source_data+param_k*get_packet_size());

    int w_f = max_row_weight + 2;
    int buf_size = (param_k + param_m) * packet_size;

    cudaError_t error;

    // error = cudaHostGetDevicePointer( &(out_buf_d), source_data, 0);
    // cuda_check_error("out_buf_d");

    if(OUTBUF==NULL){
        // puts("cudaMalloc");
        cudaMalloc(&OUTBUF,buf_size);
        OUTBUF_SIZE=buf_size;
    }
    if(buf_size>OUTBUF_SIZE){
        // puts("cudaMalloc");
        cudaFree(OUTBUF);
        cudaMalloc(&OUTBUF,buf_size);
        OUTBUF_SIZE=buf_size;
    }

        cudaMemcpy(OUTBUF,source_data,buf_size,cudaMemcpyHostToDevice);
    cuda_check_error("memcpy OUTBUF");



    if (PCM == NULL)
    {   
        // puts("cudaMalloc");      
        error = cudaMalloc(&PCM, w_f * param_m * sizeof(int));
        if(error != cudaSuccess)printf("7CUDA error: %s\n", cudaGetErrorString(error));

        error = cudaMemcpy(PCM, pcm, w_f * param_m * sizeof(int), cudaMemcpyHostToDevice);
        if(error != cudaSuccess)printf("8CUDA error: %s\n", cudaGetErrorString(error));
    }





    gpu_encode_upgrade(source_data,OUTBUF , PCM, param_k, param_m, w_f, packet_size, buf_size);

    // puts("end");

}

char *LDGM_session_gpu::decode_frame ( char *received_data, int buf_size, int *frame_size, std::map<int, int> valid_data )
{
    char *received = received_data;

    struct timeval t0, t1;
    gettimeofday(&t0, 0);


    int p_size = buf_size / (param_m + param_k);
    // printf("%d p_size K: %d, M: %d, buf_size: %d, max_row_weight: %d \n",p_size,param_k,param_m,buf_size,max_row_weight);

    //We need to merge intervals in the valid data vector
    std::map <int, int> merged_intervals;
    std::map<int, int>::iterator map_it;

    if ( valid_data.size() != 0 )
    {
        for ( map_it = valid_data.begin(); map_it != valid_data.end(); )
        {
            int start = map_it->first;
            int length = map_it->second;
            while ( start + length == (++map_it)->first )
                length += map_it->second;
            merged_intervals.insert ( std::pair<int, int> (start, length) );
        }
    }

    

    cudaError_t error;
    if (error_vec == NULL)
    {
        cudaHostAlloc(&error_vec, sizeof(int) * (param_k + param_m), cudaHostAllocDefault );
        cuda_check_error("error_vec");
    }

    if (sync_vec == NULL)
    {
        cudaHostAlloc(&sync_vec, sizeof(int) * (param_k + param_m), cudaHostAllocDefault );
        cuda_check_error("sync_vec");
    }

    // error_vec= (int* ) malloc(sizeof(int)*(param_k+param_m));
    memset(error_vec, 0, sizeof(int) * (param_k + param_m));
    // sync_vec= (int* ) malloc(sizeof(int)*(param_k+param_m));
    memset(sync_vec, 0, sizeof(int) * (param_k + param_m));
    int not_done = 0;

    if ( merged_intervals.size() != 0
       )
    {

        for (int i = 0; i < param_k + param_m; i++)
        {
            int node_offset = i * p_size;

            map_it = merged_intervals.begin();

            bool found = false;
            while ( (map_it->first <= node_offset) && map_it != merged_intervals.end() )
            {
                map_it++;
                found = true;
            }
            if ( map_it->first > 0 ) map_it--;

            if ( found && (map_it->first + map_it->second) >=
                    (node_offset + p_size) )
            {
                //OK
                error_vec[i] = 0;
                sync_vec[i] = 0;
            }
            else
            {
                //NOK
                memset(received + (i * p_size), 0x0, p_size);
                error_vec[i] = 1;
                sync_vec[i] = 1;
                not_done++;
            }

        }

    }
    


    //-------------------------------------------------------------------------------------------------------





    /* 
     * struct coding_params params;
     * params.k = param_k;
     * params.m = param_m;
     * params.max_row_weight = max_row_weight;
     * params.packet_size = p_size;
     * params.buf_size = buf_size;
     */
     int w_f = max_row_weight + 2;


    // printf("K: %d, M: %d, max_row_weight: %d, buf_size: %d,\n", param_k, param_m, max_row_weight, buf_size );
    // printf("not done: %d\n", not_done);

    // int * sync_vec_d;
    // int * error_vec_d;
    // int * pcm_d;

    if (PCM == NULL)
    {
        error = cudaMalloc(&PCM, w_f * param_m * sizeof(int));
        if (error != cudaSuccess) printf("3 %s\n", cudaGetErrorString(error));

        error = cudaMemcpy(PCM, pcm, w_f * param_m * sizeof(int), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) printf("4 %s\n", cudaGetErrorString(error));
    }

    if (SYNC_VEC == NULL)
    {
        error = cudaMalloc(&SYNC_VEC, (param_m + param_k) * sizeof(int));
        if (error != cudaSuccess) printf("7 %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(SYNC_VEC, sync_vec, (param_m + param_k) * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error("SYNC_VEC");


    if (ERROR_VEC == NULL)
    {
        cudaMalloc(&ERROR_VEC, (param_m + param_k) * sizeof(int));
        cuda_check_error("ERROR_VEC");
    }

    cudaMemcpy(ERROR_VEC, error_vec, (param_m + param_k) * sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error("ERROR_VEC");

    // // }else{
    // //     error = cudaMemcpy(ERROR_VEC, sync_vec, (param_m + param_k) * sizeof(int), cudaMemcpyHostToDevice);
    // //     if (error != cudaSuccess) printf("8 %s\n", cudaGetErrorString(error));
    // // }

    //(char *data, int * PCM,int* SYNC_VEC,int* ERROR_VEC, int not_done, int *frame_size,int * error_vec,int * sync_vec,int M,int K,int w_f,int buf_size,int packet_size)
    // if (not_done != 0)
    // {
    //     gpu_decode_upgrade(received, PCM, SYNC_VEC, ERROR_VEC, not_done, frame_size, error_vec, sync_vec, param_m, param_k, w_f, buf_size, p_size);
    // }
    // else
    // {
    //     int fs = 0;
    //     memcpy(&fs, received, 4);
    //     *frame_size = fs;
    //     printf("frame_size: %d\n",frame_size );
    // }

    gpu_decode_upgrade(received, PCM, SYNC_VEC, ERROR_VEC, not_done, frame_size, error_vec, sync_vec, param_m, param_k, w_f, buf_size, p_size);

#if 0
     assert(LDGM_session::HEADER_SIZE >= 12)
     int my_frame_size = 0;
     memcpy(&my_frame_size, received + 8, 4);
        int crc_origin = 0;
         memcpy(&crc_origin, received + 4, 4);

       // printf("my_frame_size: %d\n",my_frame_size );
         int crc =0;
         crc = crc32buf(received + LDGM_session::HEADER_SIZE, my_frame_size);

         if (crc_origin != crc)
         {
             printf("CRC NOK\n");

         }
#endif

    // if (*frame_size != 0)
    // {

    //     int my_frame_size = 0;
    //     memcpy(&my_frame_size, received + 8, 4);
    //     int crc_origin = 0;
    //     memcpy(&crc_origin, received + 4, 4);

    //     // printf("my_frame_size: %d\n",my_frame_size );
    //     int crc =0;
    //     // crc32buf(received + 12, my_frame_size);

    //     if (crc_origin != crc)
    //     {
    //         printf("CRC NOK1\n");

    //     }
    // }
    // else
    // {
    //     printf("CRC NOK\n");
    // }

    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    //printf("time: %e\n",elapsed/1000.0 );
    this->elapsed_sum2 += elapsed / 1000.0;
    this->no_frames2++;

    return received + LDGM_session::HEADER_SIZE;

}


