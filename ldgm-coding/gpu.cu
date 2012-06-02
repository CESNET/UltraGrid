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
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *        Company:  FI MUNI
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <cuda.h>

#include "timer-util.h"
#include "gpu.cuh"

#define  INTS_PER_THREAD 4

void make_compact_pcm ( char* pc_matrix, int* pcm, struct coding_params params );

void gpu_encode ( char* source_data, char* parity, int* pc_matrix, struct coding_params );

void gpu_decode ( char* received, int* error_vec, char* pc_matrix, struct coding_params );

__global__ void encode ( int* data, int* parity, int* pcm, struct coding_params params )
{

    //partial result
    extern __shared__ int shared_mem[];
//    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int num_packets = params.max_row_weight;
    int p_size = params.packet_size/4;

    //in each iteration, we copy corresponding packet to shared memory and XOR its content
    //into partial result
    for ( int i = 0; i < num_packets; i++ ) 
    { 
	int index = pcm [ blockIdx.x*(num_packets+2) + i ];
	
	if ( index > -1 && index < params.k )
 	{
	    for ( int j = 0; j < INTS_PER_THREAD; j++ )
 	    {
		shared_mem [ threadIdx.x*INTS_PER_THREAD + j ] ^= 
		    data [ index*p_size + threadIdx.x*INTS_PER_THREAD  + j ];
	    }
	}
    }

    for ( int j = 0; j < INTS_PER_THREAD; j++ )
     {
	parity [ blockIdx.x*p_size + threadIdx.x*INTS_PER_THREAD + j ] = 
	    shared_mem [ threadIdx.x*INTS_PER_THREAD + j];
	shared_mem [ threadIdx.x*INTS_PER_THREAD + j] = 0;
    }

    __syncthreads();
    
} 

__global__ void decode ( int* received, int* error_vec, int* pcm, struct coding_params params )
{
/*     extern __shared__ int shared_mem[];
 * 
 *     int num_neighbours = params.max_row_weight + 2;
 * 
 *     int p_size = params.packet_size/sizeof(int);
 *     int num_threads = p_size/4;
 * 
 *     //load the neighbouring packets into shared memory
 *     int idx = blockIdx.x*blockDim.x + threadIdx.x;
 *     for ( int i = 0; i < num_beighbours; i++ )
 *     {
 * 	if ( threadIdx.x == 0 )
 * 	    shared_mem[i] =  pcm [ blockIdx.x*num_beighbours + i ];
 * 
 * 	__syncthreads();
 * 	
 * 	if ( shared_mem[i] != -1 ) 
 * 	{
 * //	    shared_mem [ num_neighbours + threadIdx.x*4 + i ] =
 * 
 * 
 * 
 * 
 * 	}
 *     }
 * 
 *     __syncthreads();
 *     for ( int i = 0; i < (params.packet_size/sizeof(int))/num_threads; i++ )
 * 	received [ (params.k + blockIdx.x)*p_size + threadIdx.x*4 + i] = 
 * 	    pkts [ idx + i ];
 *    
 *     __syncthreads();
 */

//    for ( int i = 0; i < params.packet_size/sizeof(int); i++ )
//	received [ params.k*p_size + blockIdx.x*p_size + idx + i ] = pkts [ idx + i ];



}

void gpu_encode ( char* source_data, char* parity, int* pcm, struct coding_params params ) 
{
    int* src_data_d;
    int* parity_data_d;
    int* pcm_d;
    short show_info = 0;

    cudaError_t cuda_error;

    struct cudaDeviceProp dev_prop;

    cudaGetDeviceProperties (&dev_prop, 0);

    if (show_info)
    {
	if (!dev_prop.canMapHostMemory)
	    printf("Cannot map host memory.\n");
	printf ( "name: %s\n", dev_prop.name );
	printf ( "totalGlobalMem: %d MB\n", (unsigned int)dev_prop.totalGlobalMem/(1024*1024) );
	printf ( "sharedMemPerBlock: %d kB\n", (unsigned int)dev_prop.sharedMemPerBlock/1024 );
	printf ( "maxThreadsPerBlock: %d\n", dev_prop.maxThreadsPerBlock );
	printf ( "maxThreadsDim: %d\n", dev_prop.maxThreadsDim[0] );
	printf ( "maxThreadsDim: %d\n", dev_prop.maxThreadsDim[1] );
	printf ( "maxThreadsDim: %d\n", dev_prop.maxThreadsDim[2] );
	printf ( "maxGridSize: %d\n", dev_prop.maxGridSize[0] );
    }

//    pcm = (int*) malloc (params.m*(params.max_row_weight+2)*sizeof(int*));

//    make_compact_pcm ( pc_matrix, pcm,  params );

    cuda_error = cudaMalloc ( (void**) &src_data_d, params.k*params.packet_size);
    if ( cuda_error != cudaSuccess )
	printf ( "cudaMalloc returned %d\n", cuda_error );

    cuda_error = cudaMalloc ( (void**) &parity_data_d, params.m*params.packet_size);
    if ( cuda_error != cudaSuccess )
	printf ( "cudaMalloc returned %d\n", cuda_error );

    cuda_error = cudaMemset ( parity_data_d, 0, params.m*params.packet_size);
    if ( cuda_error != cudaSuccess )
	printf ( "cudaMemset returned %d\n", cuda_error );

    cuda_error = cudaMemcpy ( src_data_d, source_data, params.k*params.packet_size, 
	    cudaMemcpyHostToDevice );
    if ( cuda_error != cudaSuccess )
	printf ( "cudaMemcpy returned %d\n", cuda_error );

    cuda_error = cudaMalloc ( (void**) &pcm_d, params.m*(params.max_row_weight+2)*sizeof(int));
    if ( cuda_error != cudaSuccess )
	printf ( "cudaMalloc return %d\n", cuda_error );

    cuda_error = cudaMemcpy ( pcm_d, pcm, sizeof(int)*params.m*(params.max_row_weight+2), 
	    cudaMemcpyHostToDevice );
    if ( cuda_error != cudaSuccess )
	printf ( "cudaMempcy return %d\n", cuda_error );
    cuda_error = cudaDeviceSynchronize();

    if ( cuda_error != cudaSuccess )
	printf ( "cudaSyn returned %d\n", cuda_error );


    int block_size = (params.packet_size / sizeof(int))/INTS_PER_THREAD;
    int block_count = params.m;

    int num_bytes_shared = params.packet_size;


//    for ( int i = 0; i < 1000; i++)
    encode <<< block_count, block_size, num_bytes_shared >>> (src_data_d, parity_data_d, 
	    pcm_d, params );

    cuda_error = cudaGetLastError();
    if ( cuda_error != cudaSuccess )
	printf("kernel execution returned %d\n", cuda_error);

    cudaThreadSynchronize();


    cudaMemcpy ( parity, parity_data_d, params.m*params.packet_size, cudaMemcpyDeviceToHost );
    cuda_error = cudaGetLastError();
    if ( cuda_error != cudaSuccess )
	printf("cudaMemcpy from device returned %d\n", cuda_error);

    cudaFree(src_data_d);
    cudaFree(parity_data_d);

} 

void gpu_decode ( char* received, int* error_vec, char* pc_matrix, struct coding_params params )
{ 
/*     int* received_d;
 *     int* pcm_d;
 *     int* error_vec_d;
 *     cudaError_t cuda_error;
 * 
 *     int k = params.k;
 *     int m = params.m;
 *     int packet_size = params.packet_size;
 * 
 *     int **pcm = make_compact_pcm ( pc_matrix, params );
 * 
 *     //alocate space and copy data to device
 *     cuda_error = cudaMalloc ( (void**) &received_d, (k+m)*packet_size);
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMalloc return %d\n", cuda_error );
 * 
 *     cuda_error = cudaMemcpy ( received_d, received, (k+m)*packet_size, cudaMemcpyHostToDevice );
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMempcy return %d\n", cuda_error );
 *     
 *     cuda_error = cudaMalloc ( (void**) &pcm_d, m*params.max_row_weight*sizeof(int));
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMalloc return %d\n", cuda_error );
 * 
 *     cuda_error = cudaMemcpy ( pcm_d, pcm, sizeof(int)*m*params.max_row_weight, 
 * 	    cudaMemcpyHostToDevice );
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMempcy return %d\n", cuda_error );
 * 
 *     cuda_error = cudaMalloc ( (void**) &error_vec_d, params.num_lost*sizeof(int));
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMalloc return %d\n", cuda_error );
 * 
 *     cuda_error = cudaMemcpy ( pcm_d, pcm, params.num_lost*sizeof(int),
 * 	    cudaMemcpyHostToDevice );
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMempcy pcm return %d\n", cuda_error );
 * 
 *     int block_size = (packet_size/sizeof(int)) / 4;
 *     int block_count  = m;
 *     int shared_mem_size = (packet_size + sizeof(int))*(params.max_row_weight+2);
 * 
 *     decode <<< block_count, block_size, shared_mem_size >>> (received_d, error_vec, pcm_d, params );
 * 
 *     cuda_error = cudaMemcpy ( received, received_d, (k+m)*packet_size, cudaMemcpyDeviceToHost );
 *     if ( cuda_error != cudaSuccess )
 * 	printf ( "cudaMempcy from device return %d\n", cuda_error );
 * 
 *     cudaFree ( received_d );
 */
}

/* void make_compact_pcm ( char* pc_matrix, int* pcm, struct coding_params params)
 * {
 *     //we need to create a compact representation of sparse pc_matrix
 * 
 *     int counter = 0;
 *     int columns = params.max_row_weight + 2;
 * 
 *     for ( int i = 0; i < params.m; i++) {
 * 	for ( int j = 0; j < params.k; j++)
 * 	    if ( pc_matrix[i*params.k + j] )
 * 	    {
 * 		pcm[i*columns + counter] = j;
 * 		counter++;
 * 	    }
 * 	//add indices from staircase matrix
 * 	pcm[i*columns + counter] = params.k + i;
 * 	counter++;
 * 	
 * 	if ( i > 0 )
 * 	{
 * 	    pcm[i*columns + counter] = params.k + i - 1;
 * 	    counter++;
 * 	}
 * 
 * 	if ( counter < columns )
 * 	    for ( int j = counter; j < columns; j++)
 * 		pcm[i*columns + j] = -1;
 * 	counter = 0;
 *     }
 * 
 * 
 *      for ( int i = 0; i < params.m; i++)
 *       {
 *   	for ( int j = 0; j < columns; j++ )
 *   	    printf ( "%d, ", pcm[i*columns + j] );
 *   	printf ( "\n" );
 *       }
 *  
 * 
 * }
 */
