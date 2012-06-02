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
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <emmintrin.h>

#include <fstream>
#include <iostream>

#include "ldgm-session-gpu.h"
#include "gpu.cuh"
#include "timer-util.h"
    
    char*
xor_sse (char* source, char* dest, int packet_size)
{
    __m128i* wrd_ptr = (__m128i *) source;
    __m128i* wrd_end = (__m128i *) (source + packet_size);
    __m128i* dst_ptr = (__m128i *) dest;

    do
    {
	__m128i xmm1 = _mm_load_si128(wrd_ptr);
	__m128i xmm2 = _mm_load_si128(dst_ptr);

	xmm1 = _mm_xor_si128(xmm1, xmm2);     //  XOR  4 32-bit words
	_mm_store_si128(dst_ptr, xmm1);
	++wrd_ptr;
	++dst_ptr;

    } while (wrd_ptr < wrd_end);

    return dest;
}

void LDGM_session_gpu::encode ( char* source_data, char* parity )
{
    struct coding_params params;
    params.k = param_k;
    params.m = param_m;
    params.packet_size = packet_size;
    params.max_row_weight = max_row_weight;

    gpu_encode ( source_data, parity, this->pcm, params ); 
    
    //apply inverted staircase matrix
//    char parity_packet[packet_size];
//    timespec start, end;
//    clock_gettime(CLOCK_MONOTONIC, &start);
    for ( int i = 1; i < param_m; i++)
    {
	char *prev_parity = parity + (i - 1)*packet_size;
	xor_sse ( prev_parity, parity + i*packet_size, packet_size);
    }
//    clock_gettime(CLOCK_MONOTONIC, &end);

/*     Timer_util t;
 *     std::ofstream out;
 *     out.open(data_fname, std::ios::out | std::ios::app);
 * 
 *     out << t.elapsed_time_us(start,end) << ";";
 * 
 *     out.close();
 */



} 

void
LDGM_session_gpu::decode ( char* received, int* error_vec, int num_lost)
{
    struct coding_params params;
    params.num_lost = num_lost;
    params.k = param_k;
    params.m = param_m;
    params.packet_size = packet_size;
    params.max_row_weight = max_row_weight;

    gpu_decode ( received, error_vec, this->pcMatrix, params );
}


