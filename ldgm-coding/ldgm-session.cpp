/*
 * =====================================================================================
 *
 *       Filename:  ldgm-session.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/12/2012 01:03:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H
    
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string.h>

#include "ldgm-session.h"
#include "gpu.cuh"
#include "timer-util.h"

using namespace std;

/*
 *--------------------------------------------------------------------------------------
 *       Class:  LDGM_session
 *      Method:  LDGM_session
 * Description:  constructor
 *--------------------------------------------------------------------------------------
 */
LDGM_session::LDGM_session ()
{
}  /* -----  end of method LDGM_session::LDGM_session  (constructor)  ----- */

    void
LDGM_session::set_pcMatrix ( char* fname)
{
    FILE *f;

    f = fopen(fname, "rb");
    if (!f)
    {
	printf ( "Error opening matrix file\n" );
	printf ( "exiting\n" );
	abort();
    }
    unsigned int k_f, m_f, w_f;
    fscanf(f, "%d %d %d", &k_f, &m_f, &w_f);
//    printf ( "In matrix file: K %d M %d Columns %d\n", k_f, m_f, w_f );

    if ( k_f != param_k || m_f != param_m)
    {
	printf("Parity matrix size mismatch\nExpected K = %d, M = %d\nReceived K = %d, M = %d\n",
		param_k, param_m, k_f, m_f);
	return;
    }
    fseek (f, 1, SEEK_CUR );
    
    pcm = (int*) malloc(w_f*param_m*sizeof(int));
    for ( int i = 0; i < (int)w_f*param_m; i++)
	fread ( pcm+i, sizeof(int), 1, f);
    this->max_row_weight = w_f - 2; //w_f stores number of columns in adjacency list

/*     for ( int i = 0; i < param_m; i++)
 *     {
 * 	for ( int j = 0; j < w_f; j++)
 * 	    printf ( "[%3d]", (unsigned int)(pcm[i*w_f+j]) );
 * 	printf ( "\n" );
 *     }
 */




    fclose(f);



/* 
 *     this->max_row_weight = 0;
 *     int max_weight = 0;
 *     for ( int i = 0; i < param_m; ++i) {
 * 	for ( int j = 0; j < param_k; ++j) {
 * 	    if(*(matrix + i*param_k + j) == 1) {
 * 		*(pcMatrix + i*param_k + j) = *(matrix + i*param_k + j);
 * 		max_weight++;
 * 	    }
 * 	}
 * 	if ( max_weight > this->max_row_weight )
 * 	    this->max_row_weight = max_weight;
 * 	max_weight = 0;
 *     }
 * 
 *     pcm = (int*) malloc (m*(max_row_weight+2)*sizeof(int*)); 
 *     struct coding_params params;
 *     params.m = m;
 *     params.k = k;
 *     params.max_row_weight = max_row_weight;
 * 
 *     make_compact_pcm ( matrix, pcm, params);
 */


    return ;
}               /*  -----  end of method Coding_session::set_pcMatrix  ----- */

    char*
LDGM_session::encode_frame ( char* frame, int frame_size, int* out_buf_size )
{
    int buf_size;
    int ps;
    short header_size = 4;


    if ( (frame_size + header_size) % param_k == 0 )
	buf_size = frame_size + header_size;
    else
	buf_size = (((frame_size + header_size)/param_k) + 1)*param_k;

    ps = buf_size/param_k;

    packet_size = ps;
//    printf ( "ps: %d\n", ps );
    buf_size += param_m*ps;
    *out_buf_size = buf_size;

    void *out_buf;
    out_buf = aligned_malloc(buf_size, 16);
    if (!out_buf)
    {
	printf ( "Unable to allocate aligned memory\n" );
	return NULL;
    }
    memset(out_buf, 0, buf_size);

    //Insert frame size and copy input data into buffer

    int *hdr = (int*)out_buf;
    *hdr = frame_size;

    memcpy( ((char*)out_buf) + header_size, frame, frame_size);

    timespec start, end;
    //Timer_util t;

    this->encode ( (char*)out_buf, ((char*)out_buf)+param_k*ps );

//    printf ( "SSE: %.3lf\n", t.elapsed_time(start, end) );

/*     void *out_buf_check;
 *     error = posix_memalign(&out_buf_check, 16, buf_size);
 *     memset(out_buf_check, 0, buf_size);
 *     hdr = (int*)out_buf_check;
 *     *hdr = frame_size;
 *     memcpy(((char*)out_buf_check)+header_size, frame, frame_size);
 *     clock_gettime(CLOCK_MONOTONIC, &start);
 *     encode_naive ( (char*)out_buf, ((char*)out_buf_check)+param_k*ps );
 *     clock_gettime(CLOCK_MONOTONIC, &end);
 *     printf ( "CPU: %.3lf\n", t.elapsed_time(start, end) );
 * 
 *     int e = memcmp(out_buf, out_buf_check, buf_size);
 *     printf ( "memcpy on parities: %d\n", e );
 *     char *a = (char*)out_buf;
 *     char *b = (char*)out_buf_check;
 *     bool equal = true;
 *     int idx = 0;
 *     for ( int i = 0; i < param_m*ps; i++)
 *     {
 * 	if(a[i] != b[i])
 * 	{
 * 	    idx = i;
 * 	    printf ( "Error at index %d: %d vs %d\n", i, (unsigned char)a[i],
 * 		    (unsigned char)b[i]  );
 * 	    equal = false;
 * 	}
 *     }
 * 
 *     printf ( "Parities %s match.\n", (equal)?"":"do not" );
 */

//    printf ( "\n\nEncoded block:\n" );
//    printf ( "%-13s\v%10d B\n", "Frame hdr size:", header_size );
//    printf ( "%-13s\v%10d B\n", "Frame size:", frame_size );
//    printf ( "Padding size:\t%-d B\n", buf_size-frame_size-header_size-param_m*ps );
//    printf ( "Parity size:\t%-d B\n", param_m*ps );
//    printf ( "-------------------------------\n");
//    printf ( "Total block size:\t%d B\n", buf_size );
//    printf ( "Symbol size:\t%d\n\n\n", ps );


    return (char*)out_buf;

}

    char*
LDGM_session::encode_hdr_frame ( char *my_hdr, int my_hdr_size, char* frame, int frame_size, int* out_buf_size )
{
    int buf_size;
    int ps;
    short header_size = 4;
    int overall_size = my_hdr_size + frame_size;


    if ( (overall_size + header_size) % param_k == 0 )
	buf_size = overall_size + header_size;
    else
	buf_size = (((overall_size + header_size)/param_k) + 1)*param_k;

    ps = buf_size/param_k;

    packet_size = ps;
//    printf ( "ps: %d\n", ps );
    buf_size += param_m*ps;
    *out_buf_size = buf_size;

    void *out_buf;
    out_buf = aligned_malloc(buf_size, 16);
    if (!out_buf)
    {
	printf ( "Unable to allocate aligned memory\n" );
	return NULL;
    }
    memset(out_buf, 0, buf_size);

    //Insert frame size and copy input data into buffer

    int *hdr = (int*)out_buf;
    *hdr = overall_size;

    memcpy( ((char*)out_buf) + header_size, my_hdr, my_hdr_size);
    memcpy( ((char*)out_buf) + header_size + my_hdr_size, frame, frame_size);

    timespec start, end;
    //Timer_util t;

    this->encode ( (char*)out_buf, ((char*)out_buf)+param_k*ps );

//    printf ( "SSE: %.3lf\n", t.elapsed_time(start, end) );

/*     void *out_buf_check;
 *     error = posix_memalign(&out_buf_check, 16, buf_size);
 *     memset(out_buf_check, 0, buf_size);
 *     hdr = (int*)out_buf_check;
 *     *hdr = frame_size;
 *     memcpy(((char*)out_buf_check)+header_size, frame, frame_size);
 *     clock_gettime(CLOCK_MONOTONIC, &start);
 *     encode_naive ( (char*)out_buf, ((char*)out_buf_check)+param_k*ps );
 *     clock_gettime(CLOCK_MONOTONIC, &end);
 *     printf ( "CPU: %.3lf\n", t.elapsed_time(start, end) );
 * 
 *     int e = memcmp(out_buf, out_buf_check, buf_size);
 *     printf ( "memcpy on parities: %d\n", e );
 *     char *a = (char*)out_buf;
 *     char *b = (char*)out_buf_check;
 *     bool equal = true;
 *     int idx = 0;
 *     for ( int i = 0; i < param_m*ps; i++)
 *     {
 * 	if(a[i] != b[i])
 * 	{
 * 	    idx = i;
 * 	    printf ( "Error at index %d: %d vs %d\n", i, (unsigned char)a[i],
 * 		    (unsigned char)b[i]  );
 * 	    equal = false;
 * 	}
 *     }
 * 
 *     printf ( "Parities %s match.\n", (equal)?"":"do not" );
 */

//    printf ( "\n\nEncoded block:\n" );
//    printf ( "%-13s\v%10d B\n", "Frame hdr size:", header_size );
//    printf ( "%-13s\v%10d B\n", "Frame size:", frame_size );
//    printf ( "Padding size:\t%-d B\n", buf_size-frame_size-header_size-param_m*ps );
//    printf ( "Parity size:\t%-d B\n", param_m*ps );
//    printf ( "-------------------------------\n");
//    printf ( "Total block size:\t%d B\n", buf_size );
//    printf ( "Symbol size:\t%d\n\n\n", ps );


    return (char*)out_buf;

}

/*  
 *  ===  FUNCTION  ======================================================================
 *         Name:  create_edges
 *  Description:  
 * =====================================================================================
 */
    void
LDGM_session::create_edges ( Tanner_graph *graph )
{
    map<int, Node>::iterator it;
//    printf ( "graph: %p, param_k: %d, param_m: %d\n", graph, param_k, param_m );
    for ( int m = 0; m < param_m; ++m) {
	for ( int k = 0; k < max_row_weight+2; ++k ) {
	    int idx = pcm [ m*(max_row_weight+2) + k];
	    if( idx > -1 ) {
		it = graph->nodes.find(idx);
		(*it).second.neighbours.push_back(param_k + param_m + m);
		it = graph->nodes.find(param_k + param_m + m);
		(*it).second.neighbours.push_back(idx);
	    }
	}
    }
/*     it = graph->nodes.find(0);
 *     while ( it != graph->nodes.end() )
 *     {
 * 	printf ( "\nneighbours of node %d: ", it->first );
 * 	for (vector<int>::iterator i = it->second.neighbours.begin();
 * 		i != it->second.neighbours.end(); ++i)
 * 	    printf ( "%d, ", *i );
 * 	it++;
 *     }
 */

    //add edges representing the staircase matrix
/*     for ( int i = param_k + param_m; i < param_k + 2*param_m; ++i) {
 * 	it = graph->nodes.find(i);
 * 	(*it).second.neighbours.push_back(i - param_m);
 * 	it = graph->nodes.find(i - param_m);
 * 	(*it).second.neighbours.push_back(i);
 * 	if(i > param_m + param_k) {
 * 	    it = graph->nodes.find(i);
 * 	    (*it).second.neighbours.push_back(i - param_m - 1);
 * 	    it = graph->nodes.find(i - param_m - 1);
 * 	    (*it).second.neighbours.push_back(i);
 * 	}
 *     }
 */
    return ;
}               /*  -----  end of function create_edges  ----- */

    bool
LDGM_session::needs_decoding ( Tanner_graph *graph )
{
    for ( int i = 0; i < param_k; i++)
	if( ! graph->nodes.find(i)->second.isDone() )
	    return true;
    return false;
}               /*  -----  end of method LDGM_session::needs_decoding  ----- */


LDGM_session::~LDGM_session() {
    free(pcm);
}
