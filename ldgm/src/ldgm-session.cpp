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

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <sstream>
#include <string>
#include <string.h>
#include "ldgm-session.h"
#include "timer-util.h"

constexpr const int MAX_W = 128;

using namespace std;

/*
 *--------------------------------------------------------------------------------------
 *       Class:  LDGM_session
 *      Method:  LDGM_session
 * Description:  constructor
 *--------------------------------------------------------------------------------------
 */
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
    if (fscanf(f, "%d %d %d", &k_f, &m_f, &w_f) != 3) {
        throw string("Parity matrix read error!");
    }
//    printf ( "In matrix file: K %d M %d Columns %d\n", k_f, m_f, w_f );

    if (w_f < 2 || w_f > MAX_W) {
        throw string("Invalid parameter in parity matrix (allowed range [2.." + to_string(MAX_W) + "])!");
    }

    if ( k_f != param_k || m_f != param_m)
    {
        ostringstream oss;
        oss << "Parity matrix size mismatch\nExpected K = " << param_k << "% M = " << param_m <<
                "\nReceived K = " << k_f << ", M = " << m_f << "\n";
        throw oss.str();
    }
    if (fseek (f, 1, SEEK_CUR ) != 0) {
            perror("fseek");
    }

    pcm = (int*) malloc(w_f*param_m*sizeof(int));
    for ( int i = 0; i < (int)w_f*param_m; i++) {
        if (fread ( pcm+i, sizeof(int), 1, f) != 1) {
            throw string("Parity matrix read error!");
        }
    }
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
    //printf("encode_frame\n");
    Timer_util interval;
    interval.start();

    int buf_size;
    int ps;
    short header_size = LDGM_session::HEADER_SIZE;
    int align_coef = param_k*sizeof(int);

    if ( (frame_size + header_size) % align_coef == 0 )
        buf_size = frame_size + header_size;
    else
        buf_size = ( ( (frame_size + header_size) / align_coef  ) + 1 ) * align_coef;

    ps = buf_size/param_k;

    packet_size = ps;
    //printf ( "ps: %d\n", ps );
    buf_size += param_m*ps;
    *out_buf_size = buf_size;

    void *out_buf;
    out_buf = alloc_buf(buf_size);
    if (!out_buf)
    {
        printf ( "Unable to allocate memory\n" );
        return NULL;
    }
    memset(out_buf, 0, header_size);
    memset((char*)out_buf + frame_size + header_size, 0, align_coef);

    //Insert frame size and copy input data into buffer

    int *hdr = (int*)out_buf;
    *hdr = frame_size;

    memcpy( ((char*)out_buf) + header_size, frame, frame_size);

    //Timer_util t;
    //printf("2buf_size %d\n",buf_size);
    //printf("2packet_size %d\n",packet_size);
    this->encode ( (char*)out_buf, ((char*)out_buf)+param_k*ps );


    interval.end();
    // printf("time: %e\n",elapsed/1000.0 );
    this->elapsed_sum2 += interval.elapsed_time_ms();
    this->no_frames2++;

    return (char*)out_buf;

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




}

char*
LDGM_session::encode_hdr_frame ( char *my_hdr, int my_hdr_size, char* frame, int frame_size, int* out_buf_size )
{
    int buf_size;
    int ps;
    short header_size = LDGM_session::HEADER_SIZE;
    int overall_size = my_hdr_size + frame_size;
    int align_coef = param_k*sizeof(int);

    if ( (overall_size + header_size) % align_coef == 0 )
        buf_size = overall_size + header_size;
    else
        buf_size = ( ( (overall_size + header_size) / align_coef  ) + 1 ) * align_coef;

    ps = buf_size/param_k;

    packet_size = ps;
//    printf ( "ps: %d\n", ps );
    buf_size += param_m*ps;
    *out_buf_size = buf_size;

    void *out_buf;
    out_buf = alloc_buf(buf_size);
    if (!out_buf)
    {
        printf ( "Unable to allocate aligned memory\n" );
        return NULL;
    }
    memset(out_buf, 0, buf_size);

    //Insert frame size and copy input data into buffer

    int32_t *hdr = (int32_t*)out_buf;
    *hdr = overall_size;

    memcpy( ((char*)out_buf) + header_size, my_hdr, my_hdr_size);
    memcpy( ((char*)out_buf) + header_size + my_hdr_size, frame, frame_size);

#if 0
    int my_frame_size=my_hdr_size+frame_size;

    // printf("my_frame_size: %d\n",my_frame_size );

    int crc;
    crc = crc32buf((char*)out_buf+header_size,my_frame_size);

    memcpy( ((char*)out_buf) + 4, &crc, 4);
    memcpy( ((char*)out_buf) + 8, &my_frame_size, 4);
    assert(LDGM_session::HEADER_SIZE >= 12);
#endif

    Timer_util interval;
    interval.start();

    this->encode ( (char*)out_buf, ((char*)out_buf)+param_k*ps );

    interval.end();
    long elapsed;

    elapsed = interval.elapsed_time_us();
    // printf("time: %e\n",elapsed/1000.0 );
    this->elapsed_sum2+=elapsed/1000.0;
    this->no_frames2++;


    // gettimeofday(&t1,0);
    // long elapsed;

    // elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
    // // printf("time: %e\n",elapsed/1000.0 );
    // this->elapsed_sum2+=elapsed/1000.0;
    // this->no_frames2++;
    // i++;

    // if(i%100==0){
    //     printf("TIME GPU: %f ms\n",this->elapsed_sum2/(double)this->no_frames2 );
    //     printf("time: %f ms\n",elapsed/1000.0 );
    // }

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
//    printf ( "graph: %p, param_k: %d, param_m: %d\n", graph, param_k, param_m );
    for ( int m = 0; m < param_m; ++m) {
        for ( int k = 0; k < max_row_weight+2; ++k ) {
            int idx = pcm [ m*(max_row_weight+2) + k];
            if( idx > -1 ) {
                auto &node = graph->nodes.at(idx);
                node.neighbours.push_back(param_k + param_m + m);
                node = graph->nodes.at(param_k + param_m + m);
                node.neighbours.push_back(idx);
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
        if( ! graph->nodes.at(i).isDone() ) {
            return true;
        }
    return false;
}               /*  -----  end of method LDGM_session::needs_decoding  ----- */


LDGM_session::~LDGM_session() {
    free(pcm);
}
