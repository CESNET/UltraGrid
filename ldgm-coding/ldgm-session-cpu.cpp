/*
 * =====================================================================================
 *
 *       Filename:  ldgm-session-cpu.cpp
 *
 *    Description:  CPU implementation of LDGM coding
 *
 *        Version:  1.0
 *        Created:  04/12/2012 01:21:07 PM
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

#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <string.h>
#include <time.h>

#include "ldgm-session-cpu.h"
#include "timer-util.h"

using namespace std;

    char*
xor_using_sse (char* source, char* dest, int packet_size)
{
    //First, do as many 128-bit XORs as possible 
    int iter_bytes_16 = 0;
    int iter_bytes_4 = 0;
    int iter_bytes_1 = 0;

    iter_bytes_16 = (packet_size/16)*16;

    if ( iter_bytes_16 > 0)
    {

	//    printf ( "iter_bytes: %d\n", iter_bytes );
	__m128i* wrd_ptr = (__m128i *) source;
	__m128i* wrd_end = (__m128i *) (source + iter_bytes_16);
	__m128i* dst_ptr = (__m128i *) dest;

	//    printf ( "wrd_ptr address: %p\n", wrd_ptr );
	do
	{
	    __m128i xmm1 = _mm_loadu_si128(wrd_ptr);
	    __m128i xmm2 = _mm_loadu_si128(dst_ptr);

	    xmm1 = _mm_xor_si128(xmm1, xmm2);     //  XOR  4 32-bit words
	    _mm_storeu_si128(dst_ptr, xmm1);
	    ++wrd_ptr;
	    ++dst_ptr;

	} while (wrd_ptr < wrd_end);
    }
    //Check, whether further XORing is necessary
    if ( iter_bytes_16 < packet_size )
    {
	char *mark_source = source + iter_bytes_16;
	char *mark_dest = dest + iter_bytes_16;

	iter_bytes_4 = ((packet_size - iter_bytes_16)/4)*4;

	for ( int i = 0; i < (packet_size - iter_bytes_16)/4; i++)
	{
	    int *s = ((int*) mark_source) + i;
	    int *d = ((int*) mark_dest) + i;
	    *d ^= *s;
	}

	mark_source += iter_bytes_4;
	mark_dest += iter_bytes_4;

	iter_bytes_1 = packet_size - iter_bytes_16 - iter_bytes_4;

	for ( int i = 0; i < iter_bytes_1; i++)
	{
	    *(mark_dest + i) ^= *(mark_source+i);
	}
    }

//    printf ( "XORed: %d bytes using SSE, %d bytes as ints and %d bytes byte-per-byte.\n", 
//	    iter_bytes_16, iter_bytes_4, iter_bytes_1);

    return dest;
}

    void
LDGM_session_cpu::encode ( char* data_ptr, char* parity_ptr )
{
//    start encoding
//    printf ( "packet_size: %d\n", this->packet_size );
    void *ppacket;
    char *parity_packet;
    for ( int m = 0; m < param_m; ++m) {

	ppacket = aligned_malloc(packet_size, 16);
	
	if (!ppacket)
	{
	    printf ( "Error while using posix_memalign\n" );
	    return;
	}

//	printf ( "m: %d\n", m );
	memset(ppacket, 0, packet_size);
	parity_packet = (char*)ppacket;
//	printf ( "max w: %d\n", max_row_weight );
	//Find out which packets to XOR
	for ( int k = 0; k < max_row_weight+2; ++k) {
	    int idx = pcm[m*(max_row_weight+2) + k];
//	    printf ( "adept: %d\n", idx );
	    if (idx > -1 && idx < param_k) {
//		printf ( "xoring idx: %d\n", idx );
		char *ptr = data_ptr + idx*packet_size;
		parity_packet = xor_using_sse(ptr, parity_packet, packet_size);
	    }
	}

	//Apply inverted staircase matrix
	if( m > 0) {
	    char *prev_parity = parity_ptr + (m-1)*packet_size;
	    parity_packet = xor_using_sse(prev_parity, parity_packet, packet_size);
	}


	//Add the new parity packet to overall parity
	memcpy ( parity_ptr + m*packet_size, parity_packet, packet_size );
	aligned_free(ppacket);

    }
    return ;
}		/* -----  end of method LDGM_session_cpu::encode  ----- */

    void
LDGM_session_cpu::free_out_buf ( char *buf)
{
    if ( buf != NULL )
	aligned_free(buf);
}
    void
LDGM_session_cpu::encode_naive ( char* data_ptr, char* parity_ptr )
{
//    start encoding

    void *ppacket;
    char *parity_packet;
    
    for ( int m = 0; m < param_m; ++m) {

	ppacket = aligned_malloc(packet_size, 16);
	if (!ppacket)
	{
	    printf ( "Error while using posix_memalign\n" );
	    return;
	}
	memset(ppacket, 0, packet_size);
	parity_packet = (char*)ppacket;
	
	//Find out which packets to XOR
	for ( int k = 0; k < max_row_weight+2; ++k) {
	    int idx = pcm[m*(max_row_weight+2) + k];
	    if (idx > -1 && idx < param_k) {
		char *ptr = data_ptr + idx*packet_size;
//		parity_packet = xor_using_sse(ptr, parity_packet, packet_size);
		for ( int i = 0; i < packet_size; i++)
		    parity_packet[i] ^= *(ptr+i);
	    }
	}

	//Apply inverted staircase matrix
	if( m > 0) {
	    char *prev_parity = parity_ptr + (m-1)*packet_size;
//	    parity_packet = xor_using_sse(prev_parity, parity_packet, packet_size);
	    for ( int i = 0; i < packet_size; i++)
		parity_packet[i] ^= *(prev_parity+i);
	}


	//Add the new parity packet to overall parity
	memcpy ( parity_ptr + m*packet_size, parity_packet, packet_size );
	aligned_free(ppacket);

    }
    return ;
}		/* -----  end of method LDGM_session_cpu::encode  ----- */

void iterate ( Tanner_graph *g);

    char*
LDGM_session_cpu::decode_frame ( char* received, int buf_size, int* frame_size, 
	std::map<int, int> valid_data )
{
//    printf ( "buf_size: %d\n", buf_size );
    Tanner_graph graph;

    int p_size = buf_size/(param_m+param_k);
    this->packet_size = p_size;
//    printf ( "p_size %d\n", p_size );
    graph.set_data_size(p_size);

    //Timer_util timer;

    int i;
    int index = 0;


//    while ( i < (param_k + param_m)*p_size)
//    {
//	printf ( "%2d|", (unsigned char)received[i++] );
//    }

    //one variable node per each data packet in block K
    for ( i = 0; i < param_k; ++i )
	graph.add_node(Node::variable_node, index++, received + i*p_size);

    //one variable node per each parity packet in block M
    for ( i = 0; i < param_m; ++i)
	graph.add_node(Node::variable_node, index++, received + (i + param_k)*p_size);

    //one constraint node per each row of generation matrix
    for ( i = 0; i < param_m; ++i)
	graph.add_node(Node::constraint_node, index++, NULL);
    
    create_edges(&graph);

//    printf("Graph created in: %.3f s\n", t);

    /*      printf ( "graf\n" );
     *     for(map<int, Node>::iterator it = graph.nodes.begin(); it != graph.nodes.end(); ++it) {
     *      Node n = it->second;
     *      for ( int i = 0; i < p_size; ++i)
     *          printf("%2d|", *(unsigned char *)(n.getDataPtr() + i));
     *      printf ( "---\n" );
     *      }
     */

    /*      for(map<int, Node>::iterator it = graph.nodes.begin(); it != graph.nodes.end(); ++it) {
     *      Node n = it->second;
     *      printf ( "neigbours of %d (count %lu): ", it->first, (it->second).neighbours.size() );
     *      for(vector<int>::iterator itr = n.neighbours.begin(); itr != n.neighbours.end(); ++itr)
     *          printf ( "%d ", *itr );
     *      printf ( "\n" );
     *     }
     */

    //We need to merge intervals in the valid data vector
    map <int, int> merged_intervals;
    map<int,int>::iterator map_it;

    if ( valid_data.size() != 0 )
    { 
	for ( map_it = valid_data.begin(); map_it != valid_data.end(); )
	{
	    int start = map_it->first;
	    int length = map_it->second;
	    while ( start + length == (++map_it)->first )
		length += map_it->second;
	    merged_intervals.insert ( pair<int, int> (start, length) );
	}
    }
/*     printf ( "Valid data: \n" );
 *     for ( map_it = valid_data.begin(); map_it != valid_data.end(); ++map_it)
 *     {
 * 	printf ( "|%d-%d| ", map_it->first, map_it->second );
 *     }
 *     printf ( "\n" );
 *     
 *     printf ( "Merged intervals: \n" );
 *     for ( map_it = merged_intervals.begin(); map_it != merged_intervals.end(); ++map_it)
 *     {
 * 	printf ( "|%d-%d| ", map_it->first, map_it->second );
 *     }
 *     printf ( "\n" );
 * 
 */

    
    
    map<int, Node>::iterator it;
    if ( merged_intervals.size() != 0)
    {
	it = graph.nodes.find(0);
	while (it != graph.nodes.find(param_k+param_m)) {
	    (*it).second.setDone(false);
	    int node_offset = (*it).second.getDataPtr() - received;
//	    printf ( "offset: %d\n", node_offset );

	    map_it = merged_intervals.begin();
	    //Find the offset in valid data which is equal, or the first offset which is
	    //lower than node offset
	    bool found = false;
	    while ( (map_it->first <= node_offset) && map_it != merged_intervals.end() )
	    {
		map_it++;
		found = true;
	    }
	    if ( map_it->first > 0 )
		map_it--;

	    //Next, find out if some interval covers this symbol
	    if ( found && (map_it->first + map_it->second) >=
		    (node_offset + p_size) )
	    {
		(*it).second.setDone(true);
//		printf ( "setting node %d with offset %d and size %d as done\n", it->first,
//			node_offset, p_size);
	    }

	    ++it;
	}
    }

    int not_done = 0;
    for ( it = graph.nodes.begin(); it != graph.nodes.find(param_k); it++)
    {
	if ( !it->second.isDone())
	{
	    memset(it->second.getDataPtr(), 0, p_size);
//	    printf ( "resetting node: %d\n", it->first );
	    not_done++;
	}
    }
//    printf ( "not done: %d\n", not_done );
/*     srand(time(NULL));
 *     for ( int j = 0; j < 100; j++)
 *     {
 * 	it = graph.nodes.find(rand()%(param_k+param_m));
 * 	(*it).second.setDone(false);
 *     }
 */
    int iter = 0;

    while ( needs_decoding(&graph) && iter < 4) {
//	printf ( "iteratin\n" );
	iterate(&graph);
	iter++;
    }   

//    printf("decoding process: %.3f s\n", t);

    //printf ( "iterations: %d\n", iter );

    int undecoded = 0;
    it = graph.nodes.find(0);
    while ( it != graph.nodes.find(param_k) ) {
	if (!(*it).second.isDone())
	    undecoded++;
	++it;
    }
//    printf ( "Number of not recovered data packets: %d\n", undecoded );

    //    printf("rest: %.3f s\n", t);

    if ( undecoded == 0 )
    {
	union int_bytes {
	    unsigned int number;
	    unsigned char bytes[4];
	};
	union int_bytes fs;
	memcpy(&(fs.bytes), received, 4);
	*frame_size = fs.number;
    }
    else
	*frame_size = 0;


    return received+4;
}		/* -----  end of method LDGM_session_cpu::decode  ----- */


    void
LDGM_session_cpu::iterate ( Tanner_graph *graph )
{
    map<int, Node>::iterator it_c;
    vector<int> vec;

    static int recovered = 0;

    //select the first constraint node
    it_c = graph->nodes.find ( param_k + param_m );

    //iterate through constraint nodes
    while ( it_c != graph->nodes.end() ) {
	//iterate the node's neighbours to find out how many of them are not decoded
	map<int, Node>::iterator it_v;
	for(vector<int>::iterator j = it_c->second.neighbours.begin();
		j != it_c->second.neighbours.end(); ++j) {
	    it_v = graph->nodes.find(*j);
	    if ( !it_v->second.isDone() )
		vec.push_back(*j);
	}
//	printf ( "node %d has %d undecoded neighbours\n", it_c->first, vec.size() );

	//we can restore the missing packet
	if ( vec.size() == 1) 
	{
	    int r_index = vec.front();

//	    if ( r_index == 0 )
//	    {
//		printf ( "repairing first block\n" );
//	    }
	    it_v = graph->nodes.find(r_index);
	    memset(it_v->second.getDataPtr(), 0, packet_size);
	    char *r_data = it_v->second.getDataPtr();
	    //find other nodes connected to this constraint node and XOR their values
	    int count = 0;
	    for(vector<int>::iterator j = it_c->second.neighbours.begin();
		    j != it_c->second.neighbours.end(); ++j) 
	    {
		if ( *j != r_index ) 
		{
//		    printf ( "decode, packet_size: %d\n", packet_size );
		    char *g_data = (graph->nodes.find(*j))->second.getDataPtr();
		    //XOR
		    xor_using_sse(g_data, r_data, packet_size);
		    count++;
		}
	    }
	    /*           //validate recovered packet
	     *          for ( int i = 0; i < param_k; ++i) {
	     *              if(!memcmp(r_data, lost_ptr + i*packet_size, packet_size)) {
	     *                  printf ( "packet ok\n" );
	     *                  break;
	     *              }
	     *          }
	     */
	    if ( count > 0 )
		it_v->second.setDone(true);
	    recovered++;
/*           printf ( "restored data: \n" );
 *          for ( int  i = 0; i < packet_size; ++i)
 *              printf ( "%2d|", (unsigned char)(*(it_v->second.getDataPtr() + i)) );
 *          printf ( "---\n" );
 */


	}
	vec.clear();

	++it_c;
    }
//    printf ( "recovered: %d\n", recovered );

    return;

}

