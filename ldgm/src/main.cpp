/*
* =====================================================================================
*
*       Filename:  main.cpp
*
*    Description:  
*
*        Version:  1.0
*        Created:  04/12/2012 03:02:58 PM
*       Revision:  none
*       Compiler:  gcc
*
*         Author:  Milan Kabat (), kabat@ics.muni.cz
*   Organization:  
*
* =====================================================================================
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <set>
#include <unistd.h>
#include <ctype.h>
#include <string.h>

#include <iostream>
#include <fstream>

#include "ldgm-session-cpu.h"
#include "ldgm-session-gpu.h"
#include "timer-util.h"

using namespace std;

//#define DATA_FNAME "data/data15.csv"
//#define M 64                                               /*  Number of parity symbols */
//#define K 256                                /*  Number of source symbols */
#define ROW_WEIGHT 3                            /*  Number of 1's in a row of parity matrix */
#define COLUMN_WEIGHT 3                         /*  Number of 1's in a columns of parity matrix */
//#define PACKET_SIZE 16384              /*  Number of bytes in a packet */
#define PACKET_LOSS 0.08                       /*  Packet loss rate */
#define ITERATIONS 100

void printMatrix ( char **m, int height, int width );
void printData ( char *data, int count, int packet_size );
void fillParityMatrix ( char** matrix, int height, int width );
int demo( int m, int k, int frame_size, char* matrix_fname, char* data_fname, int cpu, int gpu);
void demo_gpu();

/* 
* ===  FUNCTION  ======================================================================
*         Name:  main
*  Description:  
* =====================================================================================
*/
    int
main ( int argc, char *argv[] )
{
    short m, k, column_weight;
    int frame_size;

    opterr = 0;
    int c;
    int gpu = 0;
    int cpu = 0;
    char fname[32];
    char matrix_fname[32];

    while ( ( c = getopt ( argc, argv, "cf:gk:m:o:t:w:")) != -1 ) {
	switch(c) {
	    case 'w':
		column_weight = atoi ( optarg );
		break;
	    case 'c':
		if(gpu)
		{
		    fprintf (stderr,  "Only one of the options 'c' 'g' can be chosen.\n" );
		    return EXIT_FAILURE;
		}
		cpu = 1;
		break;
	    case 'f':
		frame_size = atoi ( optarg );
		break;
	    case 'g':
		if(cpu)
		{
		    fprintf (stderr,  "Only one of the options 'c' 'g' can be chosen.\n" );
		    return EXIT_FAILURE;
		}
		gpu = 1;
		break;
	    case 'k':
		k = atoi ( optarg );
		break;
	    case 'm':
		m = atoi ( optarg );
		break;
	    case 'o':
		sprintf(fname, optarg, 32);
		break;
	    case 't':
		sprintf(matrix_fname, optarg, 32);
		break;
	    case '?':
		if ( optopt == 'c' || optopt == 'k' || optopt == 'm' )
		    fprintf ( stderr, "Option -%c requires an argument.\n", optopt);
		else if ( isprint ( optopt))
		    fprintf ( stderr, "Unknown option '-%c'.\n", optopt);
		else
		    fprintf ( stderr, "Unknown option character '\\x%x'.\n", optopt);

		return 1;
	    default:
		abort();
	}
    }

    demo( k, m, frame_size, matrix_fname, fname, cpu, gpu);

    //    demo_gpu();

    return EXIT_SUCCESS;
} 	/* ----------  end of function main  ---------- */


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  usage
 *  Description:  Prints usage info
 * =====================================================================================
 */
    void
usage() 
{
    printf ( "\nUsage:\n" );

}

void compare_int_arrays ( int *a, int*b, int size )
{
    bool equal = true;
    int idx = -1;
    for ( int i = 0; i < size; i++ )
    {
	if ( a[i] != b[i] )
	{
	    equal = false;
	    idx = i;
	    printf ( "p1: %d, p2: %d\n", a[i], b[i] );
	    break;
//	    printf ( "Difference at integer position: %d\n", idx );
//	    printf ( "Difference at packet position: %d\n", idx/2048 );
	}
    }
    printf ( "Arrays are %s.\n", (equal)?"equal":"not equal" );

}
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  demo
 *  Description:  Demonstration of coding
 * =====================================================================================
 */
    int 
demo(int k, int m, int frame_size, char* matrix_fname, char* data_fname, int cpu, int gpu ) 
{
    Timer_util t;

//    LDGM_session_gpu* coding_session_gpu = new LDGM_session_gpu();
    LDGM_session_cpu* coding_session_cpu = new LDGM_session_cpu();
    LDGM_session_cpu* coding_session_cpu_2 = new LDGM_session_cpu();

  //  coding_session_gpu->set_params ( k, m, COLUMN_WEIGHT);
    coding_session_cpu->set_params ( k, m, COLUMN_WEIGHT);
    coding_session_cpu_2->set_params ( k, m, COLUMN_WEIGHT);
    //coding_session_gpu->set_data_fname(data_fname);

    coding_session_cpu->set_pcMatrix ( matrix_fname);
    coding_session_cpu_2->set_pcMatrix ( matrix_fname);
    //coding_session_gpu->set_pcMatrix ( matrix_fname);

    //generate data
    void *data;
    
    //allocate memory with address aligned to 16 bytes
    int e = posix_memalign(&data, 16, frame_size);

    if ( e != 0 ) {
	fprintf ( stderr, "Unable to allocate memory using posix_memalign.\n");
	return EXIT_FAILURE;
    }

    for ( int i = 0; i < frame_size; ++i)
	memset((char*)data+i, rand() % 256, 1);

    //allocate space for parity packets

//    void *parity;
    //allocate memory with address aligned to 16 bytes
//    e = posix_memalign(&parity, 16, (size_t)m*packet_size);
//    
//    if ( e != 0 ) {
//	fprintf ( stderr, "Unable to allocate memory using posix_memalign.\n");
//	return EXIT_FAILURE;
//    }


//    coding_session->add_src_head ( (char*) data, src_block, (uint16_t)0);

     //print first K data packets
//    printData ( (char*) data, K, PACKET_SIZE);
//    printData ( (char*) parity, M, PACKET_SIZE);
//    void *parity2;
    //allocate memory with address aligned to 16 bytes
//    e = posix_memalign(&parity2, 16, (size_t)m*packet_size);
    
//    if ( e != 0 ) {
//	fprintf ( stderr, "Unable to allocate memory using posix_memalign.\n");
//	return EXIT_FAILURE;
//    }

//    printf ( "k %d m %d frame_size %d, matrix %s data %s cpu %d gpu %d\n", 
//	    k, m, frame_size, matrix_fname, data_fname, cpu, gpu);

//   char *output = coding_session_cpu->encode_frame( (char*) data, packet_size);


    char* output;
    int buf_size;
    int f_size;
    char *decoded;
    map<int, int>  valid_data;
    int ps;
    srand(time(NULL));
    if (cpu)
    {
        t.start();

        for ( int i = 0; i < ITERATIONS; i++)
        {
                printf ( "encoding\n" );
                output = coding_session_cpu->encode_frame ( (char*) data, frame_size, &buf_size );
                ps = 1392; // coding_session_cpu->get_packet_size();
                int total = 0;

                valid_data.clear();

                for ( int j = 0; j < buf_size ; j += ps) {
                        int size = ps;
                        if(j + ps > buf_size) {
                                size = buf_size - j;
                        }
                        if(rand() % 100 > PACKET_LOSS * 100 ) {
                                valid_data.insert(pair<int,int>(j, size));
                                total += size;
                        } else {
                                if(j == 0) {
                                        fprintf(stderr, "Dropping first packet!!!!!!!!!!!!\n");
                                }
                                memset(output + j, 0x0, size);
                        }
                }

                int c;
                printf ( "received bytes: %d\n", total );
                printf ( "decoding\n" );
                decoded = coding_session_cpu_2->decode_frame(output, buf_size, &f_size, valid_data);
                printf ( "f_size: %d\n", f_size );
                static int good = 0;
                static int bad = 0;
                if(f_size) good++; else bad++;
                fprintf(stderr, "good: %d bad: %d\n", good, bad);

                for (int x= 0; x < f_size; ++x) {
                        if(((char *)data)[x] != decoded[x]) {
                                fprintf(stderr, "Different character %d / %d (original size %d)\n", x, f_size, frame_size);
                                break;
                        }
                }
                //memset(decoded, 0, f_size);
                coding_session_cpu->free_out_buf(output);
        }
        t.end();
        //	printf ( "CPU encoded in: %.6lf\n", t.elapsed_time(start, end) );
    } 
    if (gpu)
    {
            t.start();

            for ( int i = 0; i < ITERATIONS; i++)
                    //	    coding_session_gpu->encode ( (char*) data, (char*) frame_size );

                    t.end();
            printf ( "GPU encoded in: %.6lf\n", t.elapsed_time() );
    }


#if 0
    ofstream out;
    out.open(data_fname, ios::out | ios::app);

    out << t.elapsed_time() << ",";

    out.close();

    union int_bytes {                                                                               
            unsigned int number;                                                                        
            unsigned char bytes[4];                                                                     
    };                                                                                              
    union int_bytes fs;
    memcpy(&(fs.bytes), output, 4);

    //    printf ( "frame_size: \t\t%d\n", fs.number );
    //    printf ( "packet_size: \t\t%d\n", coding_session_cpu->get_packet_size() );


    //    printData ( (char*) output, 5, coding_session_cpu->get_packet_size());


    coding_session_cpu->free_out_buf(output);
#endif
    delete coding_session_cpu;
    delete coding_session_cpu_2;
    //    delete coding_session_gpu;
    free ( data );

    return EXIT_SUCCESS;
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  printData
 *  Description:  Prints given data, each number represents a byte
 * =====================================================================================
 */
        void
printData ( char *data, int count, int packet_size )
{
        for ( int k = 0; k < count ; ++k) {
                for ( int i = 0; i < packet_size; ++i )
                        printf ( "%d|", *((unsigned char*)(data) + k*packet_size + i ));
                printf ( "---\n" );
        }
}

/*
 * ===  FUNCTION  ======================================================================
 *         Name:  printMatrix
 *  Description:  Prints given matrix
 * =====================================================================================
 */
void printMatrix ( char **m, int height, int width )
{
        for(int i = 0; i < height; ++i) {
                for(int j = 0; j < width; ++j) {
                        printf("%3d", (unsigned char)m[i][j]);
                }
                printf("\n");
        }
        return;
}               /*  -----  end of function printMatrix  ----- */



