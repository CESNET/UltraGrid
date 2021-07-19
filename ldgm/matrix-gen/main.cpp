/*
 * =====================================================================================
 *
 *       Filename:  matrix-generator.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04/30/2012 04:49:14 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#include "matrix-generator.h"

using namespace std;

void usage()
{
    fprintf(stdout, "\nOptions:\n");
    fprintf(stdout, "\t'a'\tUse algorithm from RFC 5170\n");
    fprintf(stdout, "\t'c'\tNumber of 1's per column\n");
    fprintf(stdout, "\t'f'\tOutput filename\n");
    fprintf(stdout, "\t'k'\tParameter 'K' of LDGM coding\n");
    fprintf(stdout, "\t'm'\tParameter 'M' of LDGM coding\n");
    fprintf(stdout, "\t'r'\tGenerate matrix by random\n");
    fprintf(stdout, "\t's'\tPseudo-random number generator seed\n");

    fprintf(stdout, "\n");
}

int main (int argc, char* argv[])
{
    extern int optopt;
    extern char* optarg;
    int c;
    
    int k = 0;
    int m = 0;
    int seed = 0;
    int column_weight = 0;

    int random = 0;
    int rfc = 0;
    char fname[32];


    while ( ( c = getopt ( argc, argv, "ac:f:hk:m:rs:")) != -1 ) {
	switch(c) {
	    case 'a':
		if(random)
		{
		    fprintf (stderr,  "Only one of the options 'a' 'r' can be chosen.\n" );
		    return EXIT_FAILURE;
		}
		rfc = 1;
		break;
	    case 'c':
		column_weight = atoi ( optarg );
		break;
	    case 'f':
		sprintf(fname, "%s", optarg );
		break;
	    case 'h':
                usage();
		return EXIT_SUCCESS;
	    case 'k':
		k = atoi ( optarg );
		break;
	    case 'm':
		m = atoi ( optarg );
		break;
	    case 'r':
		if(rfc)
		{
		    fprintf (stderr,  "Only one of the options 'a' 'r' can be chosen.\n" );
		    return EXIT_FAILURE;
		}
		random = 1;
		break;
	    case 's':
		seed = atoi ( optarg );
		break;
	    case '?':
		if ( optopt == 'c' || optopt == 'k' || optopt == 'm' )
		    fprintf ( stderr, "Option -%c requires an argument.\n", optopt);
		else
		    fprintf ( stderr, "Unknown option character '\\x%x'.\n", optopt);

		return 1;
	    default:
		abort();
	}
    }

    if ( ! ( k > 0 && m > 0 && column_weight > 0 && (random > 0 || rfc > 0)  ) ) 
    {
        usage();
        return EXIT_FAILURE;
    }

    return generate_ldgm_matrix(fname, k, m, column_weight, seed, 0);
}

