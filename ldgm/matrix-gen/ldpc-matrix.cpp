/*
 * =====================================================================================
 *
 *       Filename:  ldpc-matrix.c
 *
 *    Description:  Generates LDPC Staircase matrix
 *
 *        Version:  1.0
 *        Created:  04/14/2012 10:50:43 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *        Company:  FI MUNI
 *
 * =====================================================================================
 */

#include <time.h>
#include <stdlib.h>

#include "ldpc-matrix.h"
#include "rand_pmms.h"


/*  Computes degree of a given row  of the given matrix */
int degree_of_row ( char **matrix, int row, int column_num )
{
    int j = 0;
    int deg = 0;

    for ( j = 0; j < column_num; j++ )
	if ( matrix[row][j] )
	    deg++;

    return deg;
}


void left_matrix_init ( char **matrix, int k, int n, int N1, int seed )
{
    int i;                    	/* Row index */
    int j;                      /* Column index */
    int h;                      /* Temporary variable */
    int t;                      /* Left limit within the list of possible choices u[] */
    int u[3*LDGM_MAX_K];            /* A table used to have a homogenous 1 distribution. */

    Rand_pmms rand_gen;
    rand_gen.seedi(seed);

    for ( h = N1*k - 1; h >= 0; h--)
    {
	u[h] = h % (n-k);
    }

    /*  Initialize the matrix with N1 "1s" per column, homogenously */
    t = 0;
    for ( j = 0; j < k; j++ )                   /* for each source symbol column */
    {
	for ( h = 0; h < N1; h++ )              /* put N1 "1s" */
	{
	    /*  Check wether valid available choices remain */
	    for ( i = t; i < N1*k && matrix [u[i]] [j] ; i++);
	    if ( i < N1*k )
	    {
		/*  Choose one index from possible choices */
		do {
		    i = t + rand_gen.pmms_rand ( N1*k - t);
		} while ( matrix [u[i]] [j]);
		matrix [u[i]] [j] = 1;

		/*  Replace with u[t]  */
		u[i] = u[t];
		t++;
	    } else {
		/* No available choices, pick at random */
		do {
		    i = rand_gen.pmms_rand ( n-k );
		} while ( matrix[i][j] );
		matrix[i][j] = 1;
	    }
	}
    }
    
    /*  Add extra bits to avoid rows with less than two "1s". */
    for ( i = 0; i < n-k; i++ )
    {
	if ( degree_of_row ( matrix, i, k ) == 0 )
	{
	    j = rand_gen.pmms_rand(k);
	    matrix[i][j] = 1;
	}
	if ( degree_of_row ( matrix, i, k ) == 1 )
	{
	    do {
		j = rand_gen.pmms_rand(k);
	    } while ( matrix[i][j] );
	    matrix[i][j] = 1;
	}
    }
}

void right_matrix_staircase_init ( char **matrix, int k, int n )
{
    int i;

    matrix[0][k] = 1;
    for ( i = 1; i < n-k; i++)
    {
	matrix[i][k+i] = 1;
	matrix[i][k+i-1] = 1;
    }
}
