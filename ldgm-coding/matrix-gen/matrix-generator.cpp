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

#include "ldpc-matrix.h"

using namespace std;


int generate_ldgm_matrix(char *fname, unsigned int k, unsigned int m, unsigned int column_weight,
                unsigned int seed) 
{
    int random = 0;
    int rfc = 1;

    if ( ! ( k > 0 && m > 0 && column_weight > 0 && (random > 0 || rfc > 0)  ) ) 
    {
	return EXIT_FAILURE;
    }

    char **pc_matrix;

    pc_matrix = (char**)malloc(m*sizeof(char*));
    for ( int i = 0; i < m; i++)
	pc_matrix[i] = (char*)calloc(k, sizeof(char));


    //Generate random 1's per column
    if (random)
    {
	srand(time(NULL));
	for ( int i = 0; i < k; ++i ) {
	    for ( int j = 0; j < column_weight; ++j) {
		pc_matrix[rand()%m][i] = 1;
	    }
	}
    }
    //Use algorithm from RFC 5170
    if (rfc)
    {
	left_matrix_init ( pc_matrix, k, k+m, column_weight, seed );
    }
    int max_weight = 0;
    //Compute maximum row weight
    for ( int i = 0; i < m; i++)
    {
	int m = 0;
	for ( int j = 0; j < k; j++)
	    if( pc_matrix[i][j])
		m++;
	if ( m > max_weight )
	    max_weight = m;
    }

/*     for ( int i = 0; i < m; i++)
 *     {
 * 	for ( int j = 0; j < k; j++)
 * 	    printf ( "[%3d]", pc_matrix[i][j] );
 * 	printf ( "\n" );
 *     }
 *     printf ( "\n" );
 */

    int **pcm;
    pcm = (int**)malloc(m*sizeof(int*));
    for ( int i = 0; i < m; i++)
	pcm[i] = (int*)malloc((max_weight+2) * sizeof(int));

    
    int columns = max_weight + 2;
    int counter = 0;
    for ( int i = 0; i < m; i++ )
    {
	for ( int j = 0; j < k; j++)
	    if( pc_matrix[i][j])
	    {
		pcm[i][counter] = j;
		counter++;
	    }
	//add indices from staircase matrix
	pcm[i][counter] = k + i;
	counter++;

	if ( i > 0 )
	{
	    pcm[i][counter] = k + i - 1;
	    counter++;
	}

	if ( counter < columns )
	    for ( int j = counter; j < columns; j++)
		pcm[i][j] = -1;
	counter = 0;
    }

/*     for ( int i = 0; i < m; i++)
 *     {
 * 	for ( int j = 0; j < columns; j++)
 * 	    printf ( "[%3d]", pcm[i][j] );
 * 	printf ( "\n" );
 *     }
 */
    FILE *out;

    int tmp;

    out = fopen (fname, "wb");
    if(out)
    {
	fprintf( out, "%d ", k); 
	fprintf( out, "%d ", m);  
	fprintf( out, "%d\n", columns);  
	for ( int i = 0; i < m; i++)
	{ 
	    for ( int j = 0; j < columns; j++)
	    {
		int t = pcm[i][j];
		tmp = fwrite(&t, sizeof(int), 1, out);
	    }
//	    printf ( "fwrite: %d\n", tmp );
	}
    } else
    {
	fprintf(stderr, "Cannot open file for writing.");
    }
    fflush(out);
    fclose(out);

/*     out = fopen(fname, "rb");
 *    
 *     int k_f, m_f, w_f;
 *     fscanf(out, "%d %d %d", &k_f, &m_f, &w_f);
 *     printf ( "In matrix file: K %d M %d Columns %d\n", k_f, m_f, w_f );
 *    
 *     fseek ( out , 1 , SEEK_CUR );
 *     int *p;
 *     p = (int*) malloc(columns*m*sizeof(int));
 *     memset(p, 99, columns*m*sizeof(int));
 *     for ( int i = 0; i < m; i++)
 *     {
 * 	for ( int j = 0; j < columns; j++)
 * 	{
 * 	    int t;
 * 	    tmp = fread ( &t, sizeof(int), 1, out);
 * 	    p[i*columns + j] = t;
 * 	}
 *     }
 * //	printf ( "fread: %d\n", tmp );
 *     for ( int i = 0; i < m; i++)
 *     {
 * 	for ( int j = 0; j < columns; j++)
 * 	    printf ( "[%3d]", (unsigned int)(p[i*columns+j]) );
 * 	printf ( "\n" );
 *     }
 * 
 */


//Print matrix
/*     for ( int i = 0; i < m; i++)
 *     {
 * 	for ( int j = 0; j < k; j++)
 * 	    printf ( "%3d", pc_matrix[i][j] );
 * 	printf ( "\n" );
 *     }
 *     printf ( "\n" );
 */


    for ( int i = 0; i < m; i++)
	free(pc_matrix[i]);
    free(pc_matrix);
    for ( int i = 0; i < m; i++)
	free(pcm[i]);
    free(pcm);

    return EXIT_SUCCESS;
}

