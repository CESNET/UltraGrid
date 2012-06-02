/*
 * =====================================================================================
 *
 *       Filename:  ldpc-matrix.h
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

#define LDGM_MAX_K 8192

/*  Returns a pseudorandom number between 0 and bound-1 */
long unsigned int pmms_rand ( int bound ); 

/*  Computes degree of a given row  of the given matrix */
int degree_of_row ( char **matrix, int row, int column_num );

void left_matrix_init ( char **matrix, int k, int n, int N1, int seed );

void right_matrix_staircase_init ( char **matrix, int k, int n );
