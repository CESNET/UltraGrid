/*
 * =====================================================================================
 *
 *       Filename:  fec-header.h
 *
 *    Description:  LDGM FEC source header and FEC parity header definition
 *
 *        Version:  1.0
 *        Created:  04/16/2012 06:55:04 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */

/** Structure defining fields in LDGM FEC block header */
typedef struct {
    uint16_t sbn;                               /** Source block number */
    uint16_t esi;                               /** Sequence number of this data block in
						 *  the encoding block */
    uint16_t k;                                 /** Number of source symbols in the encoding
    						  * block */
    uint16_t m;                                 /** Number of parity symbols in the encoding
    						  * block */
} fec_header_t;
