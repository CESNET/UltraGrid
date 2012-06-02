/*
 * =====================================================================================
 *
 *       Filename:  ldgm-session-gpu.h
 *
 *    Description:  GPU implementation fo LDGM coding
 *
 *        Version:  1.0
 *        Created:  04/12/2012 12:54:51 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */


#ifndef  LDGM_SESSION_GPU_INC
#define  LDGM_SESSION_GPU_INC

#include <string.h>
#include "ldgm-session.h"

/*
 * =====================================================================================
 *        Class:  LDGM_session_gpu
 *  Description:  
 * =====================================================================================
 */
class LDGM_session_gpu : public LDGM_session
{
    public:
	/* ====================  LIFECYCLE     ======================================= */
	LDGM_session_gpu () {}                             /* constructor      */
	LDGM_session_gpu ( const LDGM_session_gpu &other );   /* copy constructor */
	~LDGM_session_gpu () {}                           /* destructor       */

	/* ====================  ACCESSORS     ======================================= */

	/* ====================  MUTATORS      ======================================= */

	/* ====================  OPERATORS     ======================================= */

	void 
	    encode ( char* data_ptr, char* parity_ptr );
	
	void 
	    encode_naive ( char* data_ptr, char* parity_ptr ) {}

	void
	    decode();

	void
	    iterate ( Tanner_graph *graph);

	void decode (char* received, int* error_vec, int num_lost);
	void set_data_fname(char fname[32]) { strncpy(data_fname, fname, 32); }

    protected:
	/* ====================  DATA MEMBERS  ======================================= */

    private:
	/* ====================  DATA MEMBERS  ======================================= */
	char data_fname[32];

}; /* -----  end of class LDGM_session_gpu  ----- */

#endif   /* ----- #ifndef LDGM_SESSION_GPU_INC  ----- */
