/*
 * =====================================================================================
 *
 *       Filename:  ldgm-session-cpu.h
 *
 *    Description:  CPU implementation of LDGM coding
 *
 *        Version:  1.0
 *        Created:  04/11/2012 04:19:20 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */


#ifndef  LDGM_SESSION_CPU_INC
#define  LDGM_SESSION_CPU_INC

#include "ldgm-session.h"
#include "timer-util.h"

/*
 * =====================================================================================
 *        Class:  LDGM_session_cpu
 *  Description:  
 * =====================================================================================
 */
class LDGM_session_cpu : public LDGM_session
{
    public:
	/* ====================  LIFECYCLE     ======================================= */
	LDGM_session_cpu () { }                            /* constructor */
	~LDGM_session_cpu () { }                            /* constructor */

	void
	    encode (char*, char*);

	void
	    encode_naive (char*, char*);

	char*                                                                             
	    decode_frame ( char* received_data, int buf_size, int* frame_size,
		    std::map<int, int> valid_data );

	void
	    iterate ( Tanner_graph *graph);

	void
	    free_out_buf (char *buf);

    protected:
	/* ====================  DATA MEMBERS  ======================================= */

    private:
	/* ====================  DATA MEMBERS  ======================================= */

}; /* -----  end of class LDGM_session_cpu  ----- */

#endif   /* ----- #ifndef LDGM_SESSION_CPU_INC  ----- */
