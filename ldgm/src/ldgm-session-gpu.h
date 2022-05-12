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
 *        Authors:  Milan Kabat (kabat@ics.muni.cz), Vojtech David (374572@mail.muni.cz)
 *   Organization:  
 *
 * =====================================================================================
 */


#ifndef  LDGM_SESSION_GPU_INC
#define  LDGM_SESSION_GPU_INC

#include <string.h>
#include "ldgm-session.h"

#include <map>
#include <queue>

/*
 * =====================================================================================
 *        Class:  LDGM_session_gpu
 *  Description:  
 * =====================================================================================
 */
class LDGM_session_gpu : public LDGM_session
{
    public:

    // int * error_vec;
    // int * sync_vec;

    // int * ERROR_VEC;
    // int * SYNC_VEC;



	/* ====================  LIFECYCLE     ======================================= */
	LDGM_session_gpu ();                                  /* constructor      */
	LDGM_session_gpu ( const LDGM_session_gpu &other );   /* copy constructor */
	~LDGM_session_gpu ();                                 /* destructor       */

	/* ====================  ACCESSORS     ======================================= */

	/* ====================  MUTATORS      ======================================= */

	/* ====================  OPERATORS     ======================================= */

	void 
	    encode ( char* data_ptr, char* parity_ptr );
	
	void 
	    encode_naive ( char* /* data_ptr */, char* /* parity_ptr */ ) {}

	void
	    decode();

	void
	    free_out_buf (char *buf);

	 void *
		alloc_buf(int size);

	char * decode_frame ( char* received_data, int buf_size, int* frame_size, std::map<int, int> valid_data );
	void set_data_fname(char fname[32]) { strncpy(data_fname, fname, 32); }

    protected:
	/* ====================  DATA MEMBERS  ======================================= */

    private:
	/* ====================  DATA MEMBERS  ======================================= */
	char data_fname[32];

        std::queue<char *> freeBuffers;
        std::map<char *, size_t> bufferSizes;

	int OUTBUF_SIZE;
	int * OUTBUF;

	int * error_vec;
	int * sync_vec;

    int * SYNC_VEC;
	int * ERROR_VEC;

	int * PCM;



}; /* -----  end of class LDGM_session_gpu  ----- */

#endif   /* ----- #ifndef LDGM_SESSION_GPU_INC  ----- */
