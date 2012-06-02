/*
 * =====================================================================================
 *
 *       Filename:  ldgm-session.h
 *
 *    Description:  LDGM coding session
 *
 *        Version:  1.0
 *        Created:  04/12/2012 11:05:35 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */


#include <stdint.h>

#include "coding-session.h"
#include "tanner.h"


#ifndef  LDGM_SESSION_INC
#define  LDGM_SESSION_INC

/**
 *  \class LDGM_session
 *  \brief Parent class for CPU and GPU implementations of LDGM coding
 *
 *  This class implements common mechanisms used during LDGM coding, however
 *  actual coding implementation has to be provided in child classes.
 * */
class LDGM_session : public Coding_session
{
    public:
	/* ====================  LIFECYCLE     ======================================= */
	LDGM_session ();                               /* constructor **/
	~LDGM_session ();                              /* destructor **/

	/* ====================  ACCESSORS     ======================================= */
	/**
	 * This method checks whether the given graph needs further decoding
	 *
	 * @param tanner_graph Given Tanner graph
	 * @return true if the given graph needs decoding, false otherwise
	 * */
	bool
	    needs_decoding ( Tanner_graph *tanner_graph );

	unsigned int 
	    get_packet_size () const
	    {
		return packet_size;
	    }

	void
	    set_pcMatrix ( char * matrix );

	/* ====================  OPERATORS     ======================================= */

	void
	    create_edges ( Tanner_graph *g);

	char*
	    encode_frame ( char* frame, int frame_size, int* out_buf_size );

	char*
	    encode_hdr_frame( char *hdr, int hdr_size, char* frame, int frame_size, int* out_buf_size );

	virtual void
	    encode ( char* data, char* parity ) = 0;
	
	virtual void
	    encode_naive ( char* data, char* parity ) = 0;

	virtual char*
	    decode_frame ( char* received_data, int buf_size, int* frame_size, 
		    std::map<int, int>  valid_data ) = 0;

	void
	    set_params ( unsigned short k,
		         unsigned short m,
			 unsigned short rw
		    )
	    {
		param_k = k;
		param_m = m;
		row_weight = rw;
	    }
		    

    protected:
	/* ====================  DATA MEMBERS  ======================================= */

	char* pcMatrix;
	int *pcm; //compact

	unsigned short param_k;
	unsigned short param_m;
	unsigned short row_weight;
	unsigned short packet_size;
	unsigned short max_row_weight;

	char *received_ptr;
	char *lost_ptr;

    private:

	/* ====================  DATA MEMBERS  ======================================= */

}; /* -----  end of class LDGM_session  ----- */

#endif   /* ----- #ifndef LDGM_SESSION_INC  ----- */

