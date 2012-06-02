/*
 * =====================================================================================
 *
 *       Filename:  coding-session.h
 *
 *    Description:  This abstract class represents a general coding session object.
 *                  Every class implementing a particular coding method/algorithm
 *                  must inherit from this class.
 *
 *         Author:  Milan Kabat, kabat@ics.muni.cz
 *
 * =====================================================================================
 */

#ifndef CODING_SESSION
#define CODING_SESSION

#include <map>

/** \class Coding_session
 *  \brief Abstract class Coding_session
 *   
 *   This abstract class represents a general coding session object. Every class implementing
 *   a particular coding method/algorithm must inherit from this class.
 */
class Coding_session
{
    public:

	/**
	 * Method encode should compute parity data for the given source data and
	 * add suitable FEC parity header.
	 *
	 * @param source_data Data to encode
	 * @param frame_size Size of the frame being encoded
	 * @param buf_size Output parameter for storing total size of the returned buffer
	 * @return Computed parity data with appropriate FEC parity header
	 * */
	virtual char*
	    encode_frame ( char* source_data, int frame_size, int* buf_size ) = 0;

	/**
	 * Method decode should recover original source data
	 *
	 * @param received_data Received data (source and parity)
	 * @param buf_size Size of the received buffer
	 * @param frame_size Output parameter for storing size of the decoded frame
	 * @param valid_data Map storing pairs <offset, number of bytes> of received data
	 * @return Recovered source data
	 * */
	virtual char*
	    decode_frame ( char* received_data, int buf_size, int* frame_size, 
		    std::map<int,int> valid_data) = 0;
};

#endif

