/*
 * AUTHOR:  Gerard Castillo <gerard.castillo@i2cat.net>,
 * 			David Cassany   <david.cassany@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H
#include "debug.h"
#include "perf.h"
#include "transmit.h"
#include "module.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/rtpenc_h264.h"
#include "video.h"
#include "video_codec.h"
#include "compat/platform_spin.h"
#include "video_frame.h"

struct rtpenc_h264_state {
	bool haveSeenFirstStartCode;
	uint8_t firstByteOfNALUnit;
	unsigned char* startOfFrame;
	unsigned char* to;
	unsigned char* from;
	unsigned curParserIndex; // <= inputFrameSize
	unsigned curParserIndexOffset;
	unsigned inputFrameSize;
	bool haveSeenEOF;
};

//UTILS DECLARATIONS
static uint32_t test4Bytes(struct rtpenc_h264_state *rtpench264state);
static unsigned char* startOfFrame(struct rtpenc_h264_state *rtpench264state);
static unsigned char* nextToParse(struct rtpenc_h264_state *rtpench264state);
static void checkEndOfFrame(struct rtpenc_h264_state *rtpench264state,
                unsigned numBytesNeeded);
static uint8_t get1Byte(struct rtpenc_h264_state *rtpench264state);
static void setFromState(struct rtpenc_h264_state *rtpench264state);
static void setToState(struct rtpenc_h264_state *rtpench264state);
static void skipBytes(struct rtpenc_h264_state *rtpench264state, unsigned numBytes);
static unsigned curNALSize(struct rtpenc_h264_state *rtpench264state);


struct rtpenc_h264_state * rtpenc_h264_init_state() {
	return calloc(1, sizeof(struct rtpenc_h264_state));
}

void rtpenc_h264_reset(struct rtpenc_h264_state *rtpench264state) {
	rtpench264state->haveSeenEOF = false;
	rtpench264state->haveSeenFirstStartCode = false;
}

unsigned rtpenc_h264_frame_parse(struct rtpenc_h264_state *rtpench264state,	uint8_t *buf_in, int size) {

	uint32_t next4Bytes = 0l;

	if (!rtpench264state->haveSeenFirstStartCode) {
		//reset pointers and params of interest for this new frame to parse and send
		rtpench264state->startOfFrame = rtpench264state->from = rtpench264state->to = buf_in;
		rtpench264state->curParserIndex = 0;
		rtpench264state->inputFrameSize = size;
		rtpench264state->curParserIndexOffset = 0;
		// The frame must start with a 0x00000001:
		// Skip over any input bytes that precede the first 0x00000001 and assert it
		while (test4Bytes(rtpench264state) != 0x00000001) {
			get1Byte(rtpench264state);
			if(rtpenc_h264_have_seen_eof(rtpench264state)){
				error_msg("No NAL found!\n");
				return 0; //this shouldn't happen -> this would mean that we got new frame but no start code was found inside....
			}
		}
		skipBytes(rtpench264state, 4); // skip this initial code
		setFromState(rtpench264state); //set 'from' pointer of curr input frame
		rtpench264state->haveSeenFirstStartCode = true;

	} else {
		//CONTINUE WITH THE SAME FRAME
		// Assert: next4Bytes starts with 0x00000001 or 0x000001, and we've saved and sent all previous bytes (forming a complete NAL unit).
		// Skip over these remaining bytes
		if (test4Bytes(rtpench264state) == 0x00000001) {
			skipBytes(rtpench264state, 4);
		} else {
			skipBytes(rtpench264state, 3);
		}
		if(rtpenc_h264_have_seen_eof(rtpench264state)){
			error_msg("No NAL found!\n");
			return 0; //this shouldn't happen -> this would mean that we got more to parse but we run out of space....
		}

		setFromState(rtpench264state); //re-set 'from' pointer of current input data
	}

	// Then save everything up until the next 0x00000001 (4 bytes) or 0x000001 (3 bytes), or we hit EOF.
	// Also make note of the first byte, because it contains the "nal_unit_type":
	next4Bytes = test4Bytes(rtpench264state);
	rtpench264state->firstByteOfNALUnit = next4Bytes >> 24;

	while (next4Bytes != 0x00000001 && (next4Bytes & 0xFFFFFF00) != 0x00000100 && !rtpenc_h264_have_seen_eof(rtpench264state)) {
		// We save at least some of "next4Bytes".
		if ((unsigned) (next4Bytes & 0xFF) > 1) {
			// Common case: 0x00000001 or 0x000001 definitely doesn't begin anywhere in "next4Bytes", so we save all of it:
			skipBytes(rtpench264state, 4);
		} else {
			// Save the first byte, and continue testing the rest:
			skipBytes(rtpench264state, 1);
		}
		next4Bytes = test4Bytes(rtpench264state);
	}
	setToState(rtpench264state);

	return curNALSize(rtpench264state);
}

bool rtpenc_h264_have_seen_eof(struct rtpenc_h264_state *rtpench264state) {
	return rtpench264state->haveSeenEOF;
}

unsigned char *rtpenc_h264_get_from_state(struct rtpenc_h264_state *rtpench264state) {
        return rtpench264state->from;
}

//UTILS
static uint32_t test4Bytes(struct rtpenc_h264_state *rtpench264state) {
	checkEndOfFrame(rtpench264state, 4);

	unsigned char const* ptr = nextToParse(rtpench264state);
	return (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
}
static unsigned char* startOfFrame(struct rtpenc_h264_state *rtpench264state) {
	return rtpench264state->startOfFrame;
}
static unsigned char* nextToParse(struct rtpenc_h264_state *rtpench264state) {
	return &startOfFrame(rtpench264state)[rtpench264state->curParserIndex];
}
static void checkEndOfFrame(struct rtpenc_h264_state *rtpench264state,
		unsigned numBytesNeeded) {
	// assure EOF check
	if (rtpench264state->curParserIndex + numBytesNeeded
			>= rtpench264state->inputFrameSize){
		rtpench264state->haveSeenEOF = true;
	}
}
static uint8_t get1Byte(struct rtpenc_h264_state *rtpench264state) { // byte-aligned
	checkEndOfFrame(rtpench264state, 1);
	return startOfFrame(rtpench264state)[rtpench264state->curParserIndex++];
}
static void setFromState(struct rtpenc_h264_state *rtpench264state) {
	rtpench264state->from = rtpench264state->startOfFrame
			+ rtpench264state->curParserIndex;
	rtpench264state->curParserIndexOffset = rtpench264state->curParserIndex;
}
static void setToState(struct rtpenc_h264_state *rtpench264state) {
	if(rtpenc_h264_have_seen_eof(rtpench264state)) {
		rtpench264state->to = rtpench264state->startOfFrame + rtpench264state->inputFrameSize;
	} else {
		rtpench264state->to = rtpench264state->from
			+ (rtpench264state->curParserIndex - rtpench264state->curParserIndexOffset);
	}
}
static void skipBytes(struct rtpenc_h264_state *rtpench264state, unsigned numBytes) {
	checkEndOfFrame(rtpench264state, numBytes);
	rtpench264state->curParserIndex += numBytes;
}
static unsigned curNALSize(struct rtpenc_h264_state *rtpench264state) {
	return (rtpench264state->to - rtpench264state->from);
}
