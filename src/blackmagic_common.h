/**
 * @file   blackmagic_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
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
 */

#ifndef BLACKMAGIC_COMMON_H
#define BLACKMAGIC_COMMON_H

#ifdef WIN32
#include "DeckLinkAPI_h.h" /*  From DeckLink SDK */
#else
#include "DeckLinkAPI.h" /*  From DeckLink SDK */
#endif

#include <map>
#include <string>
#include <utility>

#include "video.h"

std::string bmd_hresult_to_string(HRESULT res);

// Order of codecs is important because it is used as a preference list (upper
// codecs are favored) returned by DISPLAY_PROPERTY_CODECS property (display)
static std::map<codec_t, BMDPixelFormat> uv_to_bmd_codec_map = {
                  { R12L, bmdFormat12BitRGBLE },
                  { R10k, bmdFormat10BitRGBX },
                  { v210, bmdFormat10BitYUV },
                  { RGBA, bmdFormat8BitBGRA },
                  { UYVY, bmdFormat8BitYUV },
};

#ifdef WIN32
#define BMD_BOOL BOOL
#define BMD_TRUE TRUE
#define BMD_FALSE FALSE
#else
#define BMD_BOOL bool
#define BMD_TRUE true
#define BMD_FALSE false
#endif

#ifdef HAVE_MACOSX
#define BMD_STR CFStringRef
#elif defined WIN32
#define BMD_STR BSTR
#else
#define BMD_STR const char *
#endif
char *get_cstr_from_bmd_api_str(BMD_STR string);
void release_bmd_api_str(BMD_STR string);

IDeckLinkIterator *create_decklink_iterator(bool verbose = true, bool coinit = true);
void decklink_uninitialize();
bool blackmagic_api_version_check();
void print_decklink_version(void);

#endif // defined BLACKMAGIC_COMMON_H

