/**
 * @file   compat/endian.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * endian.h compat macro for macOS, Windows
 *
 * It will be likely possible to remove later, at least for macOS, because it is
 * required by POSIX 2024.
 */
/*
 * Copyright (c) 2025 CESNET, zájmové sdružení právnických osob
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

#ifndef COMPAT_ENDIAN_H_A5D7C443_895D_465E_A46D_BF5E6DAA833F
#define COMPAT_ENDIAN_H_A5D7C443_895D_465E_A46D_BF5E6DAA833F

#ifdef __APPLE__
        // https://github.com/zeromq/zmqpp/issues/164#issuecomment-246346344
        #include <libkern/OSByteOrder.h>
        #ifndef be16toh
                #define be16toh(x) OSSwapBigToHostInt16(x)
                #define be32toh(x) OSSwapBigToHostInt32(x)
                #define be64toh(x) OSSwapBigToHostInt64(x)

                #define htobe16(x) OSSwapHostToBigInt16(x)
                #define htobe32(x) OSSwapHostToBigInt32(x)
                #define htobe64(x) OSSwapHostToBigInt64(x)

                #define htole16(x) OSSwapHostToLittleInt16(x)
                #define htole32(x) OSSwapHostToLittleInt32(x)
                #define htole64(x) OSSwapHostToLittleInt64(x)

                #define le16toh(x) OSSwapLittleToHostInt16(x)
                #define le32toh(x) OSSwapLittleToHostInt32(x)
                #define le64toh(x) OSSwapLittleToHostInt64(x)
        #endif
        // BYTE_ORDER, LITTLE_ENDIAN, BIG_ENDIAN
        #include <machine/endian.h>

#elif defined _WIN32
        // little endian archs
        #if defined(_M_AMD64) || defined(_M_IX86) || defined(_M_ARM) || \
            defined(_M_ARM64) || \
            (defined(__BYTE_ORDER__) && \
             __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
                #include <stdlib.h>
                #ifndef be16toh
                        #define be16toh(x) _byteswap_ushort(x)
                        #define be32toh(x) _byteswap_ulong(x)
                        #define be64toh(x) _byteswap_uint64(x)

                        #define htobe16(x) _byteswap_ushort(x)
                        #define htobe32(x) _byteswap_ulong(x)
                        #define htobe64(x) _byteswap_uint64(x)

                        #define htole16(x) (x)
                        #define htole32(x) (x)
                        #define htole64(x) (x)

                        #define le16toh(x) (x)
                        #define le32toh(x) (x)
                        #define le64toh(x) (x)
                #endif
                #ifndef BYTE_ORDER
                        #define LITTLE_ENDIAN 1234
                        #define BIG_ENDIAN    4321
                        #define BYTE_ORDER    LITTLE_ENDIAN
                #endif
        #else
                #error "Unrecognized Win32 configuration (big-endian or unhandled)"
        #endif

#else
        #include <endian.h>
#endif

#endif // defined COMPAT_ENDIAN_H_A5D7C443_895D_465E_A46D_BF5E6DAA833F
