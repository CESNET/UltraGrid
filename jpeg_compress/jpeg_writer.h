/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef JPEG_WRITER
#define JPEG_WRITER

#include "jpeg_type.h"

/** JPEG encoder structure predeclaration */
struct jpeg_encoder;

/** JPEG writer structure */
struct jpeg_writer 
{
    // Output buffer
    uint8_t* buffer;
    // Output buffer current position
    uint8_t* buffer_current;
};

/**
 * Create JPEG writer
 * 
 * @return writer structure if succeeds, otherwise NULL
 */
struct jpeg_writer*
jpeg_writer_create(struct jpeg_encoder* encoder);

/**
 * Destroy JPEG writer
 * 
 * @param writer  Writer structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_writer_destroy(struct jpeg_writer* writer);

/**
 * Write one byte to file
 * 
 * @param writer  Writer structure
 * @param value  Byte value to write
 * @return void
 */
#define jpeg_writer_emit_byte(writer, value) { \
    *writer->buffer_current = (uint8_t)(value); \
    writer->buffer_current++; }
    
/**
 * Write two bytes to file
 * 
 * @param writer  Writer structure
 * @param value  Two-byte value to write
 * @return void
 */
#define jpeg_writer_emit_2byte(writer, value) { \
    *writer->buffer_current = (uint8_t)(((value) >> 8) & 0xFF); \
    writer->buffer_current++; \
    *writer->buffer_current = (uint8_t)((value) & 0xFF); \
    writer->buffer_current++; }
    
/**
 * Write marker to file
 * 
 * @param writer  Writer structure
 * @oaran marker  Marker to write (JPEG_MARKER_...)
 * @return void
 */
#define jpeg_writer_emit_marker(writer, marker) { \
    *writer->buffer_current = 0xFF;\
    writer->buffer_current++; \
    *writer->buffer_current = (uint8_t)(marker); \
    writer->buffer_current++; }
    
/**
 * Write JPEG header (write soi, app0, Y_dqt, CbCr_dqt, sof, 4 * dht blocks)
 * 
 * @param encoder  Encoder structure
 * @return void
 */
void
jpeg_writer_write_header(struct jpeg_encoder* encoder);

/**
 * Write scan header for one component
 * 
 * @param encoder  Encoder structure
 * @param type  Component scan type
 * @return void
 */
void
jpeg_writer_write_scan_header(struct jpeg_encoder* encoder, int index, enum jpeg_component_type type);

#endif // JPEG_WRITER
