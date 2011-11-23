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
 
#include "jpeg_writer.h"
#include "jpeg_format_type.h"
#include "jpeg_encoder.h"
#include "jpeg_util.h"

/** Documented at declaration */
struct jpeg_writer*
jpeg_writer_create(struct jpeg_encoder* encoder)
{
    struct jpeg_writer* writer = malloc(sizeof(struct jpeg_writer));
    if ( writer == NULL )
        return NULL;
    
    // Allocate output buffer
    int buffer_size = 1000;
    buffer_size += encoder->param_image.width * encoder->param_image.height * encoder->param_image.comp_count * 2;
    writer->buffer = malloc(buffer_size * sizeof(uint8_t));
    if ( writer->buffer == NULL )
        return NULL;
    
    return writer;
}

/** Documented at declaration */
int
jpeg_writer_destroy(struct jpeg_writer* writer)
{
    assert(writer != NULL);
    assert(writer->buffer != NULL);
    free(writer->buffer);
    free(writer);
    return 0;
}

/**
 * Write SOI
 * 
 * @param writer  Writer structure
 * @return void
 */
void
jpeg_writer_write_soi(struct jpeg_writer* writer)
{
	jpeg_writer_emit_marker(writer, JPEG_MARKER_SOI);
}

/**
 * Write APP0 block
 * 
 * @param writer  Writer structure
 * @return void
 */
void jpeg_writer_write_app0(struct jpeg_writer* writer)
{
    // Length of APP0 block	(2 bytes)
    // Block ID			(4 bytes - ASCII "JFIF")
    // Zero byte			(1 byte to terminate the ID string)
    // Version Major, Minor	(2 bytes - 0x01, 0x01)
    // Units			(1 byte - 0x00 = none, 0x01 = inch, 0x02 = cm)
    // Xdpu			(2 bytes - dots per unit horizontal)
    // Ydpu			(2 bytes - dots per unit vertical)
    // Thumbnail X size		(1 byte)
    // Thumbnail Y size		(1 byte)
    jpeg_writer_emit_marker(writer, JPEG_MARKER_APP0);
    
    // Length
    jpeg_writer_emit_2byte(writer, 2 + 4 + 1 + 2 + 1 + 2 + 2 + 1 + 1);

    // Identifier: ASCII "JFIF"
    jpeg_writer_emit_byte(writer, 0x4A);
    jpeg_writer_emit_byte(writer, 0x46);
    jpeg_writer_emit_byte(writer, 0x49);
    jpeg_writer_emit_byte(writer, 0x46);
    jpeg_writer_emit_byte(writer, 0);

    // We currently emit version code 1.01 since we use no 1.02 features.
    // This may avoid complaints from some older decoders.
    // Major version
    jpeg_writer_emit_byte(writer, 1);
    // Minor version 
    jpeg_writer_emit_byte(writer, 1);
    // Pixel size information
    jpeg_writer_emit_byte(writer, 1);
    jpeg_writer_emit_2byte(writer, 300);
    jpeg_writer_emit_2byte(writer, 300);
    // No thumbnail image
    jpeg_writer_emit_byte(writer, 0);
    jpeg_writer_emit_byte(writer, 0);
}

/**
 * Write DQT block
 * 
 * @param encoder  Encoder structure
 * @param type  Component type for table retrieve
 * @return void
 */
void
jpeg_writer_write_dqt(struct jpeg_encoder* encoder, enum jpeg_component_type type)
{
	jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_DQT);
    
    // Length
	jpeg_writer_emit_2byte(encoder->writer, 67);
    
    // Index: Y component = 0, Cb or Cr component = 1
	jpeg_writer_emit_byte(encoder->writer, (int)type); 

    // Table changed from default with quality
	uint8_t* dqt = encoder->table_quantization[type].table_raw;
    
    // Emit table
	unsigned char qval;
	for ( int i = 0; i < 64; i++ )  {
        unsigned char qval = (unsigned char)((char)(dqt[jpeg_order_natural[i]]));
		jpeg_writer_emit_byte(encoder->writer, qval);
    }
}

/**
 * Currently support JPEG_MARKER_SOF0 baseline implementation
 * 
 * @param encoder  Encoder structure 
 * @return void
 */
void
jpeg_writer_write_sof0(struct jpeg_encoder* encoder)
{
	jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_SOF0);
    
    // Length
	jpeg_writer_emit_2byte(encoder->writer, 17);

    // Precision (bit depth)
	jpeg_writer_emit_byte(encoder->writer, 8);
    // Dimensions
	jpeg_writer_emit_2byte(encoder->writer, encoder->param_image.height);
	jpeg_writer_emit_2byte(encoder->writer, encoder->param_image.width);
    // Number of components
	jpeg_writer_emit_byte(encoder->writer, 3);

	// Component Y
	jpeg_writer_emit_byte(encoder->writer, 1);  // component index
	jpeg_writer_emit_byte(encoder->writer, 17); // (1 << 4) + 1 (sampling h: 1, v: 1)
	jpeg_writer_emit_byte(encoder->writer, 0);  // quantization table index

	// Component Cb
	jpeg_writer_emit_byte(encoder->writer, 2);  // component index
	jpeg_writer_emit_byte(encoder->writer, 17); // (1 << 4) + 1 (sampling h: 1, v: 1)
	jpeg_writer_emit_byte(encoder->writer, 1);  // quantization table index

	// Component Cr
	jpeg_writer_emit_byte(encoder->writer, 3);  // component index
	jpeg_writer_emit_byte(encoder->writer, 17); // (1 << 4) + 1 (sampling h: 1, v: 1)
	jpeg_writer_emit_byte(encoder->writer, 1);  // quantization table index
}

/**
 * Write DHT block
 * 
 * @param encoder  Encoder structure
 * @param type  Component type for table retrieve
 * @param is_ac  Flag if table AC or DC should be written
 * @return void
 */
void
jpeg_writer_write_dht(struct jpeg_encoder* encoder, enum jpeg_component_type comp_type, enum jpeg_huffman_type huff_type)
{
    // Get proper table and its index
    struct jpeg_table_huffman_encoder* table = NULL;
	int index;
	if ( comp_type == JPEG_COMPONENT_LUMINANCE ) {
		if ( huff_type == JPEG_HUFFMAN_AC ) {
			table = &encoder->table_huffman[comp_type][huff_type];
			index = 16;
		} else {
			table = &encoder->table_huffman[comp_type][huff_type];
			index = 0;
		}
	} else {
		if ( huff_type == JPEG_HUFFMAN_AC ) {
			table = &encoder->table_huffman[comp_type][huff_type];
			index = 17;
		} else {
			table = &encoder->table_huffman[comp_type][huff_type];
			index = 1;
		}
	}

	jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_DHT);

	int length = 0;
    for ( int i = 1; i <= 16; i++ )
		length += table->bits[i];

	jpeg_writer_emit_2byte(encoder->writer, length + 2 + 1 + 16);

    jpeg_writer_emit_byte(encoder->writer, index);

	for ( int i = 1; i <= 16; i++ )
		jpeg_writer_emit_byte(encoder->writer, table->bits[i]);
    
    // Varible-length
    for ( int i = 0; i < length; i++ )
		jpeg_writer_emit_byte(encoder->writer, table->huffval[i]);  
}

/**
 * Write restart interval
 * 
 * @param encoder  Encoder structure
 * @return void
 */
void
jpeg_writer_write_dri(struct jpeg_encoder* encoder)
{
	jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_DRI);
    
    // Length
	jpeg_writer_emit_2byte(encoder->writer, 4);

    // Restart interval
    jpeg_writer_emit_2byte(encoder->writer, encoder->param.restart_interval);
}

/** Documented at declaration */
void
jpeg_writer_write_header(struct jpeg_encoder* encoder)
{        
	jpeg_writer_write_soi(encoder->writer);
	jpeg_writer_write_app0(encoder->writer);
    
	jpeg_writer_write_dqt(encoder, JPEG_COMPONENT_LUMINANCE);      
	jpeg_writer_write_dqt(encoder, JPEG_COMPONENT_CHROMINANCE);
	
    jpeg_writer_write_sof0(encoder);
    
	jpeg_writer_write_dht(encoder, JPEG_COMPONENT_LUMINANCE, JPEG_HUFFMAN_DC);   // DC table for Y component
	jpeg_writer_write_dht(encoder, JPEG_COMPONENT_LUMINANCE, JPEG_HUFFMAN_AC);   // AC table for Y component
	jpeg_writer_write_dht(encoder, JPEG_COMPONENT_CHROMINANCE, JPEG_HUFFMAN_DC); // DC table for Cb or Cr component
	jpeg_writer_write_dht(encoder, JPEG_COMPONENT_CHROMINANCE, JPEG_HUFFMAN_AC); // AC table for Cb or Cr component
    
    jpeg_writer_write_dri(encoder);
}

/** Documented at declaration */
void
jpeg_writer_write_scan_header(struct jpeg_encoder* encoder, int index, enum jpeg_component_type type)
{        
    jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_SOS);

    // Length
	int length = 2 + 1 + 2 + 3;
	jpeg_writer_emit_2byte(encoder->writer, length);

    // Component count
	jpeg_writer_emit_byte(encoder->writer, 1);

    if ( type == JPEG_COMPONENT_LUMINANCE ) {
        // Component Y
        jpeg_writer_emit_byte(encoder->writer, index + 1); // index
        jpeg_writer_emit_byte(encoder->writer, 0);         // (0 << 4) | 0
    } else {
        // Component Cb or Cr
        jpeg_writer_emit_byte(encoder->writer, index + 1); // index
        jpeg_writer_emit_byte(encoder->writer, 0x11);      // (1 << 4) | 1
    }

	jpeg_writer_emit_byte(encoder->writer, 0);    // Ss
	jpeg_writer_emit_byte(encoder->writer, 0x3F); // Se
	jpeg_writer_emit_byte(encoder->writer, 0);    // Ah/Al
}