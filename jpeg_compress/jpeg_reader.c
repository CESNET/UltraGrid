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
 
#include "jpeg_reader.h"
#include "jpeg_decoder.h"
#include "jpeg_format_type.h"
#include "jpeg_util.h"

/** Documented at declaration */
struct jpeg_reader*
jpeg_reader_create()
{
    struct jpeg_reader* reader = malloc(sizeof(struct jpeg_reader));
    if ( reader == NULL )
        return NULL;
    
    return reader;
}

/** Documented at declaration */
int
jpeg_reader_destroy(struct jpeg_reader* reader)
{
    assert(reader != NULL);
    free(reader);
    return 0;
}

/**
 * Read byte from image data
 * 
 * @param image
 * @return byte
 */
#define jpeg_reader_read_byte(image) \
    (uint8_t)(*(image)++)

/**
 * Read two-bytes from image data
 * 
 * @param image
 * @return 2 bytes
 */    
#define jpeg_reader_read_2byte(image) \
    (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
    image += 2;

/**
 * Read marker from image data
 * 
 * @param image
 * @return marker code
 */
int
jpeg_reader_read_marker(uint8_t** image)
{
    if( jpeg_reader_read_byte(*image) != 0xFF )
        return -1;
    int marker = jpeg_reader_read_byte(*image);
    return marker;
}

/**
 * Skip marker content (read length and that much bytes - 2)
 * 
 * @param image
 * @return void
 */
void
jpeg_reader_skip_marker_content(uint8_t** image)
{
    int length = (int)jpeg_reader_read_2byte(*image);

    *image += length - 2;
}

/**
 * Read application ifno block from image
 * 
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_app0(uint8_t** image)
{
    int length = (int)jpeg_reader_read_2byte(*image);
    if ( length != 16 ) {
        fprintf(stderr, "Error: APP0 marker length should be 16 but %d was presented!\n", length);
        return -1;
    }

    char jfif[4];
    jfif[0] = jpeg_reader_read_byte(*image);
    jfif[1] = jpeg_reader_read_byte(*image);
    jfif[2] = jpeg_reader_read_byte(*image);
    jfif[3] = jpeg_reader_read_byte(*image);
    jfif[4] = jpeg_reader_read_byte(*image);
    if ( strcmp(jfif, "JFIF") != 0 ) {
        fprintf(stderr, "Error: APP0 marker identifier should be 'JFIF' but '%s' was presented!\n", jfif);
        return -1;
    }

    int version_major = jpeg_reader_read_byte(*image);
    int version_minor = jpeg_reader_read_byte(*image);
    if ( version_major != 1 || version_minor != 1 ) {
        fprintf(stderr, "Error: APP0 marker version should be 1.1 but %d.%d was presented!\n", version_major, version_minor);
        return -1;
    }
    
    int pixel_units = jpeg_reader_read_byte(*image);
    int pixel_xdpu = jpeg_reader_read_2byte(*image);
    int pixel_ydpu = jpeg_reader_read_2byte(*image);
    int thumbnail_width = jpeg_reader_read_byte(*image);
    int thumbnail_height = jpeg_reader_read_byte(*image);

    return 0;
}

/**
 * Read quantization table definition block from image
 * 
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_dqt(struct jpeg_decoder* decoder, uint8_t** image)
{    
    int length = (int)jpeg_reader_read_2byte(*image);
    length -= 2;
    
    if ( length != 65 ) {
        fprintf(stderr, "Error: DQT marker length should be 65 but %d was presented!\n", length);
        return -1;
    }
    
    int index = jpeg_reader_read_byte(*image);
    struct jpeg_table_quantization* table;
    if( index == 0 ) {
        table = &decoder->table_quantization[JPEG_COMPONENT_LUMINANCE];
    } else if ( index == 1 ) {
        table = &decoder->table_quantization[JPEG_COMPONENT_CHROMINANCE];
    } else {
        fprintf(stderr, "Error: DQT marker index should be 0 or 1 but %d was presented!\n", index);
        return -1;
    }

    for ( int i = 0; i < 64; i++ ) {
        table->table_raw[jpeg_order_natural[i]] = jpeg_reader_read_byte(*image);
    }
    
    // Prepare quantization table for read raw table
    jpeg_table_quantization_decoder_compute(table);
    
    return 0;
}

/**
 * Read start of frame block from image
 * 
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_sof0(struct jpeg_decoder* decoder, uint8_t** image)
{    
    int length = (int)jpeg_reader_read_2byte(*image);
    if ( length < 6 ) {
        fprintf(stderr, "Error: SOF0 marker length should be greater than 6 but %d was presented!\n", length);
        return -1;
    }
    length -= 2;

    int precision = (int)jpeg_reader_read_byte(*image);
    if ( precision != 8 ) {
        fprintf(stderr, "Error: SOF0 marker precision should be 8 but %d was presented!\n", precision);
        return -1;
    }
    int height = (int)jpeg_reader_read_2byte(*image);
    int width = (int)jpeg_reader_read_2byte(*image);
    int comp_count = (int)jpeg_reader_read_byte(*image);
    jpeg_decoder_init(decoder, width, height, comp_count);
    if ( width != decoder->param_image.width || height != decoder->param_image.height ) {
        fprintf(stderr, "Error: SOF0 marker image size should be %dx%d but %dx%d was presented!\n", decoder->param_image.width, decoder->param_image.height, width, height);
        return -1;
    }
    if ( comp_count != decoder->param_image.comp_count ) {
        fprintf(stderr, "Error: SOF0 marker component count should be %d but %d was presented!\n", decoder->param_image.comp_count, comp_count);
        return -1;
    }
    length -= 6;

    for ( int comp = 0; comp < comp_count; comp++ ) {
        int index = (int)jpeg_reader_read_byte(*image);
        if ( index != (comp + 1) ) {
            fprintf(stderr, "Error: SOF0 marker component %d id should be %d but %d was presented!\n", comp, comp + 1, index);
            return -1;
        }
        int sampling = (int)jpeg_reader_read_byte(*image);
        int sampling_h = (sampling >> 4) & 15;
        int sampling_v = sampling & 15;
        if ( sampling_h != 1 || sampling_v != 1 ) {
            fprintf(stderr, "Error: SOF0 marker component %d sampling factor %dx%d is not supported!\n", comp, sampling_h, sampling_v);
            return -1;
        }
        int table_index = (int)jpeg_reader_read_byte(*image);
        if ( comp == 0 && table_index != 0 ) {
            fprintf(stderr, "Error: SOF0 marker component Y should have quantization table index 0 but %d was presented!\n", table_index);
            return -1;
        }
        if ( (comp == 1 || comp == 2) && table_index != 1 ) {
            fprintf(stderr, "Error: SOF0 marker component Cb or Cr should have quantization table index 1 but %d was presented!\n", table_index);
            return -1;
        }
        length -= 3;
    }
    
    // Check length
    if ( length > 0 ) {
        fprintf(stderr, "Warning: SOF0 marker contains %d more bytes than needed!\n", length);
        *image += length;
    }
    
    return 0;
}

/**
 * Read huffman table definition block from image
 * 
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_dht(struct jpeg_decoder* decoder, uint8_t** image)
{    
    int length = (int)jpeg_reader_read_2byte(*image);
    length -= 2;
    
    int index = jpeg_reader_read_byte(*image);
    struct jpeg_table_huffman_decoder* table = NULL;
    struct jpeg_table_huffman_decoder* d_table = NULL;
    switch(index) {
    case 0:
        table = &decoder->table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_DC];
        d_table = decoder->d_table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_DC];
        break;
    case 16:
        table = &decoder->table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_AC];
        d_table = decoder->d_table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_AC];
        break;
    case 1:
        table = &decoder->table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_DC];
        d_table = decoder->d_table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_DC];
        break;
    case 17:
        table = &decoder->table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_AC];
        d_table = decoder->d_table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_AC];
        break;
    default:
        fprintf(stderr, "Error: DHT marker index should be 0, 1, 16 or 17 but %d was presented!\n", index);
        return -1;
    }
    length -= 1;
    
    // Read in bits[]
    table->bits[0] = 0;
    int count = 0;
    for ( int i = 1; i <= 16; i++ ) {
        table->bits[i] = jpeg_reader_read_byte(*image);
        count += table->bits[i];
        if ( length > 0 ) {
            length--;
        } else {
            fprintf(stderr, "Error: DHT marker unexpected end when reading bit counts!\n", index);
            return -1;
        }
    }   

    // Read in huffval
    for ( int i = 0; i < count; i++ ){
        table->huffval[i] = jpeg_reader_read_byte(*image);
        if ( length > 0 ) {
            length--;
        } else {
            fprintf(stderr, "Error: DHT marker unexpected end when reading huffman values!\n", index);
            return -1;
        }
    }
    
    // Check length
    if ( length > 0 ) {
        fprintf(stderr, "Warning: DHT marker contains %d more bytes than needed!\n", length);
        *image += length;
    }
    
    // Compute huffman table for read values
    jpeg_table_huffman_decoder_compute(table, d_table);
    
    return 0;
}

/**
 * Read restart interval block from image
 * 
 * @param decoder
 * @param image
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_dri(struct jpeg_decoder* decoder, uint8_t** image)
{
    int length = (int)jpeg_reader_read_2byte(*image);
    if ( length != 4 ) {
        fprintf(stderr, "Error: DRI marker length should be 4 but %d was presented!\n", length);
        return -1;
    }
    
    if ( decoder->restart_interval != 0 ) {
        fprintf(stderr, "Error: DRI marker can't redefine restart interval!");
        fprintf(stderr, "This may be caused when more DRI markers are presented which is not supported!\n");
        return -1;
    }
    
    decoder->restart_interval = jpeg_reader_read_2byte(*image);
    
    return 0;
}

/**
 * Read start of scan block from image
 * 
 * @param image
 * @param image_end
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_reader_read_sos(struct jpeg_decoder* decoder, uint8_t** image, uint8_t* image_end)
{    
    int length = (int)jpeg_reader_read_2byte(*image);
    length -= 2;
    
    int comp_count = (int)jpeg_reader_read_byte(*image);
    if ( comp_count != 1 ) {
        fprintf(stderr, "Error: SOS marker component count %d is not supported!\n", comp_count);
        return -1;
    }
    
    // Collect the component-spec parameters
    for ( int comp = 0; comp < comp_count; comp++ ) 
    {
        int index = (int)jpeg_reader_read_byte(*image);
        int table = (int)jpeg_reader_read_byte(*image);
        int table_dc = (table >> 4) & 15;
        int table_ac = table & 15;
        
        if ( index == 1 && (table_ac != 0 || table_dc != 0) ) {
            fprintf(stderr, "Error: SOS marker for Y should have huffman tables 0,0 but %d,%d was presented!\n", table_dc, table_ac);
            return -1;
        }
        if ( (index == 2 || index == 3) && (table_ac != 1 || table_dc != 1) ) {
            fprintf(stderr, "Error: SOS marker for Cb or Cr should have huffman tables 1,1 but %d,%d was presented!\n", table_dc, table_ac);
            return -1;
        }
    }

    // Collect the additional scan parameters Ss, Se, Ah/Al.
    int Ss = (int)jpeg_reader_read_byte(*image);
    int Se = (int)jpeg_reader_read_byte(*image);
    int Ax = (int)jpeg_reader_read_byte(*image);
    int Ah = (Ax >> 4) & 15;
    int Al = (Ax) & 15;
    
    // Check maximum scan count
    if ( decoder->scan_count >= 3 ) {
        fprintf(stderr, "Error: SOS marker reached maximum number of scans (3)!\n");
        return -1;
    }
    // Get scan structure
    struct jpeg_decoder_scan* scan = &decoder->scan[decoder->scan_count];
    decoder->scan_count++;
    
    // Scan segments begin at the end of previous scan segments or from zero index
    scan->segment_index = decoder->segment_count;
    // Every scan has first segment
    scan->segment_count = 0;
    decoder->data_scan_index[scan->segment_index + scan->segment_count] = decoder->data_scan_size;
    scan->segment_count++;
    // Read scan data
    uint8_t byte = 0;
    uint8_t byte_previous = 0;
    do {
        byte_previous = byte;
        byte = jpeg_reader_read_byte(*image);
        decoder->data_scan[decoder->data_scan_size] = byte;
        //printf("set byte %d = 0x%X\n", &decoder->data_scan[decoder->data_scan_size], (unsigned char)byte);
        decoder->data_scan_size++;        
        
        // Check markers
        if ( byte_previous == 0xFF ) {
            // Check restart marker
            if ( byte >= JPEG_MARKER_RST0 && byte <= JPEG_MARKER_RST7 ) {
                decoder->data_scan_size -= 2;
                
                // Set data start index for next scan segment
                decoder->data_scan_index[scan->segment_index + scan->segment_count] = decoder->data_scan_size;
                scan->segment_count++;
                //printf("restart marker 0x%X (revert to %d)\n", (unsigned char)byte, &decoder->data_scan[decoder->data_scan_size]);
            }
            // Check scan end
            else if ( byte == JPEG_MARKER_EOI || byte == JPEG_MARKER_SOS ) {
                *image -= 2;
                decoder->data_scan_size -= 2;
                
                // Add scan segment count to decoder segment count
                decoder->segment_count += scan->segment_count;
                
                //printf("end marker 0x%X (revert to %d)\n", (unsigned char)byte, &decoder->data_scan[decoder->data_scan_size]);
                return 0;
            }
        }
    } while( *image < image_end );
    
    fprintf(stderr, "Error: JPEG data unexpected ended while reading SOS marker!\n");
    
    return -1;
}

/** Documented at declaration */
int
jpeg_reader_read_image(struct jpeg_decoder* decoder, uint8_t* image, int image_size)
{
    // Setup decoder
    decoder->scan_count = 0;
    decoder->data_scan_size = 0;
    decoder->segment_count = 0; // Total segment count for all scans
    decoder->restart_interval = 0;
    
    // Get image end
    uint8_t* image_end = image + image_size;
    
    // Check first SOI marker
    int marker_soi = jpeg_reader_read_marker(&image);
    if ( marker_soi != JPEG_MARKER_SOI ) {
        fprintf(stderr, "Error: JPEG data should begin with SOI marker, but marker %s was found!\n", jpeg_marker_name(marker_soi));
        return -1;
    }
        
    int eoi_presented = 0;
    while ( eoi_presented == 0 ) {
        // Read marker
        int marker = jpeg_reader_read_marker(&image);

        // Read more info according to the marker
        // the order of cases is in jpg file made by ms paint
        switch (marker) 
        {
        case JPEG_MARKER_APP0:
            if ( jpeg_reader_read_app0(&image) != 0 )
                return -1;
            break;
        case JPEG_MARKER_APP1:
        case JPEG_MARKER_APP2:
        case JPEG_MARKER_APP3:
        case JPEG_MARKER_APP4:
        case JPEG_MARKER_APP5:
        case JPEG_MARKER_APP6:
        case JPEG_MARKER_APP7:
        case JPEG_MARKER_APP8:
        case JPEG_MARKER_APP9:
        case JPEG_MARKER_APP10:
        case JPEG_MARKER_APP11:
        case JPEG_MARKER_APP12:
        case JPEG_MARKER_APP13:
        case JPEG_MARKER_APP14:
        case JPEG_MARKER_APP15:
            fprintf(stderr, "Warning: JPEG data contains not supported %s marker\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            break;
            
        case JPEG_MARKER_DQT:
            if ( jpeg_reader_read_dqt(decoder, &image) != 0 )
                return -1;
            break;

        case JPEG_MARKER_SOF0: 
            // Baseline
            if ( jpeg_reader_read_sof0(decoder, &image) != 0 )
                return -1;
            break;
        case JPEG_MARKER_SOF1:
            // Extended sequential with Huffman coder
            fprintf(stderr, "Warning: Reading SOF1 as it was SOF0 marker (should work but verify it)!\n", jpeg_marker_name(marker));
            if ( jpeg_reader_read_sof0(decoder, &image) != 0 )
                return -1;
            break;
        case JPEG_MARKER_SOF2:
            fprintf(stderr, "Error: Marker SOF2 (Progressive with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF3:
            fprintf(stderr, "Error: Marker SOF3 (Lossless with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF5:
            fprintf(stderr, "Error: Marker SOF5 (Differential sequential with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF6:
            fprintf(stderr, "Error: Marker SOF6 (Differential progressive with Huffman coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF7:
            fprintf(stderr, "Error: Marker SOF7 (Extended lossless with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_JPG:
            fprintf(stderr, "Error: Marker JPG (Reserved for JPEG extensions ) is not supported!");
            return -1;
        case JPEG_MARKER_SOF10:
            fprintf(stderr, "Error: Marker SOF10 (Progressive with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF11:
            fprintf(stderr, "Error: Marker SOF11 (Lossless with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF13:
            fprintf(stderr, "Error: Marker SOF13 (Differential sequential with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF14:
            fprintf(stderr, "Error: Marker SOF14 (Differential progressive with Arithmetic coding) is not supported!");
            return -1;
        case JPEG_MARKER_SOF15:
            fprintf(stderr, "Error: Marker SOF15 (Differential lossless with Arithmetic coding) is not supported!");
            return -1;
            
        case JPEG_MARKER_DHT:
            if ( jpeg_reader_read_dht(decoder, &image) != 0 )
                return -1;
            break;
            
        case JPEG_MARKER_DRI:
            if ( jpeg_reader_read_dri(decoder, &image) != 0 )
                return -1;
            break;

        case JPEG_MARKER_SOS:
            if ( jpeg_reader_read_sos(decoder, &image, image_end) != 0 )
                return -1;
            break;
            
        case JPEG_MARKER_EOI:
            eoi_presented = 1;
            break;

        case JPEG_MARKER_COM:
        case JPEG_MARKER_DAC:
        case JPEG_MARKER_DNL:
            fprintf(stderr, "Warning: JPEG data contains not supported %s marker\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            break;
            
        default:   
            fprintf(stderr, "Error: JPEG data contains not supported %s marker!\n", jpeg_marker_name(marker));
            jpeg_reader_skip_marker_content(&image);
            return -1;
        }
    }
    
    // Check EOI marker
    if ( eoi_presented == 0 ) {
        fprintf(stderr, "Error: JPEG data should end with EOI marker!\n");
        return -1;
    }
    
    return 0;
}