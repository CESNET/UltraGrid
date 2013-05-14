#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "audio/wav_reader.h"

#define READ_N(buf, len) if (fread(buf, len, 1, wav_file) != 1) return WAV_HDR_PARSE_READ_ERROR;

int read_wav_header(FILE *wav_file, struct wav_metadata *metadata)
{
        char buffer[16];
        rewind(wav_file);

        READ_N(buffer, 4);
        if(strncmp(buffer, "RIFF", 4) != 0) {
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        uint32_t chunk_size;
        READ_N(&chunk_size, 4);

        READ_N(buffer, 4);
        if(strncmp(buffer, "WAVE", 4) != 0) {
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        // format chunk
        READ_N(buffer, 4);
        if(strncmp(buffer, "fmt ", 4) != 0) {
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        uint32_t fmt_chunk_size;
        READ_N(&fmt_chunk_size, 4);
        if(fmt_chunk_size != 16) {
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        uint16_t format;
        READ_N(&format, 2);
        if(format != 0x0001) {
                return WAV_HDR_PARSE_NOT_PCM;
        }

        uint16_t ch_count;
        READ_N(&ch_count, 2);
        metadata->ch_count = ch_count;

        uint32_t sample_rate;
        READ_N(&sample_rate, sizeof(sample_rate));
        metadata->sample_rate = sample_rate;

        uint32_t avg_bytes_per_sec;
        READ_N(&avg_bytes_per_sec, sizeof(avg_bytes_per_sec));

        uint16_t block_align_offset;
        READ_N(&block_align_offset, sizeof(block_align_offset));

        uint16_t bits_per_sample;
        READ_N(&bits_per_sample, sizeof(bits_per_sample));
        metadata->bits_per_sample = bits_per_sample;

        // data chunk
        READ_N(buffer, 4);
        if(strncmp(buffer, "data", 4) != 0) {
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        uint32_t data_chunk_size;
        READ_N(&data_chunk_size, 4);
        metadata->data_size = data_chunk_size;

        metadata->data_offset = ftell(wav_file);

        return WAV_HDR_PARSE_OK;
}

void print_wav_error(int errcode)
{
        switch(errcode) {
                case WAV_HDR_PARSE_OK:
                        printf("Wav header OK.\n");
                        break;
                case WAV_HDR_PARSE_READ_ERROR:
                        fprintf(stderr, "Premature end of WAV file.\n");
                        break;
                case WAV_HDR_PARSE_WRONG_FORMAT:
                        fprintf(stderr, "Wav header in wrong format.\n");
                        break;
                case WAV_HDR_PARSE_NOT_PCM:
                        fprintf(stderr, "Wav not in PCM.\n");
                        break;
        }
}

