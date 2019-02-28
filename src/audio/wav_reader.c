#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "audio/wav_reader.h"
#include "debug.h"

#define READ_N(buf, len) \
        if (fread(buf, len, 1, wav_file) != 1) {\
                log_msg(LOG_LEVEL_ERROR, "[WAV] Read error: %s.\n", strerror(errno));\
                return WAV_HDR_PARSE_READ_ERROR;\
        }

static int read_fmt_chunk(FILE *wav_file, struct wav_metadata *metadata)
{
        uint16_t format;
        READ_N(&format, 2);
        if (format != 0x0001) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected format 0x0001, 0x%04d given.\n", format);
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

        return WAV_HDR_PARSE_OK;
}

int read_wav_header(FILE *wav_file, struct wav_metadata *metadata)
{
        char buffer[16];
        uint32_t chunk_size;
        rewind(wav_file);

        READ_N(buffer, 4);
        if(strncmp(buffer, "RIFF", 4) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected RIFF chunk, %.4s given.\n", buffer);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        // this is the length of the rest of the file, we may ignore
        READ_N(&chunk_size, 4);

        READ_N(buffer, 4);
        if (strncmp(buffer, "WAVE", 4) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected WAVE chunk, %.4s given.\n", buffer);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        bool found_data_chunk = false;
        bool found_fmt_chunk = false;
        while (fread(buffer, 4, 1, wav_file) == 1) {
                READ_N(&chunk_size, 4);
                if (strncmp(buffer, "data", 4) == 0) {
                        found_data_chunk = true;
                        metadata->data_size = chunk_size;
                        metadata->data_offset = ftell(wav_file);
                        fseek(wav_file, chunk_size, SEEK_CUR);
                } else if (strncmp(buffer, "fmt ", 4) == 0) {
                        found_fmt_chunk = true;
                        if (chunk_size != 16) {
                                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected fmt chunk size 16, %d given.\n", chunk_size);
                                return WAV_HDR_PARSE_WRONG_FORMAT;
                        }
                        int rc = read_fmt_chunk(wav_file, metadata);
                        if (rc != WAV_HDR_PARSE_OK) {
                                return rc;
                        }
                } else if (strncmp(buffer, "LIST", 4) == 0) {
                        log_msg(LOG_LEVEL_DEBUG, "[WAV] Skipping LIST chunk.\n");
                        fseek(wav_file, chunk_size, SEEK_CUR);
                } else if (strncmp(buffer, "JUNK", 4) == 0) {
                        log_msg(LOG_LEVEL_DEBUG, "[WAV] Skipping JUNK chunk.\n");
                        fseek(wav_file, chunk_size, SEEK_CUR);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[WAV] Unknown chunk \"%4s\" found!\n", buffer);
                        return WAV_HDR_PARSE_WRONG_FORMAT;
                }
        }

        if (!found_data_chunk) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Data chunk not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        if (!found_fmt_chunk) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Fmt chunk not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        log_msg(LOG_LEVEL_VERBOSE, "[WAV] File parsed correctly - length %u bytes, offset %u.\n",
                        metadata->data_size, metadata->data_offset);
        fseek(wav_file, metadata->data_offset, SEEK_SET);

        return WAV_HDR_PARSE_OK;
}

const char *get_wav_error(int errcode)
{
        switch(errcode) {
                case WAV_HDR_PARSE_OK:
                        return "Wav header OK";
                case WAV_HDR_PARSE_READ_ERROR:
                        return "Premature end of WAV file";
                case WAV_HDR_PARSE_WRONG_FORMAT:
                        return "Wav header in wrong format";
                case WAV_HDR_PARSE_NOT_PCM:
                        return "Wav not in PCM";
                default:
                        log_msg(LOG_LEVEL_ERROR, "[WAV] Unknown error code %d passed!\n", errcode);
                        return "Unknown error";
        }
}

