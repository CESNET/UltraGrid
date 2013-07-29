#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif // HAVE_CONFIG_H
#include "debug.h"
#include "perf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
//#include "rtp/rtpdec_h264.h"
#include "video_capture/rtsp.h"
#include "rtp/pbuf.h"

static const uint8_t start_sequence[] = { 0, 0, 0, 1 };

int decode_frame_h264(struct coded_data *cdata, void *rx_data);

int decode_frame_h264(struct coded_data *cdata, void *rx_data) {
	rtp_packet *pckt = NULL;
	struct coded_data *orig = cdata;

	uint8_t nal;
	uint8_t type;

	int pass;
	int total_length = 0;

	char *dst = NULL;
	int src_len;

	struct recieved_data *buffers = (struct recieved_data *) rx_data;

	for (pass = 0; pass < 2; pass++) {

		if (pass > 0) {
			cdata = orig;
			buffers->buffer_len = total_length;
			dst = buffers->frame_buffer + total_length;
		}

		while (cdata != NULL) {
			pckt = cdata->data;

			if (pckt->pt != PT_H264) {
				error_msg("Wrong Payload type: %u\n", pckt->pt);
				return FALSE;
			}

			nal = (uint8_t) pckt->data[0];
			type = nal & 0x1f;

			if (type >= 1 && type <= 23) {
				type = 1;
			}

			const uint8_t *src = NULL;

			switch (type) {
			case 0:
			case 1:
				if (pass == 0) {
					debug_msg("NAL type 1\n");
					total_length += sizeof(start_sequence) + pckt->data_len;
				} else {
					dst -= pckt->data_len + sizeof(start_sequence);
					memcpy(dst, start_sequence, sizeof(start_sequence));
					memcpy(dst + sizeof(start_sequence), pckt->data, pckt->data_len);
					unsigned char *dst2 = (unsigned char *)dst;
					debug_msg("\n\nFirst six bytes: %02x %02x %02x %02x %02x %02x\n", dst2[0], dst2[1], dst2[2], dst2[3], dst2[4], dst2[5]);
					debug_msg("Last six bytes: %02x %02x %02x %02x %02x %02x\n", dst2[pckt->data_len + sizeof(start_sequence) - 6],
							dst2[pckt->data_len + sizeof(start_sequence) - 5],
							dst2[pckt->data_len + sizeof(start_sequence) - 4],
							dst2[pckt->data_len + sizeof(start_sequence) - 3],
							dst2[pckt->data_len + sizeof(start_sequence) - 2],
							dst2[pckt->data_len + sizeof(start_sequence) - 1]);
					debug_msg("NAL size: %d\n\n", pckt->data_len + sizeof(start_sequence));
				}
				break;
				case 24:
				src = (const uint8_t *) pckt->data;
				src_len = pckt->data_len;

				src++;
				src_len--;

				while (src_len > 2) {
					//TODO: Not properly tested
					uint16_t nal_size;
					memcpy(&nal_size, src, sizeof(uint16_t));

					src += 2;
					src_len -= 2;

					if (nal_size <= src_len) {
						if (pass == 0) {
							total_length += sizeof(start_sequence) + nal_size;
						} else {
							dst -= nal_size + sizeof(start_sequence);
							memcpy(dst, start_sequence, sizeof(start_sequence));
							memcpy(dst + sizeof(start_sequence), src, nal_size);
						}
					} else {
						error_msg("NAL size exceeds length: %u %d\n", nal_size, src_len);
						return FALSE;
					}
					src += nal_size;
					src_len -= nal_size;

					if (src_len < 0) {
						error_msg("Consumed more bytes than we got! (%d)\n", src_len);
						return FALSE;
					}
				}
				break;

				case 25:
				case 26:
				case 27:
				case 29:
				error_msg("Unhandled NAL type\n");
				return FALSE;
				case 28:
				src = (const uint8_t *) pckt->data;
				src_len = pckt->data_len;

				src++;
				src_len--;

				if (src_len > 1) {
					uint8_t fu_header = *src;
					uint8_t start_bit = fu_header >> 7;
					//uint8_t end_bit 		= (fu_header & 0x40) >> 6;
					uint8_t nal_type = fu_header & 0x1f;
					uint8_t reconstructed_nal;

					// Reconstruct this packet's true nal; only the data follows.
					/* The original nal forbidden bit and NRI are stored in this
					 * packet's nal. */
					reconstructed_nal = nal & 0xe0;
					reconstructed_nal |= nal_type;

					// skip the fu_header
					src++;
					src_len--;

					if (pass == 0) {
						if (start_bit) {
							total_length += sizeof(start_sequence) + sizeof(reconstructed_nal) + src_len;
						} else {
							total_length += src_len;
						}
					} else {
						if (start_bit) {
							dst -= sizeof(start_sequence) + sizeof(reconstructed_nal) + src_len;
							memcpy(dst, start_sequence, sizeof(start_sequence));
							memcpy(dst + sizeof(start_sequence), &reconstructed_nal, sizeof(reconstructed_nal));
							memcpy(dst + sizeof(start_sequence) + sizeof(reconstructed_nal), src, src_len);
						} else {
							dst -= src_len;
							memcpy(dst, src, src_len);
						}
					}
				} else {
					error_msg("Too short data for FU-A H264 RTP packet\n");
					return FALSE;
				}
				break;
				default:
				error_msg("Unknown NAL type\n");
				return FALSE;
			}
			cdata = cdata->nxt;
		}
	}
	return TRUE;
}
