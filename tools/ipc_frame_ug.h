#ifndef IPC_FRAME_UG_32ee5c748f3e
#define IPC_FRAME_UG_32ee5c748f3e

#include "ipc_frame.h"
#include "types.h"

struct video_frame;

/**
 * @brief Make an ipc frame from ultragrid frame
 *
 * Data is copied, src frame can be freed afterwards.
 *
 * @param[out] dst            destination ipc frame
 * @param[in]  src            source ultragrid frame
 * @param[in]  codec          codec to convert into, VIDEO_CODEC_NONE for no conversion
 * @param[in]  scale_factor   frame dimensions are divided by this amount. 0 for original size
 */
bool ipc_frame_from_ug_frame(struct Ipc_frame *dst,
		const struct video_frame *src,
		codec_t codec,
		unsigned scale_factor);

bool ipc_frame_write_to_fd(const struct Ipc_frame *f, int fd);

int ipc_frame_get_scale_factor(int src_w, int src_h, int target_w, int target_h);

#endif //IPC_FRAME_UG_32ee5c748f3e
