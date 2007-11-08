struct video_compress;

struct video_compress * initialize_video_compression(void);
void compress_data(void *args, struct video_frame * tx);
static void compress_thread(void *args);
void compress_exit(void *args);
