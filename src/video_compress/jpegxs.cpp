#include <memory>

#include "lib_common.h"
#include "video_compress.h"
#include <svt-jpegxs/SvtJpegxsEnc.h>

using std::shared_ptr;

namespace {
struct state_video_compress_jpegxs;

struct state_video_compress_jpegxs {
private:
        state_video_compress_jpegxs(struct module *parent, const char *opts);
public:

        void push(std::shared_ptr<video_frame> in_frame);
        std::shared_ptr<video_frame> pop();
};

void *
jpegxs_compress_init(struct module *parent, const char *opts)
{
        struct state_video_compress_jpegxs *s;
        // TODO
}

shared_ptr<video_frame> jpegxs_compress(void *state, shared_ptr<video_frame> tx)
{
        // TODO
}

void state_video_compress_jpegxs::push(std::shared_ptr<video_frame> in_frame)
{
        // TODO
}

std::shared_ptr<video_frame> state_video_compress_jpegxs::pop()
{
        // TODO
}

static auto jpegxs_compress_push(void *state, std::shared_ptr<video_frame> in_frame) {
        static_cast<struct state_video_compress_jpegxs *>(state)->push(std::move(in_frame));
}

static auto jpegxs_compress_pull(void *state) {
        return static_cast<struct state_video_compress_jpegxs *>(state)->pop();
}

static void
jpegxs_compress_done(void *state)
{
        delete (struct state_video_compress_jpegxs *) state;
}

const struct video_compress_info jpegxs_info = {
        jpegxs_compress_init, // jpegxs_compress_init
        jpegxs_compress_done, // jpegxs_compress_done
        NULL, // jpegxs_compress (synchronous)
        NULL,
        NULL, // jpegxs_compress_push (asynchronous)
        NULL, //  jpegxs_compress_pull
        NULL,
        NULL,
        NULL // get_jpegxs_module_info
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}