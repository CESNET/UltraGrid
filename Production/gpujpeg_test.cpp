#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

extern "C" int gpujpeg_test_simple();

#if defined HAVE_GPUJPEG
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "host.h"
#include "module.h"
#include "unit_common.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"

using std::clog;
using std::for_each;
using std::max;
using std::shared_ptr;
using std::to_string;
using std::vector;

struct compress_state;
struct state_decompress;

static bool no_gpu = false;
static compress_state *compress{nullptr};
static state_decompress *decompress{nullptr};

static void gpujpeg_test_teardown()
{
        module_done(CAST_MODULE(compress));
        if (decompress != nullptr) {
                decompress_done(decompress);
        }
}

static pthread_once_t set_up = PTHREAD_ONCE_INIT;

static void gpujpeg_test_setup()
{
        struct compress_state *compression;
        int ret = compress_init(nullptr, "GPUJPEG:check", &compression);
        if(ret >= 0) {
                if(ret == 0) {
                        module_done(CAST_MODULE(compression));
                }
        } else {
                clog << "Either GPUJPEG not compiled in or no CUDA-capable devices found - skipping GPUJPEG tests.\n";
                no_gpu = true;
                return;
        }

        ret = compress_init(nullptr, "GPUJPEG", &compress);
        assert(ret >= 0 && "Compression initialization failed");

        commandline_params["decompress"] = "gpujpeg";
        atexit(gpujpeg_test_teardown);
}

int gpujpeg_test_simple()
{
        using namespace std::string_literals;
        pthread_once(&set_up, gpujpeg_test_setup);
        if (no_gpu) {
                return 1;
        }

        struct video_desc desc{1920, 1080, RGB, 1, PROGRESSIVE, 1};
        auto in = shared_ptr<video_frame>(vf_alloc_desc_data(desc), vf_free);
        memset(in->tiles[0].data, 127, in->tiles[0].data_len); /// @todo use some more reasonable stuff

        compress_frame(compress, in);
        auto compressed = compress_pop(compress);
        ASSERT_MESSAGE("Compression failed", compressed);

        vector<unsigned char> decompressed(in->tiles[0].data_len);
        auto comp_desc = desc;
        comp_desc.color_spec = JPEG;
        if (bool ret = decompress_init_multi(JPEG, pixfmt_desc{}, RGB, &decompress, 1)) {
                ASSERT_MESSAGE("Decompression init failed", ret);
        }
        auto ret = decompress_reconfigure(decompress, comp_desc, 0, 8, 16, vc_get_linesize(desc.width, desc.color_spec), desc.color_spec);
        ASSERT_MESSAGE("Decompression reconfiguration failed", ret == TRUE);
        auto status = decompress_frame(decompress,
                decompressed.data(),
                reinterpret_cast<unsigned char *>(compressed->tiles[0].data),
                compressed->tiles[0].data_len,
                0,
                nullptr,
                nullptr);
        ASSERT_MESSAGE("Decompression failed", status == DECODER_GOT_FRAME);

        int i = 0;
        int max_diff = 0;
        for_each(decompressed.begin(), decompressed.end(), [&](unsigned char &x) {max_diff = max(abs(x - (unsigned char) in->tiles[0].data[i++]), max_diff);});
        ASSERT_MESSAGE("Maximal allowed difference 1, found "s + to_string(max_diff), max_diff <= 1);
        return 0;
}
#else
int gpujpeg_test_simple()
{
        return 1;
}
#endif // defined HAVE_GPUJPEG
