#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#if defined HAVE_CPPUNIT && defined HAVE_GPUJPEG

#include <algorithm>
#include <cppunit/config/SourcePrefix.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gpujpeg_test.h"
#include "host.h"
#include "module.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"

using std::clog;
using std::for_each;
using std::max;
using std::shared_ptr;
using std::to_string;
using std::vector;

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( gpujpeg_test );

void
gpujpeg_test::setUp()
{
        struct compress_state *compression;
        int ret = compress_init(nullptr, "GPUJPEG:check", &compression);
        if(ret >= 0) {
                if(ret == 0) {
                        module_done(CAST_MODULE(compression));
                }
        } else {
                clog << "Either GPUJPEG not compiled in or no CUDA-capable devices found - skipping GPUJPEG tests.\n";
                m_skip = true;
                return;
        }

        ret = compress_init(nullptr, "GPUJPEG", &m_compress);
        CPPUNIT_ASSERT_MESSAGE("Compression initialization failed", ret >= 0);

        commandline_params["decompress"] = "gpujpeg";
}

void
gpujpeg_test::tearDown()
{
        module_done(CAST_MODULE(m_compress));
        if (m_decompress != nullptr) {
                decompress_done(m_decompress);
        }
}

void
gpujpeg_test::test_simple()
{
        using namespace std::string_literals;

        if (m_skip) {
                return;
        }

        struct video_desc desc{1920, 1080, RGB, 1, PROGRESSIVE, 1};
        auto in = shared_ptr<video_frame>(vf_alloc_desc_data(desc), vf_free);
        memset(in->tiles[0].data, 127, in->tiles[0].data_len); /// @todo use some more reasonable stuff

        compress_frame(m_compress, in);
        auto compressed = compress_pop(m_compress);
        CPPUNIT_ASSERT_MESSAGE("Compression failed", compressed);

        vector<unsigned char> decompressed(in->tiles[0].data_len);
        auto comp_desc = desc;
        comp_desc.color_spec = JPEG;
        if (bool ret = decompress_init_multi(JPEG, pixfmt_desc{}, RGB, &m_decompress, 1)) {
                CPPUNIT_ASSERT_MESSAGE("Decompression init failed", ret);
        }
        auto ret = decompress_reconfigure(m_decompress, comp_desc, 0, 8, 16, vc_get_linesize(desc.width, desc.color_spec), desc.color_spec);
        CPPUNIT_ASSERT_MESSAGE("Decompression reconfiguration failed", ret == TRUE);
        auto status = decompress_frame(m_decompress,
                decompressed.data(),
                reinterpret_cast<unsigned char *>(compressed->tiles[0].data),
                compressed->tiles[0].data_len,
                0,
                nullptr,
                nullptr);
        CPPUNIT_ASSERT_MESSAGE("Decompression failed", status == DECODER_GOT_FRAME);

        int i = 0;
        int max_diff = 0;
        for_each(decompressed.begin(), decompressed.end(), [&](unsigned char &x) {max_diff = max(abs(x - (unsigned char) in->tiles[0].data[i++]), max_diff);});
        CPPUNIT_ASSERT_MESSAGE("Maximal allowed difference 1, found "s + to_string(max_diff), max_diff <= 1);
}

#endif // defined HAVE_CPPUNIT && defined HAVE_GPUJPEG
