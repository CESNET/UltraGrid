#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#if defined HAVE_LAVC

#include <list>
#include <string>
#include <tuple>

#include "libavcodec/lavc_common.h"
#include "unit_common.h"
#include "video_codec.h"

using std::get;
using std::list;
using std::make_tuple;
using std::string;
using std::tuple;

extern "C" decoder_t (*testable_get_decoder_from_uv_to_uv)(codec_t in, enum AVPixelFormat av, codec_t *out);

extern "C" bool libavcodec_test_get_decoder_from_uv_to_uv();

bool libavcodec_test_get_decoder_from_uv_to_uv()
{
        using namespace std::string_literals;

        // testing mostly sanity - if the straightforward conversion is selected
        list<tuple<codec_t, codec_t, decoder_t, string, AVPixelFormat>> expected_decoders {
                make_tuple(RG48, RG48, &vc_memcpy, "vc_memcpy"s, AV_PIX_FMT_YUV444P16LE),
                make_tuple(RGB,  RGB,  get_decoder_from_to(RGB, RGB), "vc_copylineRGB"s, AV_PIX_FMT_RGB24),
                make_tuple(BGR,  RGB,  get_decoder_from_to(BGR, RGB), "vc_copylineBGRtoRGB"s, AV_PIX_FMT_RGB24),
                make_tuple(BGR,  BGR,  &vc_memcpy, "vc_memcpy"s, AV_PIX_FMT_BGR24),
                make_tuple(RG48, RG48, &vc_memcpy, "vc_memcpy"s, AV_PIX_FMT_RGB48LE),
        };

        for (auto & test_case : expected_decoders) {
                codec_t out = VIDEO_CODEC_NONE;
                decoder_t dec = testable_get_decoder_from_uv_to_uv(get<0>(test_case), get<4>(test_case), &out);
                ASSERT_EQUAL_MESSAGE("Expected intermediate "s + get_codec_name(get<1>(test_case)) + " for UG decoder for "s
                                + get_codec_name(get<0>(test_case)) + " to "s + av_get_pix_fmt_name(get<4>(test_case)), get<1>(test_case), out);
                ASSERT_EQUAL_MESSAGE("Expected UG decoder "s + get<3>(test_case) + " for "s + get_codec_name(get<0>(test_case)) + " to "s
                                + av_get_pix_fmt_name(get<4>(test_case)), (decoder_t) get<2>(test_case), dec);
        }
        return true;
}

#endif // defined HAVE_LAVC
