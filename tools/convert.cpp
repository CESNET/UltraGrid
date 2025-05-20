#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/config_unix.h"
#include "../src/video_codec.h"
#include "../src/video_frame.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::cout;
using std::cerr;
using std::exception;
using std::ifstream;
using std::ofstream;
using std::stoi;
using std::string;
using std::vector;

static void benchmark() {
        int width = 3840;
        int height = 2160;
        vector<unsigned char> in(width * height * MAX_BPS + MAX_PADDING);
        vector<unsigned char> out(width * height * MAX_BPS + MAX_PADDING);
        for (int i = 0; i < VIDEO_CODEC_END; ++i) {
                bool src_print = false;
                for (int j = 0; j < VIDEO_CODEC_END; ++j) {
                        codec_t inc = static_cast<codec_t>(i);
                        codec_t outc = static_cast<codec_t>(j);
                        decoder_t conv = nullptr;
                        if ((conv = get_decoder_from_to(inc, outc)) == nullptr || i == j) {
                                continue;
                        }
                        auto t0 = high_resolution_clock::now();
                        const size_t src_linesize = vc_get_linesize(width, inc);
                        const size_t dst_linesize = vc_get_linesize(width, outc);
                        for (int i = 0; i < height; ++i) {
                                conv(out.data() + i * src_linesize,
                                     in.data() + i * dst_linesize,
                                     (int) dst_linesize, DEFAULT_R_SHIFT,
                                     DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                        }
                        auto t1 = high_resolution_clock::now();
                        cout << get_codec_name(inc) << "->" << get_codec_name(outc) << ": " << duration_cast<microseconds>(t1 - t0).count() / 1000.0 << " ms\n";
                }
        }
}

static void print_conversions() {
        for (int i = 0; i < VIDEO_CODEC_END; ++i) {
                bool src_print = false;
                for (int j = 0; j < VIDEO_CODEC_END; ++j) {
                        if (get_decoder_from_to(static_cast<codec_t>(i), static_cast<codec_t>(j)) == nullptr || i == j) {
                                continue;
                        }
                        if (!src_print) {
                                cout << get_codec_name(static_cast<codec_t>(i)) << " ->\n";
                                src_print = true;
                        }
                        cout << "\t" << get_codec_name(static_cast<codec_t>(j)) << "\n";
                }
        }
}

static int
write_to_container(int width, int height, const char *in_f,
                   const char *out_fname)
{
        const char *ext = strrchr(in_f, '.');
        if (ext == nullptr) {
                fprintf(stderr, "File with no extension!\n");
                return 1;
        }
        codec_t c = get_codec_from_file_extension(ext + 1);
        if (c == VIDEO_CODEC_NONE) {
                fprintf(stderr, "wrong extension: %s!\n", ext);
                return 1;
        }
        struct video_frame *f = load_video_frame(in_f, c, width, height);
        if (f == nullptr) {
                return 1;
        }
        save_video_frame(f, out_fname, false);
        return 0;
}

int main(int argc, char *argv[]) {
        if (argc == 2 && string("list-conversions") == argv[1]) {
                print_conversions();
                return 0;
        }
        if (argc == 2 && string("benchmark") == argv[1]) {
                benchmark();
                return 0;
        }
        if (argc != 5 && argc != 7) {
                cout << "Tool to convert between UltraGrid raw pixel format with supported conversions.\n\n"
                        "Usage:\n"
                        "\t" << argv[0] << " benchmark | help | list-conversions\n\n"
                        "\t" << argv[0] << " <width> <height> <in_codec> <out_codec> <in_file> <out_file>\n"
                        "\t\t" << "Eg.: " << argv[0] << " 1920 1080 UYVY RGB 00000001.yuv out.rgb\n\n"
                        "\t" << argv[0] << " <width> <height> <in_file> <oname_wo_ext>\n"
                        "\t\t- converts to output Y4M or PNM file (depending on in_file extension color space)\n"
                        "\t\t" << "Eg.: " << argv[0] << " 1920 1080 00000001.yuv out  # creates out.y4m\n\n"
                        "\n"
                        "where\n"
                        "\t" << "benchmark        - benchmark conversions\n"
                        "\t" << "help             - show this help\n"
                        "\t" << "list-conversions - prints valid conversion pairs\n";
                return (argc == 1 || argc == 2 && string("help") == argv[1]) ? 0 : 1;
        }
        int width = stoi(argv[1]);
        int height = stoi(argv[2]);
        if (argc == 5) {
                return write_to_container(width, height, argv[3], argv[4]);
        }
        codec_t in_codec = get_codec_from_name(argv[3]);
        codec_t out_codec = get_codec_from_name(argv[4]);
        ifstream in(argv[5], ifstream::ate | ifstream::binary);
        std::ofstream out(argv[6], ofstream::binary);
        try {
                in.exceptions(ifstream::failbit  | ifstream::badbit | ifstream::eofbit);
                out.exceptions(ofstream::failbit | ofstream::badbit);

                assert (width && height && in_codec && out_codec && in && out);

                size_t in_size = vc_get_datalen(width, height, in_codec);
                assert(in.tellg() >= in_size);
                in.seekg (0, ifstream::beg);

                vector<char> in_data(in_size);
                in.read(in_data.data(), in_size);
                vector<char> out_data(vc_get_datalen(width, height, out_codec));

                auto *decode = get_decoder_from_to(in_codec, out_codec);
                if (decode == nullptr) {
                        cerr << "Cannot find decoder from " << argv[3] << " to " << argv[4] << "! See '" << argv[0] << " list-conversions'\n";
                        return 1;
                }

                size_t dst_linesize = vc_get_linesize(width, out_codec);
                for (int y = 0; y < height; ++y) {
                        decode(reinterpret_cast<unsigned char *>(&out_data[y * dst_linesize]),
                                        reinterpret_cast<unsigned char *>(&in_data[y * vc_get_linesize(width, in_codec)]),
                                        dst_linesize, 0, 8, 16);
                }
                out.write(out_data.data(), out_data.size());
        } catch (exception &e) {
                cerr << "ERROR: " << e.what() << ": " << strerror(errno) << "\n";
        }
}
