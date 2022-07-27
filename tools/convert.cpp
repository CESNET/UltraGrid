#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/config_unix.h"
#include "../src/video_codec.h"

using std::cout;
using std::cerr;
using std::ifstream;
using std::ofstream;
using std::stoi;
using std::string;

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

int main(int argc, char *argv[]) {
        if (argc == 2 && string("list-conversions") == argv[1]) {
                print_conversions();
                return 0;
        }
        if (argc < 7) {
                cout << "Usage:\n"
                                "\t" << argv[0] << " <width> <height> <in_codec> <out_codec> <in_file> <out_file> | list-conversions\n"
                                "\n"
                                "where\n"
                                "\t" << "list-conversions - prints valid conversion pairs\n";
                return 1;
        }
        int width = stoi(argv[1]);
        int height = stoi(argv[2]);
        codec_t in_codec = get_codec_from_name(argv[3]);
        codec_t out_codec = get_codec_from_name(argv[4]);
        ifstream in(argv[5], ifstream::ate | ifstream::binary);
        std::ofstream out(argv[6], ofstream::binary);
        in.exceptions(ifstream::failbit  | ifstream::badbit | ifstream::eofbit);
        out.exceptions(ofstream::failbit | ofstream::badbit);

        assert (width && height && in_codec && out_codec && in && out);

        size_t in_size = vc_get_datalen(width, height, in_codec);
        assert(in.tellg() >= in_size);
        in.seekg (0, ifstream::beg);

        std::vector<char> in_data(in_size);
        in.read(in_data.data(), in_size);
        std::vector<char> out_data(vc_get_datalen(width, height, out_codec));

        auto *decode = get_decoder_from_to(in_codec, out_codec);
        if (decode == nullptr) {
                cerr << "Cannot find decoder from " << argv[3] << " to " << argv[4] << "! See '" << argv[0] << " conversions'\n";
                return 1;
        }

        size_t dst_linesize = vc_get_linesize(width, out_codec);
        for (int y = 0; y < height; ++y) {
                decode(reinterpret_cast<unsigned char *>(&out_data[y * dst_linesize]),
                                reinterpret_cast<unsigned char *>(&in_data[y * vc_get_linesize(width, in_codec)]),
                                dst_linesize, 0, 8, 16);
        }
        out.write(out_data.data(), out_data.size());
}
