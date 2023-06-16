#include <iostream>
#include <filesystem>
#include <string_view>
#include <memory>
#include <chrono>
#include <cassert>
#include <stdio.h>
#include <jpeglib.h>
#include "ipc_frame_unix.h"

struct file_deleter { void operator()(FILE *f){ fclose(f); } };
using File_uniq = std::unique_ptr<FILE, file_deleter>;

class Img_writer{
public:
        Img_writer(){
                compress_ctx.err = jpeg_std_error(&err_mgr);
                jpeg_create_compress(&compress_ctx);

                compress_ctx.input_components = 3;
                compress_ctx.in_color_space = JCS_RGB;
                jpeg_set_defaults(&compress_ctx);
        }

        Img_writer(const Img_writer&) = delete;
        Img_writer& operator=(const Img_writer&) = delete;

        ~Img_writer(){
                jpeg_destroy_compress(&compress_ctx);
        }

        bool write_img(Ipc_frame *f, const std::string& path){
                assert(f->header.color_spec == IPC_FRAME_COLOR_RGB);
                File_uniq outfile(fopen(path.c_str(), "wb"));
                if (!outfile) {
                        fprintf(stderr, "Can't open output file %s\n", path.c_str());
                        return false;
                }
                jpeg_stdio_dest(&compress_ctx, outfile.get());

                compress_ctx.image_width = f->header.width;
                compress_ctx.image_height = f->header.height;

                jpeg_start_compress(&compress_ctx, true);
                while(compress_ctx.next_scanline < compress_ctx.image_height){
                        JSAMPROW row = (unsigned char *) f->data + compress_ctx.next_scanline * f->header.width * 3;
                        jpeg_write_scanlines(&compress_ctx, &row, 1);
                }
                jpeg_finish_compress(&compress_ctx);

                return true;
        }

private:
        struct jpeg_compress_struct compress_ctx;
        struct jpeg_error_mgr err_mgr;
};

int main(int argc, char **argv){
        if(argc != 3){
                fprintf(stderr, "Usage: %s <socket path> <output path>\n", argv[0]);
                return 1;
        }

        Ipc_frame_reader_uniq reader(ipc_frame_reader_new(argv[1]));
        Ipc_frame_uniq ipc_frame(ipc_frame_new());

        Img_writer img_writer;

        using clock = std::chrono::steady_clock;
        auto frame_time = std::chrono::seconds(1);
        auto next_frame = clock::now();

        while(true){
                printf("Waiting for connection...\n");
                ipc_frame_reader_wait_connect(reader.get());
                printf("Connected...\n");
                while(ipc_frame_reader_read(reader.get(), ipc_frame.get())){
                        auto now = clock::now();
                        if(now < next_frame)
                                continue;

                        std::string path = argv[2];
                        std::string tmp_path = path + ".swp";

                        img_writer.write_img(ipc_frame.get(), tmp_path);
                        std::filesystem::rename(tmp_path, path);

                        next_frame = std::max(next_frame + frame_time, now);
                }
                fprintf(stderr, "Failed to read frame\n");
        }

        return 0;
}
