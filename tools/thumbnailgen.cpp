#include <iostream>
#include <filesystem>
#include <string_view>
#include <chrono>
#include <cassert>
#include <stdio.h>
#include <jpeglib.h>
#include "ipc_frame_unix.h"

bool write_img(Ipc_frame *f, std::string path){
        struct jpeg_compress_struct compress_ctx;
        struct jpeg_error_mgr err_mgr;

        compress_ctx.err = jpeg_std_error(&err_mgr);
        jpeg_create_compress(&compress_ctx);


        FILE *outfile = fopen(path.c_str(), "wb");
        if (!outfile) {
                fprintf(stderr, "Can't open output file %s\n", path.c_str());
                return false;
        }
        jpeg_stdio_dest(&compress_ctx, outfile);

        compress_ctx.image_width = f->header.width;
        compress_ctx.image_height = f->header.height;
        compress_ctx.input_components = 3;
        compress_ctx.in_color_space = JCS_RGB;
        jpeg_set_defaults(&compress_ctx);

        jpeg_start_compress(&compress_ctx, true);
        for(int y = 0; y < f->header.height; y++){
                JSAMPROW row = (unsigned char *) f->data + y * f->header.width * 3;
                jpeg_write_scanlines(&compress_ctx, &row, 1);
        }

        jpeg_finish_compress(&compress_ctx);
        jpeg_destroy_compress(&compress_ctx);

        fclose(outfile);

        return true;
}

int main(int argc, char **argv){
        if(argc != 3){
                fprintf(stderr, "Usage: %s <socket path> <output path>\n", argv[0]);
                return 1;
        }

        Ipc_frame_reader_uniq reader(ipc_frame_reader_new(argv[1]));
        Ipc_frame_uniq ipc_frame(ipc_frame_new());

        using clock = std::chrono::steady_clock;

        auto next_frame = clock::now();

        while(true){
                printf("Waiting for connection...\n");
                while(!ipc_frame_reader_is_connected(reader.get()))
                {
                        //TODO don't busywait
                }
                while(ipc_frame_reader_read(reader.get(), ipc_frame.get())){
                        assert(ipc_frame->header.color_spec == IPC_FRAME_COLOR_RGB);
                        auto now = clock::now();
                        if(now < next_frame)
                                continue;

                        std::string path = argv[2];
                        std::string tmp_path = path + ".swp";
                        write_img(ipc_frame.get(), path + ".swp");

                        std::filesystem::rename(tmp_path, path);
                        next_frame += std::chrono::seconds(1);
                        if(next_frame < now)
                                next_frame = now;
                }
                fprintf(stderr, "Failed to read frame\n");
        }

        return 0;
}
