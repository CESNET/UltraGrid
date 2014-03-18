#include "capture_filter/resize_utils.h"

using namespace cv;
FILE *F_save=NULL;

int resize_frame(char *indata, char *outdata, unsigned int *data_len, unsigned int width, unsigned int height, double scale_factor){
    int res = 0;
    Mat out, in, rgb;

    if (indata == NULL || outdata == NULL || data_len == NULL) {
        return 1;
    }

    in.create(height, width, CV_8UC2);
    in.data = (uchar*)indata;
    out.data = (uchar*)outdata;

    //printf("\nRESIZING by %f!!!\n",scale_factor );

    cvtColor(in, rgb, CV_YUV2RGB_Y422); //CV_YUV2RGB_UYVY
    resize(rgb, out, Size(0,0), scale_factor, scale_factor, INTER_LINEAR);

    *data_len = out.step * out.rows * sizeof(char);

//    //MODUL DE CAPTURA AUDIO A FITXER PER COMPROVACIONS EN TX
//            //CAPTURA FRAMES ABANS DE DESCODIFICAR PER COMPROVAR RECEPCIÓ.
//            if(F_save==NULL){
//                    printf("recording resized...\n");
//                    F_save=fopen("rgb.raw", "wb");
//            }
//
//            //fwrite(tx_frame->audio_data,tx_frame->audio_data_len,1,F_audio_tx_embed_BM);
//            fwrite(outdata,*data_len,1,F_save);
//    //FI CAPTURA

    return res;
}
