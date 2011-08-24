/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv/svram program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/

#include "svprogram.h"


#define SEQUENCE_USAGE "\
*****************************************************************************\n\
* sv sequence [options] start [frames [object [angle [startangle [size [radius\n\
*       [mode [fld [Y|R [Cr|G [Cb|B [mx [my]]]]]]]]]]]] [options]\n\
*\n\
*****************************************************************************\n\
* parameter: [options]:\n\
* start                     start frame\n\
* frames                    length of sequence\n\
*                           [frames needed for a full cycle (depending\n\
*                            on 'angle')] \n\
*                           [number of frames per second if 'angle' is not\n\
*                            defined]\n\
* object     -o object      object {fcircle|circle|fsquare|square|fdiamond|\n\
*                            diamond}\n\
*                           [fcircle]\n\
* angle      -a angle       angle (in degrees) between two frames\n\
*                           [angle needed for a full cycle in 'frames' frames]\n\
* startangle -t startangle  angle (in degrees) of first frame\n\
*                           [0]\n\
* size       -s size        object size\n\
*                           [1/6 screen height]\n\
* radius     -r radius      distance from center\n\
*                           [1/2 screen height - size]\n\
* mode fld   -f mode [fld]  animation mode {frame|field}\n\
*                           [field]\n\
*                           field dominance {even|odd}\n\
*                           even: 0 2 4 6 8 ...\n\
*                           odd:  1 3 5 7 9 ...\n\
*                           [field dominance according to video raster]\n\
* Y Cr Cb    -c Y [Cr [Cb]] object color (when YUV-mode)\n\
* R G B      -c R [G [B]]   object color (when RGB-mode)\n\
*                           [color white]\n\
* mx my      -m mx my       center\n\
*                           [1/2 screen height/width]\n\
*****************************************************************************\n"


#define SEQUENCE_NONE	0
#define SEQUENCE_FCIRCLE 1	/*object type for filled circle*/
#define SEQUENCE_CIRCLE  2	/*object type for circle*/
#define SEQUENCE_FSQUARE 3	/*object type for filled square*/
#define SEQUENCE_SQUARE  4	/*object type for square*/
#define SEQUENCE_FDIAMOND 5	/*object type for filled diamond*/
#define SEQUENCE_DIAMOND 6	/*object type for diamond*/

#define SEQUENCE_ANIM_FRAME 1	/*one frame is one animation-step*/ 
#define SEQUENCE_ANIM_FIELD 2	/*one field is one animation-step*/

#define SEQUENCE_FIRST_ODD  1	/*odd line numbers field first*/
#define SEQUENCE_FIRST_EVEN 2	/*even line numbers field first*/

#if !defined(M_PI)
# define M_PI 3.141592654
#endif


typedef struct {
  int type;			/*object type*/
  int size;			/*object size*/
  int dist;			/*object distance*/
  struct {
    int Y, Cr, Cb;
  } color;			/*object color*/
  struct {
    int x, y;
  } center;			/*center of animation*/
  double angle;		/*angle of between two animation steps*/
  double startangle;	/*angle of first frame*/
  int anim;			/*frame or field animation*/
  int field;		/*field dominance*/
  int field_native;		/*raster native field dominance*/

  char *buffer;
  int buffer_size;
  int pixel_values;
  int pixel_size;
  int sv_mode;
} sequence_handle;

/*** sequence_getpos() - find the buffer position according to x and y */
static void sequence_getpos(sequence_handle * s, sv_info * info, int * pos, int x, int y)
{
  if (x>=info->xsize || x<0 || y>=info->ysize || y<0) {
    *pos=-1;
  } else {
    if (s->field==SEQUENCE_FIRST_EVEN) {
      if (y%2 != 0) {
        *pos = info->xsize * info->ysize / 2;
        *pos += (y-1)/2 * info->xsize;
      }
      else if (y%2 == 0) {
        *pos = 0;
        *pos += y / 2 * info->xsize;
      }
      *pos += x;
      *pos *= s->pixel_values;
    }
    else {
      if (y%2 != 0) {
        *pos = 0;
        *pos += (y-1) / 2 * info->xsize;
      }
      else if (y%2 == 0) {
        *pos = info->xsize * info->ysize / 2;
        *pos += y / 2 * info->xsize;
      }
      *pos += x;
      *pos *= s->pixel_values;
    }
  }
}
/* end of sequence_getpos() */


/**********************************************************************
 * put_pixel_xxx functions set a pixel in the specified format
 *
 * yuv422:			Cb0  Y0 Cr0  Y1 Cb2  Y2 Cr2  Y3 ...
 * yuv422a (8bit):	Cb0  Y0  A0 Cr0  Y1  A1 Cb2  Y2  A2 Cr2  Y3  A3 ...
 * yuv422a (10bit):	 Y0 Cb0  A0  Y1 Cr0  A1  Y2 Cb2  A2  Y3 Cr2  A3 ...
 * rgb:			 	 G0  B0  R0  G1  B1  R1 ...
 * rgba:			 G0  B0  R0  A0  G1  B1  R1  A1 ...
 **********************************************************************/

static void put_pixel_yuv422(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    if (x%2 == 0) {		/*every second one*/
      *((char *)buffer + pos)     = (char)(s->color).Cb;
      *((char *)buffer + pos + 2) = (char)(s->color).Cr;
    }
    *((char *)buffer + pos + 1) = (char)(s->color).Y;
  }
}

static void put_pixel_yuv422_10b(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    if (x%2 == 0) {		/*every second one*/
      *((short *)buffer + pos)     = (short)(s->color).Cb<<6;
      *((short *)buffer + pos + 2) = (short)(s->color).Cr<<6;
    }
    *((short *)buffer + pos + 1) = (short)(s->color).Y<<6;
  }
}

static void put_pixel_rgb(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    *((char *)buffer + pos)     = (char)(s->color).Cr;		/*means (g)reen*/
    *((char *)buffer + pos + 1) = (char)(s->color).Cb;		/*means (b)lue*/
    *((char *)buffer + pos + 2) = (char)(s->color).Y;		/*means (r)ed*/
  }
}

static void put_pixel_rgb_10b(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    *((short *)buffer + pos)     = (short)(s->color).Cr<<6;	/*means (g)reen*/
    *((short *)buffer + pos + 1) = (short)(s->color).Cb<<6;	/*means (b)lue*/
    *((short *)buffer + pos + 2) = (short)(s->color).Y<<6;	/*means (r)ed*/
  }
}

static void put_pixel_yuv422a(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    if (x%2 == 0) {	/*every second one*/
      *((char *)buffer + pos)     = (char)(s->color).Cb;
      *((char *)buffer + pos + 3) = (char)(s->color).Cr;
    }
    *((char *)buffer + pos + 1) = (char)(s->color).Y;
    /*leave out alpha value*/
  }
}

static void put_pixel_yuv422a_10b(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    if ((x)%2 == 0) {	/*every second one*/
      *((short *)buffer + pos + 1) = (short)(s->color).Cb<<6;
      *((short *)buffer + pos + 4) = (short)(s->color).Cr<<6;
    }
    *((short *)buffer + pos + 0) = (short)(s->color).Y<<6;
    /*leave out alpha value*/
  }
}

static void put_pixel_rgba(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    *((char *)buffer + pos)     = (char)(s->color).Cr;	/*means (g)reen*/
    *((char *)buffer + pos + 1) = (char)(s->color).Cb;	/*means (b)lue*/
    *((char *)buffer + pos + 2) = (char)(s->color).Y;	/*means (r)ed*/
    /*leave out alpha value*/
  }
}

static void put_pixel_rgba_10b(char * buffer, int pos, int x, sequence_handle * s)
{
  if (pos>=0) {
    *((short *)buffer + pos)     = (short)(s->color).Cr<<6;	/*means (g)reen*/
    *((short *)buffer + pos + 1) = (short)(s->color).Cb<<6;	/*means (b)lue*/
    *((short *)buffer + pos + 2) = (short)(s->color).Y<<6;	/*means (r)ed*/
    /*leave out alpha value*/
  }
}

static void sequence_loop(sv_handle * sv, sv_info * info, int start, int limit, sequence_handle *s, void (*put_pixel)())
{
  int res;
  double alpha;			/*actual angle*/
  double half_angle;	/*for angle/2 (used when ANIM_FIELD)*/
  int frame;			/*frame counter*/
  int field;			/*field counter*/
  int pos;				/*position in frame buffer*/
  int mx, my;			/*object coordinates (center)*/
  int i, j;

  /*** frame animation ************************************************/

  if (s->anim==SEQUENCE_ANIM_FRAME)
    for (alpha=s->startangle, frame=start; frame<start+limit;
      alpha+=s->angle, frame++)
    {
      res=sv_sv2host(sv, s->buffer, s->buffer_size, info->xsize, info->ysize,
        frame, 1, s->sv_mode);		/*get frame from device*/

      if (res!=SV_OK) sv_errorprint(sv, res);

      mx = s->center.x + (int)((double)s->dist * cos(alpha));	/*get object position*/
      my = s->center.y + (int)((double)s->dist * sin(alpha));
      printf("frame %3d  (angle: %7.2f)\n", frame, alpha/M_PI*180);

      /*** draw a filled circle ***************************************/
      if (s->type==SEQUENCE_FCIRCLE) {
        for (j=my-s->size; j<my+s->size; j++)
          for (i=mx-s->size; i<mx+s->size; i++)
            if (((j-my)*(j-my) + (i-mx)*(i-mx)) <= s->size*s->size) {
              sequence_getpos(s, info, &pos, i, j);
              (*put_pixel)(s->buffer, pos, i, s);
            }
      }

      /*** draw a circle **********************************************/
      else if (s->type==SEQUENCE_CIRCLE) {
        for (i=mx-s->size; i<mx+s->size; i++) {
          j = (int)sqrt((double) ((s->size*s->size) - (i-mx)*(i-mx)));
          sequence_getpos(s, info, &pos, i, j);
          (*put_pixel)(s->buffer, pos, i, s);
          sequence_getpos(s, info, &pos, i, j);
          (*put_pixel)(s->buffer, pos, mx+j, s);
          sequence_getpos(s, info, &pos, i, j);
          (*put_pixel)(s->buffer, pos, i, s);
          sequence_getpos(s, info, &pos, i, j);
          (*put_pixel)(s->buffer, pos, mx-j, s);
        }
      }

      /*** draw a filled square ***************************************/
      else if (s->type==SEQUENCE_FSQUARE) {
        for(j=my-s->size; j<my+s->size; j++)
          for(i=mx-s->size; i<mx+s->size; i++) {
            sequence_getpos(s, info, &pos, i, j);
            (*put_pixel)(s->buffer, pos, i, s);
          }
      }

      /*** draw a square **********************************************/
      else if (s->type==SEQUENCE_SQUARE) {
        for(j=my-s->size; j<my+s->size; j++) {
          sequence_getpos(s, info, &pos, mx-s->size, j);
          (*put_pixel)(s->buffer, pos, mx-s->size, s);
          sequence_getpos(s, info, &pos, mx+s->size, j);
          (*put_pixel)(s->buffer, pos, mx+s->size, s);
          sequence_getpos(s, info, &pos, j-my+mx, my-s->size);
          (*put_pixel)(s->buffer, pos, j-my+mx, s);
          sequence_getpos(s, info, &pos, j-my+mx, my+s->size);
          (*put_pixel)(s->buffer, pos, j-my+mx, s);
        }
      }

      /*** draw a filled diamond **************************************/
      else if (s->type==SEQUENCE_FDIAMOND) {
        for (i=mx, j=my-s->size; j<=my; i++, j++) {
          int t;
          for (t=i; t>=mx-(i-mx); t--) {
            sequence_getpos(s, info, &pos, t, j);
            (*put_pixel)(s->buffer, pos, t, s);
          }
        }
        for (i-=2; i>=mx; i--, j++) {
          int t;
          for (t=i; t>=mx-(i-mx); t--) {
            sequence_getpos(s, info, &pos, t, j);
            (*put_pixel)(s->buffer, pos, t, s);
          }
        }
      }

      /*** draw a diamond *********************************************/
      else if (s->type==SEQUENCE_DIAMOND) {
        for (i=mx, j=my-s->size; j<=my; i++, j++) {
          sequence_getpos(s, info, &pos, i, j);
          (*put_pixel)(s->buffer, pos, i, s);
          sequence_getpos(s, info, &pos, mx-(i-mx), j);
          (*put_pixel)(s->buffer, pos, mx-(i-mx), s);
        }
        for (i-=2; i>=mx; i--, j++) {
          sequence_getpos(s, info, &pos, i, j);
          (*put_pixel)(s->buffer, pos, i, s);
          sequence_getpos(s, info, &pos, mx-(i-mx), j);
          (*put_pixel)(s->buffer, pos, mx-(i-mx), s);
        }
      }

      res=sv_host2sv(sv, s->buffer, s->buffer_size, info->xsize, info->ysize,
        frame, 1, s->sv_mode);		/*write frame to device*/
      if (res!=SV_OK) sv_errorprint(sv, res);
    }

  /*** field animation ************************************************/

  else {
    half_angle = s->angle / 2.0;	/*angle between two fields*/
    for (alpha=s->startangle, frame=start; frame<start+limit; frame++)
    {
      res=sv_sv2host(sv, s->buffer, s->buffer_size, info->xsize, info->ysize,
        frame, 1, s->sv_mode);		/*get frame from device*/

      if (res!=SV_OK) sv_errorprint(sv, res);

      for (field=1; field<=2; field++, alpha+=half_angle) {
        if (s->field!=s->field_native && field==1)
          alpha+=half_angle;
        else if
          (s->field!=s->field_native && field==2)
          alpha-=s->angle;
        mx = s->center.x + (int)((double)s->dist * cos(alpha));
        my = s->center.y + (int)((double)s->dist * sin(alpha));
        j=my-s->size;
        if (s->field==SEQUENCE_FIRST_EVEN) {
          if (((field==1) && (j%2!=0)) || ((field==2) && (j%2==0))) {
            j++;
          }
        }
        else if (s->field==SEQUENCE_FIRST_ODD)
          if (((field==1) && (j%2==0)) || ((field==2) && (j%2!=0))) {
            j++;
          }

        printf("frame %3d field%d  (angle: %7.2f)\n", frame, field,
          alpha/M_PI*180);

        /*** draw filled circle *****************************************/
        if (s->type==SEQUENCE_FCIRCLE) {
          for ( ; j<my+s->size; j+=2)
            for (i=mx-s->size; i<mx+s->size; i++)
              if (((j-my)*(j-my) + (i-mx)*(i-mx)) <= s->size*s->size) {
                sequence_getpos(s, info, &pos, i, j);
                (*put_pixel)(s->buffer, pos, i, s);
              }
        }

        /*** draw circle ***********************************************/
        else if (s->type==SEQUENCE_CIRCLE) {
          for (; j<my+s->size; j+=2) {
            i = (int)sqrt((double) (((s->size)*(s->size)) - ((j-my)*(j-my))));
            sequence_getpos(s, info, &pos, mx+i, j);
            (*put_pixel)(s->buffer, pos, mx+i, s);
            sequence_getpos(s, info, &pos, mx-i, j);
            (*put_pixel)(s->buffer, pos, mx-i, s);

            for (i=mx-s->size; i<mx+s->size; i++)
              if (j-my == (int) sqrt((s->size)*(s->size) - (i-mx)*(i-mx))) {
                sequence_getpos(s, info, &pos, i, j);
                (*put_pixel)(s->buffer, pos, i, s);
                sequence_getpos(s, info, &pos, i, my-j+my);
                (*put_pixel)(s->buffer, pos, i, s);
              }
          }
        }

        /*** draw a filled square ***************************************/
        else if (s->type==SEQUENCE_FSQUARE) {
          for( ; j<my+s->size; j+=2)
            for(i=mx-s->size; i<mx+s->size; i++) {
              sequence_getpos(s, info, &pos, i, j);
              (*put_pixel)(s->buffer, pos, i, s);
            }
        }

        /*** draw a square **********************************************/
        else if (s->type==SEQUENCE_SQUARE) {
          if (field==1)
            for (i=mx-s->size; i<mx+s->size; i++) {
              sequence_getpos(s, info, &pos, i, j);
              (*put_pixel)(s->buffer, pos, i, s);
            }
          for( ; j<my+s->size; j+=2) {
              sequence_getpos(s, info, &pos, mx-s->size, j);
              (*put_pixel)(s->buffer, pos, mx-s->size, s);
              sequence_getpos(s, info, &pos, mx+s->size-1, j);
              (*put_pixel)(s->buffer, pos, mx+s->size, s);
          }
          if (field==2)
            for (i=mx-s->size; i<mx+s->size; i++) {
              sequence_getpos(s, info, &pos, i, j-2);
              (*put_pixel)(s->buffer, pos, i, s);
            }
        }

        /*** draw a filled diamond ***************************************/
        else if (s->type==SEQUENCE_FDIAMOND) {
          if (field==1)
            i=mx;
          else
            i=mx+1;

          for (; i<mx+s->size; i+=2, j+=2) {
            int t;
            for (t=i; t>=mx-(i-mx); t--) {
              sequence_getpos(s, info, &pos, t, j);
              (*put_pixel)(s->buffer, pos, t, s);
            }
          }
          for (i-=4; i>=mx; i-=2, j+=2) {
            int t;
            for (t=i; t>=mx-(i-mx); t--) {
              sequence_getpos(s, info, &pos, t, j);
              (*put_pixel)(s->buffer, pos, t, s);
            }
          }
        }

        /*** draw a diamond *********************************************/
        else if (s->type==SEQUENCE_DIAMOND) {
          if (field==1)
            i=mx;
          else
            i=mx+1;

          for (; i<mx+s->size/*j<=my*/; i+=2, j+=2) {
            sequence_getpos(s, info, &pos, i, j);
            (*put_pixel)(s->buffer, pos, i, s);
            sequence_getpos(s, info, &pos, mx-(i-mx), j);
            (*put_pixel)(s->buffer, pos, mx-(i-mx), s);
          }
          for (i-=4; i>=mx/*j<my+s->size*/; i-=2, j+=2) {
            sequence_getpos(s, info, &pos, i, j);
            (*put_pixel)(s->buffer, pos, i, s);
            sequence_getpos(s, info, &pos, mx-(i-mx), j);
            (*put_pixel)(s->buffer, pos, mx-(i-mx), s);
          }
        }
      }

      if (s->field!=s->field_native)
        alpha+=half_angle;

      res=sv_host2sv(sv, s->buffer, s->buffer_size, info->xsize, info->ysize,
        frame, 1, s->sv_mode);		/*write frame to device*/
      if (res!=SV_OK) sv_errorprint(sv, res);
    }
  }
}


/**********************************************************************/
static int get_mode(sv_info * info)
{
  int mode;

  /**** get actual data type ****/
  if (info->colormode == SV_COLORMODE_RGB_BGR)
    mode = SV_TYPE_RGB_BGR;
  else if (info->colormode == SV_COLORMODE_YUV422)
    mode = SV_TYPE_YUV422;
  else if (info->colormode == SV_COLORMODE_RGBA)
    mode = SV_TYPE_RGBA_RGBA;
  else if (info->colormode == SV_COLORMODE_YUV422A)
    mode = SV_TYPE_YUV422A;
  else
    return -1;

  /**** get actual data size ****/
  if (info->nbit <= 8)
    mode = mode | SV_DATASIZE_8BIT;
  else if (info->nbit <= 16)
#ifdef WORDS_BIGENDIAN
    mode = mode | SV_DATASIZE_16BIT_BIG;
#else
    mode = mode | SV_DATASIZE_16BIT_LITTLE;
#endif
  else
    return -1;

  return mode;
}
/*** end get_mode() ***************************************************/


/**********************************************************************/
static int get_buffer_size(sv_info * info, sequence_handle * s)
{
  /**** get number of values that describe one pixel ****/
  if (info->colormode==SV_COLORMODE_YUV422)
    s->pixel_values = 2;
  else if (info->colormode==SV_COLORMODE_RGB_BGR)
    s->pixel_values = 3;
  else if (info->colormode==SV_COLORMODE_YUV422A)
    s->pixel_values = 3;
  else if (info->colormode==SV_COLORMODE_RGBA)
    s->pixel_values = 4;
  else
    s->pixel_values = 0;

  /**** get size of one pixel in byte ****/
  if (info->nbit<=8)
    s->pixel_size = s->pixel_values * sizeof(char);
  else if (info->nbit<=16)
    s->pixel_size = s->pixel_values * sizeof(short);
  else
    s->pixel_size = 0;

  return (s->pixel_size * info->xsize * info->ysize);
}
/*** end get_buffer_size() ********************************************/


/**********************************************************************/
int jpeg_sequence(sv_handle * sv, int argc, char ** argv)
{
  sv_info info;
  int res;
  sequence_handle s;		/*structure for data that specify the animation*/
  int start;				/*start frame for animation*/
  int limit;				/*frame limit*/
  double frames_per_sec;
  int count = 0;
  void (*put_pixel_func)(char *buffer, int pos, int x, sequence_handle *s) = 0;

  /*** check if user needs help ***************************************/
  if ((argc<2) || (argc>1 && strcmp(*(argv+1), "help")==0)) {
    printf(SEQUENCE_USAGE);
    return -1;
  }

  memset(&s, 0, sizeof(s));

  /*** make initialization ********************************************/
  res=sv_status(sv, &info);		/*get status from device*/
  if (res!=SV_OK) {
    sv_errorprint(sv, res);
    return -1;
  }

  s.sv_mode = get_mode(&info);	/*set sv_mode for further use by
                                  sv_host2sv and sv_sv2host*/
  if (s.sv_mode==-1) {
    fprintf(stderr, "actual colormode 0x%x not implemented\n", info.colormode);
    return -1;
  }
  s.buffer_size = get_buffer_size(&info, &s);	/*set buffer_size*/
  s.buffer = (char *) malloc(s.buffer_size);	/*allocate buffer for data
                                                 transfer*/
  if (s.buffer==NULL) {
    fprintf(stderr, "not enough memory\n");
    return -1;
  }

  /*** set default values *********************************************/
  start = -1;				/*frame number not set yet*/
  limit = -1;				/*limit value not set yet*/
  s.angle = 0.0;			/*angle value not set yet*/
  s.startangle = 0.0;

  s.type = SEQUENCE_FCIRCLE;			/*default object type*/
  s.size = info.ysize / 12;				/*default object size*/
  s.dist = (info.ysize>>1) - s.size;	/*default distance*/
  if (info.colormode==SV_COLORMODE_YUV422 ||
      info.colormode==SV_COLORMODE_YUV422A)
  {
    s.color.Y = (1 << info.nbit) - 1;	/*default color is white*/
    s.color.Cr = (1 << info.nbit) >> 1;
    s.color.Cb = (1 << info.nbit) >> 1;
  }
  else if (info.colormode==SV_COLORMODE_RGB_BGR ||
           info.colormode==SV_COLORMODE_RGBA)
  {
    s.color.Cr = (1 << info.nbit) - 1;	/*means red*/
    s.color.Cb = (1 << info.nbit) - 1;	/*means green*/
    s.color.Y = (1 << info.nbit) - 1;	/*means blue*/
  }
  s.center.x = info.xsize >> 1;	/*default animation center is screen center*/
  s.center.y = info.ysize >> 1;
  s.anim = SEQUENCE_ANIM_FIELD;		/*default is field animation*/
  switch(info.ysize) {
    case 576:				/*seems to be PAL mode*/
    case 1080:
    case 1152:
      s.field_native = s.field = SEQUENCE_FIRST_EVEN;
      frames_per_sec = 25.0;
      break;
    case 486:				/*seems to be NTSC mode*/
    case 1035:
    case 1036:
      s.field_native = s.field = SEQUENCE_FIRST_ODD;
      frames_per_sec = 29.97;
      break;
    default:
      fprintf(stderr, "video mode not recognized - setting field dominance to even\n");
      s.field_native = s.field = SEQUENCE_FIRST_EVEN;
      frames_per_sec = 25.0;
  }

  /*** parameter evaluation *******************************************/
  while (--argc>0) {

    /*** scan options field ***********************************************/
    if (**++argv=='-') {		/*check for parameters beginning with a slash*/
      switch(*(*argv+1)) {		/*check for the second parameter character*/
        case 'a':
          if (--argc>0)			/*get step angle*/
            s.angle = atof(*++argv) * M_PI / 180.0;	/*convert to rad*/
          if (s.angle==0.0) {
            fprintf(stderr, "angle cannot be 0\n");
            free(s.buffer);
            return -1;
          }
          break;
        case 'c':				/*get object color*/
          if (--argc>0) {
            s.color.Y = atoi(*++argv);
            if (argc-1>0)
              if (**(argv+1)>='0' && **(argv+1)<='9') {
                --argc;
                s.color.Cr = atoi(*++argv);
                if (argc-1>0)
                  if (**(argv+1)>='0' && **(argv+1)<='9') {
                    --argc;
                    s.color.Cb = atoi(*++argv);
                  }
              }
          }
          break;
        case 'f':			/*get animation mode and field dominance*/
          if (--argc>0) {
            if (strcmp(*++argv, "frame")==0)
              s.anim = SEQUENCE_ANIM_FRAME;
            else if (strcmp(*argv, "field")==0)
              s.anim = SEQUENCE_ANIM_FIELD;
            else {
              fprintf(stderr, "enter 'frame|field' for animation mode\n");
              free(s.buffer);
              return -1;
            }
            if (argc-1>0)
              if (strcmp(*(argv+1), "odd")==0 || strcmp(*(argv+1), "even")==0) {
                --argc;
                if (strcmp(*++argv, "odd")==0)
                  s.field = SEQUENCE_FIRST_ODD;
                else if (strcmp(*argv, "even")==0)
                  s.field = SEQUENCE_FIRST_EVEN;
                else {
                  fprintf(stderr, "enter 'even|odd' for field dominance\n");
                  free(s.buffer);
                  return -1;
                }
              }
          }
          break;
        case 'm':			/*get animation center*/
          if (--argc>0)
            s.center.x = atoi(*++argv);
          if (--argc>0)
            s.center.y = atoi(*++argv);
          break;
        case 'o':			/*get object name*/
          if (--argc>0) {
            if (strcmp(*++argv, "fcircle")==0) {
              s.type = SEQUENCE_FCIRCLE;
            } else if (strcmp(*argv, "circle")==0) {
              s.type = SEQUENCE_CIRCLE;
            } else if (strcmp(*argv, "fsquare")==0) {
              s.type = SEQUENCE_FSQUARE;
            } else if (strcmp(*argv, "square")==0) {
              s.type = SEQUENCE_SQUARE;
            } else if (strcmp(*argv, "fdiamond")==0) {
              s.type = SEQUENCE_FDIAMOND;
            } else if (strcmp(*argv, "diamond")==0) {
              s.type = SEQUENCE_DIAMOND;
            } else {
              fprintf(stderr, "unknown object name\n");
              free(s.buffer);
              return -1;
            }
          }
          break;
        case 'r':			/*get radius*/
          if (--argc>0)
            s.dist = atoi(*++argv);
          break;
        case 's':			/*get object size*/
          if (--argc>0)
            s.size = atoi(*++argv)>>1;	/*program needs half the size*/
          break;
        case 't':
          if (--argc>0)			/*get start angle*/
            s.startangle = atof(*++argv) * M_PI / 180.0;
          break;
        default:
          fprintf(stderr, "unknown option parameter -%c\n", *(*argv+1));
          free(s.buffer);
          return -1;
      }
    }

    /*** scan parameters ******************************************************/
    else {
      if (count==0)			/*get start frame*/
        start = atoi(*argv);
      else if (count==1)		/*get frame limit*/
        limit = atoi(*argv);
      else if (count==2) {		/*get object type*/
         if (strcmp(*argv, "fcircle")==0)
           s.type = SEQUENCE_FCIRCLE;
         else if (strcmp(*argv, "circle")==0)
           s.type = SEQUENCE_CIRCLE;
         else if (strcmp(*argv, "fsquare")==0)
           s.type = SEQUENCE_FSQUARE;
         else if (strcmp(*argv, "square")==0)
           s.type = SEQUENCE_SQUARE;
         else if (strcmp(*argv, "fdiamond")==0)
           s.type = SEQUENCE_FDIAMOND;
         else if (strcmp(*argv, "diamond")==0)
           s.type = SEQUENCE_DIAMOND;
         else {
           fprintf(stderr, "unknown object name\n");
           free(s.buffer);
           return -1;
         }
      }
      else if (count==3) {		/*get angle*/
        s.angle = atof(*argv) * M_PI / 180.0;
        if (s.angle==0.0) {
          fprintf(stderr, "angle cannot be 0\n");
          free(s.buffer);
          return -1;
        }
      }
      else if (count==4)		/*get start angle*/
        s.startangle = atof(*argv) * M_PI / 180.0;
      else if (count==5)		/*get size*/
        s.size = atoi(*argv)>>1;
      else if (count==6)		/*get radius*/
        s.dist = atoi(*argv);
      else if (count==7) {		/*get animation mode*/
        if (strcmp(*argv, "frame")==0)
          s.anim = SEQUENCE_ANIM_FRAME;
        else if (strcmp(*argv, "field")==0)
          s.anim = SEQUENCE_ANIM_FIELD;
        else {
          fprintf(stderr, "enter 'frame|field' for animation mode\n");
          free(s.buffer);
          return -1;
        }
      }
      else if (count==8) {		/*get field dominance*/
        if (strcmp(*argv, "odd")==0)
          s.field = SEQUENCE_FIRST_ODD;
        else if (strcmp(*argv, "even")==0)
          s.field = SEQUENCE_FIRST_EVEN;
        else {
          fprintf(stderr, "enter 'even|odd' for field dominance\n");
          free(s.buffer);
          return -1;
        }
      }
      else if (count==9)		/*get color (Y)*/
        s.color.Y = atoi(*argv);
      else if (count==10)		/*get color (Cr)*/
        s.color.Cr = atoi(*argv);
      else if (count==11)		/*get color (Cb)*/
        s.color.Cb = atoi(*argv);
      else if (count==12)		/*get animation center (x)*/
        s.center.x = atoi(*argv);
      else if (count==13)		/*get animation center (y)*/
        s.center.y = atoi(*argv);
      else if (count>13)
        fprintf(stderr, "too many parameters - ignored last one(s)\n");

      count++;
    }
  }

  /*** make last checks ***********************************************/

  if (s.color.Y >= (1<<info.nbit) ||
      s.color.Cr >= (1<<info.nbit) ||
      s.color.Cb >= (1<<info.nbit))
  {
    fprintf(stderr, "color values have to range from 0 to %d\n",
      (1<<info.nbit)-1);
    free(s.buffer);
    return -1;
  }

  if (limit==-1 && s.angle==0.0)	/*set limit and s.angle for one cycle per
                                     second*/
  {
    s.angle = 2.0 * M_PI / frames_per_sec;
    limit = (int)frames_per_sec;
  }
  else if (limit==-1 && s.angle!=0.0)	/*set limit depending on s.angle for one
										 full cycle*/
  {
    /*make absolute value (angle possibly was negative)*/
    limit = abs((int)ceil(2.0 * M_PI/s.angle));	/*because of rounding faults*/
  }
  else if (limit!=-1 && s.angle==0.0)	/*set angle depending on limit
                                          for one full cycle*/
  {
    s.angle = 2.0 * M_PI / limit;
  }

  if (start < 0) {
    fprintf(stderr, "start frame number missing\n");
    free(s.buffer);
    return -1;
  }
  else if (start >= info.setup.nframes) {	/*start <= highest available frame*/
    fprintf(stderr, "frame number %d not available\n", start);
    free(s.buffer);
    return -1;
  }

  if (start+limit-1 >= info.setup.nframes)		/*sequence should not reach out
												  of available frames*/
  {
    fprintf(stderr, "sequence would reach above last available frame\n");
    free(s.buffer);
    return -1;
  }

#ifdef DEBUG
  printf("%dx%d %dbit\ncolormode: %x  sv_mode: %x\n",
    info.xsize, info.ysize, info.nbit, info.colormode, s.sv_mode);
  printf("buffer_size: %d  pixel_values: %d  pixel_size: %d\n",
    s.buffer_size, s.pixel_values, s.pixel_size);
  printf("start: %d  frames: %d  available frames: %d\n",
    start, limit, info.setup.nframes);
  printf("s.type: %d\ns.color: (%d|%d|%d)\n",
    s.type, s.color.Y, s.color.Cr, s.color.Cb);
  printf("s.size: %d\ns.dist: %d\ns.center: (%d|%d)\n",
    s.size, s.dist, s.center.x, s.center.y);
  printf("s.angle: %f  s.startangle: %f\n", s.angle, s.startangle);
  printf("s.anim: %d  s.field: %d\n\n", s.anim, s.field);
#endif

  /*** perform action *************************************************/

  if(info.nbit == 8) {
    switch(info.colormode) {
    case SV_COLORMODE_YUV422: 
      put_pixel_func = &put_pixel_yuv422;
      break;
    case SV_COLORMODE_RGB_BGR:
      put_pixel_func = &put_pixel_rgb;
      break;
    case SV_COLORMODE_YUV422A:
      put_pixel_func = &put_pixel_yuv422a;
      break;
    case SV_COLORMODE_RGBA:
      put_pixel_func = &put_pixel_rgba;
      break;
    }
  } else {
    switch(info.colormode) {
    case SV_COLORMODE_YUV422:
      put_pixel_func = &put_pixel_yuv422_10b;
      break;
    case SV_COLORMODE_RGB_BGR:
      put_pixel_func = &put_pixel_rgb_10b;
      break;
    case SV_COLORMODE_YUV422A:
      put_pixel_func = &put_pixel_yuv422a_10b;
      break;
    case SV_COLORMODE_RGBA:
      put_pixel_func = &put_pixel_rgba_10b;
      break;
    }
  }

  if(put_pixel_func) {
    sequence_loop(sv, &info, start, limit, &s, put_pixel_func);
  }

  free(s.buffer);

  return 0;
}
/*** end sv_sequence() ************************************************/
