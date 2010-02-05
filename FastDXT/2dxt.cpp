/******************************************************************************
 * Fast DXT - a realtime DXT compression tool
 *
 * Author : Luc Renambot
 *
 * Copyright (C) 2007 Electronic Visualization Laboratory,
 * University of Illinois at Chicago
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either Version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 *****************************************************************************/

//
// Usage: ./2dxt width height type rawfile dxtfile
//

#define NUMTHREAD 2

#include "libdxt.h"

int
main(int argc, char** argv)
{
  byte *in;
  byte *out;
  int nbbytes;

  if (argc != 6)
  {
	fprintf(stderr, "Usage: ./2dxt width height type rawfile dxtfile\n");
	exit(0);
  }
	// Initialize some timing functions and else
  aInitialize();

  /*
    Read an image.
  */
  unsigned long width = atoi(argv[1]);
  unsigned long height = atoi(argv[2]);

  int format = 1;
  format = atoi(argv[3]);

  in = (byte*)memalign(16, width*height*4);
  memset(in, 0, width*height*4);
  
  FILE *f=fopen(argv[4], "rb");
  int res=(int)fread(in, 1, width*height*4, f);
  fclose(f);

  out = (byte*)memalign(16, width*height*4);
  memset(out, 0, width*height*4);

  fprintf(stderr, "Converting to raw: %ldx%ld\n", width, height);

  double t1, t2;
  t1 = aTime();
  nbbytes = 0;
  switch (format) {
  case 1:
    nbbytes = CompressDXT(in, out, width, height, FORMAT_DXT1, NUMTHREAD);
    fprintf(stderr, "Converted to DXT1: from %d bytes to %ld bytes\n",
	    width*height*4, nbbytes);
    break;
  case 5:
    nbbytes = CompressDXT(in, out, width, height, FORMAT_DXT5, NUMTHREAD);
    fprintf(stderr, "Converted to DXT5: from %d bytes to %ld bytes\n",
	  width*height*4, nbbytes);
    break;
  case 6:
    nbbytes = CompressDXT(in, out, width, height, FORMAT_DXT5YCOCG, NUMTHREAD);
    fprintf(stderr, "Converted to DXT5-YCoCg: from %d bytes to %ld bytes\n",
	    width*height*4, nbbytes);
    break;
  }
  t2 = aTime();

  fprintf(stderr, "Time %.2f sec, Freq %.2f Hz\n",
	  t2-t1, 1.0/(t2-t1) );
  fprintf(stderr, "MP/sec %.2f\n",
          ((double)(width*height)) / ((t2-t1)*1000000.0) );

  FILE *g=fopen(argv[5], "wb+");
  fwrite(&width, 4, 1, g);
  fwrite(&height, 4, 1, g);
  int res2=(int)fwrite(out, 1, nbbytes, g);
  fclose(g);

  memfree(in);
  memfree(out);

  return 0;
}

