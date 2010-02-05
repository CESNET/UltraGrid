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

#include "dxt.h"
#include "util.h"

#if defined(__APPLE__)
#define memalign(x,y) malloc((y))
#else
#include <malloc.h>
#endif

int
main(int argc, char** argv)
{
  ALIGN16( byte *in );
  ALIGN16( byte *out);
  double t1, t2;
  int nbbytes;

	// Initialize some timing functions and else
  aInitialize();

  /*
    Read an image.
  */    
  unsigned long width = atoi(argv[1]);
  unsigned long height = atoi(argv[2]);

  in = (byte*)memalign(16, width*height*4);
  memset(in, 0, width*height*4);
  
  FILE *f=fopen(argv[3], "rb");
  fread(in, 1, width*height*4, f);
  fclose(f);

  out = (byte*)memalign(16, width*height*4);
  memset(out, 0, width*height*4);

  fprintf(stderr, "Converting to raw: %ldx%ld\n", width, height);

  t1 = aTime();
  for (int k=0;k<100;k++)
    {
		CompressImageDXT1( in, out, width, height, nbbytes);
    }
  t2 = aTime();

  fprintf(stderr, "Converted to DXT: %d byte, compression %ld\n",
	  nbbytes, (width*height*4) / (nbbytes));
  fprintf(stderr, "Time %.2f sec, Single %.2f sec, Freq %.2f Hz\n",
	  t2-t1, (t2-t1)/100.0, 100.0/(t2-t1) );
  fprintf(stderr, "MP/sec %.2f\n",
	  ((double)(width*height)) / ((t2-t1)*10000.0) );

#if 1
  FILE *g=fopen("out.dxt", "wb+");
  fwrite(&width, 4, 1, g);
  fwrite(&height, 4, 1, g);
  //nbbytes = width * height * 4 / 4; //DXT5
  nbbytes = width * height * 3 / 6; // DXT1
  fwrite(out, 1, nbbytes, g);
  fclose(g);
#endif

  memfree(in);
  memfree(out);

  return 0;
}
