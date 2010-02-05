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

#ifdef __cplusplus
#include "dxt.h"
#include "util.h"
#endif

#define FORMAT_DXT1      1
#define FORMAT_DXT5      2
#define FORMAT_DXT5YCOCG 3

#ifdef __cplusplus
extern "C"
#endif
int CompressDXT(const unsigned char *in, unsigned char *out, int width, int height, int format, int numthreads);
#ifdef __cplusplus
extern "C"
#endif
int DirectDXT1(const unsigned char *in, unsigned char *out, int width, int height);


