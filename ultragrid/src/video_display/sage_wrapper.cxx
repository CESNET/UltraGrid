/*
 * FILE:    video_display/sage_wrapper.cxx
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2209 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 */

#include <GL/glut.h>
#include "sage_wrapper.h"
#include <sail.h>
#include <misc.h>

sail *sageInf; // sage sail object

void initSage(int appID, int nodeID, int width, int height, int yuv, int dxt)
{
            sageInf = new sail;
            sailConfig sailCfg;
            sailCfg.init("ultragrid.conf");
            sailCfg.setAppName("ultragrid");
            sailCfg.rank = nodeID;
            sailCfg.resX = width;
            sailCfg.resY = height;

            sageRect renderImageMap;
            renderImageMap.left = 0.0;
            renderImageMap.right = 1.0;
            renderImageMap.bottom = 0.0;
            renderImageMap.top = 1.0;

            sailCfg.imageMap = renderImageMap;
            if (!yuv) {
                    if(dxt)
                            sailCfg.pixFmt = PIXFMT_DXT;
                    else
                            sailCfg.pixFmt = PIXFMT_8888_INV;
            } else {
                    sailCfg.pixFmt = PIXFMT_YUV;
            }
            //sailCfg.rowOrd = BOTTOM_TO_TOP;
            sailCfg.rowOrd = TOP_TO_BOTTOM;
            sailCfg.master = true;

            sageInf->init(sailCfg);
}

void sage_shutdown()
{
    sageInf->shutdown();
    delete sageInf;
}

void sage_swapBuffer()
{
    sageInf->swapBuffer(SAGE_NON_BLOCKING);
}

GLubyte * sage_getBuffer()
{
    return (GLubyte *)sageInf->getBuffer();
}
