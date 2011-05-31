/* util.c
 *
 * Utility functions for
 * the Master Linux Software Development Kit example programs.
 *
 * Copyright (C) 2004-2010 Linear Systems Ltd. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of Linear Systems Ltd. nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LINEAR SYSTEMS LTD. "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LINEAR SYSTEMS LTD. OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Linear Systems can be contacted at <http://www.linsys.ca/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

#include "pci_ids.h"
#include "util.h"

static struct util_info info[] = {
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBTX,
		.name = "DVB Master Send",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBRX,
		.name = "DVB Master Receive",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFD,
		.name = "DVB Master FD",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDU,
		.name = "DVB Master FD-U",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDU_R,
		.name = "DVB Master FD-UR",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBTXU,
		.name = "DVB Master III Tx",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBRXU,
		.name = "DVB Master III Rx",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQI,
		.name = "DVB Master Q/i",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_MMSA,
		.name = "MultiMaster SDI-R",
		.flags = UTIL_ASITX | UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDIM,
		.name = "SDI Master",
		.flags = UTIL_SDITX | UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_MMAS,
		.name = "MultiMaster SDI-T",
		.flags = UTIL_SDITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDB,
		.name = "DVB Master FD-B",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDB_R,
		.name = "DVB Master FD-BR",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVB2FD,
		.name = "DVB Master II FD",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVB2FD_R,
		.name = "DVB Master II FD-R",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVB2FD_RS,
		.name = "DVB Master Dual In",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_ATSC2FD,
		.name = "ATSC Master II FD",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_ATSC2FD_R,
		.name = "ATSC Master II FD-R",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_ATSC2FD_RS,
		.name = "ATSC Master II FD-RS",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPFD,
		.name = "DVB Master LP",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDILPFD,
		.name = "SDI Master LP",
		.flags = UTIL_SDITX | UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQLF,
		.name = "DVB Master Q/i",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVB2FDE,
		.name = "DVB Master II FD PCIe ",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVB2FDE_R,
		.name = "DVB Master II FD-R PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVB2FDE_RS,
		.name = "DVB Master Dual In PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDE,
		.name = "DVB Master FD PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDE_R,
		.name = "DVB Master FD-R PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDEB,
		.name = "DVB Master FD-B PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBFDEB_R,
		.name = "DVB Master FD-BR PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBTXE,
		.name = "DVB Master III Tx PCIe",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBRXE,
		.name = "DVB Master III Rx PCIe",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_ATSC2FDE,
		.name = "ATSC Master II FD PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_ATSC2FDE_R,
		.name = "ATSC Master II FD-R PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_MMSAE,
		.name = "MultiMaster SDI-R PCIe",
		.flags = UTIL_ASITX | UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDIME,
		.name = "SDI Master PCIe",
		.flags = UTIL_SDITX | UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_MMASE,
		.name = "MultiMaster SDI-T PCIe",
		.flags = UTIL_SDITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPFDE,
		.name = "DVB Master LP PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDILPFDE,
		.name = "SDI Master LP PCIe",
		.flags = UTIL_SDITX | UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQLF4,
		.name = "DVB Master Q/i",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQIE,
		.name = "DVB Master Q/i PCIe",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQO,
		.name = "DVB Master Q/o",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQOE,
		.name = "DVB Master Q/o PCIe",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQOE,
		.name = "DVB Master Q/o LP PCIe",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC,
		.name = "DVB Master Q/o LP PCIe Mini BNC",
		.flags = UTIL_ASITX
	},	
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQDUAL,
		.name = "DVB Master Quad-2in2out",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQDUALE,
		.name = "DVB Master Quad-2in2out PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDIQOE,
		.name = "SDI Master Q/o",
		.flags = UTIL_SDITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDIQIE,
		.name = "SDI Master Q/i PCIe",
		.flags = UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_SDIQI,
		.name = "SDI Master Q/i",
		.flags = UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQLF,
		.name = "DVB Master Q/i LP PCIe",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC,
		.name = "DVB Master Q/i LP PCIe",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER,
		.name = "DVB Master Q/i LP PCIe",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQDUALE,
		.name = "DVB Master Quad-2in2out LP PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC,
		.name = "DVB Master Quad-2in2out LP PCIe Mini BNC",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQ3IOE,
		.name = "DVB Master Quad 1in3out PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBQ3INOE,
		.name = "DVB Master Quad-3in1out PCIe",
		.flags = UTIL_ASITX | UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPTXE,
		.name = "DVB Master III Tx LP PCIe",
		.flags = UTIL_ASITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_DVBLPRXE,
		.name = "DVB Master III Rx LP PCIe",
		.flags = UTIL_ASIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_HDSDITXE,
		.name = "VidPort SD/HD O",
		.flags = UTIL_SDITX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_HDSDIQI,
		.name = "QuadPort H/i",
		.flags = UTIL_SDIRX
	},
	{
		.id = PCI_DEVICE_ID_LINSYS_HDSDIRXE,
		.name = "VidPort SD/HD I",
		.flags = UTIL_SDIRX
	}
};

ssize_t
util_read (const char *name, char *buf, size_t count)
{
	ssize_t fd, ret;

	if ((fd = open (name, O_RDONLY)) < 0) {
		return fd;
	}
	ret = read (fd, buf, count);
	close (fd);
	return ret;
}

ssize_t
util_write (const char *name, const char *buf, size_t count)
{
	ssize_t fd, ret;

	if ((fd = open (name, O_WRONLY)) < 0) {
		return fd;
	}
	ret = write (fd, buf, count);
	close (fd);
	return ret;
}

ssize_t
util_strtoul (const char *name, unsigned long int *val)
{
	ssize_t ret;
	char data[256], *endptr;
	unsigned long int tmp;

	memset (data, 0, sizeof (data));
	if ((ret = util_read (name, data, sizeof (data))) < 0) {
		return ret;
	}
	tmp = strtoul (data, &endptr, 0);
	if (*endptr != '\n') {
		return -1;
	}
	*val = tmp;
	return ret;
}

ssize_t
util_strtoull (const char *name, unsigned long long int *val)
{
	ssize_t ret;
	char data[256], *endptr;
	unsigned long long int tmp;

	memset (data, 0, sizeof (data));
	if ((ret = util_read (name, data, sizeof (data))) < 0) {
		return ret;
	}
	tmp = strtoull (data, &endptr, 0);
	if (*endptr != '\n') {
		return -1;
	}
	*val = tmp;
	return ret;
}

void
fprinttime (FILE *stream, const char *progname)
{
	time_t timeval;

	time (&timeval);
	fprintf (stream, "%.15s %s[%u]: ",
		(ctime (&timeval)) + 4, progname, getpid ());
	return;
}

struct util_info *
getinfo (unsigned int id)
{
	struct util_info *p = info;

	while (p < (info + sizeof (info) / sizeof (*info))) {
		if (id == p->id) {
			return p;
		}
		p++;
	}
	return NULL;
}

