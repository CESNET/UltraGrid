/* miface.c
 *
 * Support functions for Linear Systems Ltd. Master interfaces.
 *
 * Copyright (C) 2005, 2007 Linear Systems Ltd.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either Version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public Licence for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * Linear Systems can be contacted at <http://www.linsys.ca/>.
 *
 */

#include <linux/kernel.h> /* snprintf () */

#include <linux/slab.h> /* kfree () */

#include <asm/semaphore.h> /* down_interruptible () */

#include "../include/master.h"
#include "miface.h"

static const char fmt_u[] = "%u\n";

/**
 * miface_show_version - class attribute read handler
 * @cls: class being read
 * @buf: output buffer
 **/
ssize_t
miface_show_version (struct class *cls, char *buf)
{
	return snprintf (buf, PAGE_SIZE, "%s\n", MASTER_DRIVER_VERSION);
}

/**
 * miface_show_* - interface attribute read handler
 * @cd: class_device being read
 * @buf: output buffer
 **/
#define MIFACE_SHOW(var,format) \
	ssize_t miface_show_##var (struct class_device *cd, char *buf) \
	{ \
		struct master_iface *iface = class_get_devdata (cd); \
		return snprintf (buf, PAGE_SIZE, format, iface->var); \
	}
MIFACE_SHOW(buffers,fmt_u)
MIFACE_SHOW(bufsize,fmt_u)
MIFACE_SHOW(clksrc,fmt_u)
MIFACE_SHOW(count27,fmt_u)
MIFACE_SHOW(granularity,fmt_u)
MIFACE_SHOW(mode,fmt_u)
MIFACE_SHOW(standard,fmt_u)
MIFACE_SHOW(null_packets,fmt_u)
MIFACE_SHOW(timestamps,fmt_u)
MIFACE_SHOW(transport,fmt_u)

/**
 * miface_set_boolean_minmaxmult - return the desired properties for a boolean attribute
 * @iface: interface being written
 * @min: pointer to the minimum value
 * @max: pointer to the maximum value
 * @mult: pointer to the granularity
 **/
void
miface_set_boolean_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult)
{
	*min = 0;
	*max = ULONG_MAX;
	*mult = 1;
	return;
}

/**
 * miface_store - generic Master interface attribute write handler
 * @iface: interface being written
 * @var: pointer to variable being written
 * @buf: input buffer
 * @count: buffer size
 * @min: minimum value
 * @max: maximum value
 * @mult: granularity
 **/
ssize_t
miface_store (struct master_iface *iface,
	unsigned int *var,
	const char *buf,
	size_t count,
	unsigned long min,
	unsigned long max,
	unsigned long mult)
{
	char *endp;
	unsigned long new = simple_strtoul (buf, &endp, 0);
	int retcode = count;

	if (endp == buf) {
		return -EINVAL;
	}
	if (down_interruptible (&iface->card->users_sem)) {
		return -ERESTARTSYS;
	}
	if (iface->users) {
		retcode = -EBUSY;
		goto OUT;
	}
	if ((new < min) || (new > max) || (new % mult)) {
		retcode = -EINVAL;
		goto OUT;
	}
	*var = new;
OUT:
	up (&iface->card->users_sem);
	return retcode;
}

