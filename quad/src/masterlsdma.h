/* masterlsdma.h
 *
 * Header file for masterlsdma.c.
 *
 * Copyright (C) 2004-2005 Linear Systems Ltd.
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

#ifndef _MASTERLSDMA_H
#define _MASTERLSDMA_H

#include <linux/types.h> /* size_t, loff_t */
#include <linux/fs.h> /* file */
#include <linux/poll.h> /* poll_table */
#include <linux/mm.h> /* vm_area_struct */

#include "mdev.h"
#include "miface.h"

/* External function prototypes */
int masterlsdma_open (struct inode *inode,
	struct file *filp,
	void (*init)(struct master_iface *iface),
	void (*start)(struct master_iface *iface),
	u32 data_addr,
	unsigned int flags);
ssize_t masterlsdma_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset);
ssize_t masterlsdma_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset);
unsigned int masterlsdma_txpoll (struct file *filp, poll_table *wait);
unsigned int masterlsdma_rxpoll (struct file *filp, poll_table *wait);
int masterlsdma_mmap (struct file *filp, struct vm_area_struct *vma);
int masterlsdma_release (struct master_iface *iface,
	void (*stop)(struct master_iface *iface),
	void (*exit)(struct master_iface *iface));
int masterlsdma_txdqbuf (struct file *filp, unsigned long arg);
int masterlsdma_txqbuf (struct file *filp, unsigned long arg);
int masterlsdma_rxdqbuf (struct file *filp, unsigned long arg);
int masterlsdma_rxqbuf (struct file *filp, unsigned long arg);

/* Inline functions */

#ifndef wait_event_timeout
#define __wait_event_timeout(wq, condition, ret)				\
do {															\
	DEFINE_WAIT(__wait);										\
																\
	for (;;) {													\
		prepare_to_wait(&wq, &__wait, TASK_UNINTERRUPTIBLE);	\
		if (condition)											\
			break;												\
		ret = schedule_timeout(ret);							\
		if (!ret)												\
			break;												\
	}															\
	finish_wait(&wq, &__wait);									\
} while (0)
#define wait_event_timeout(wq, condition, timeout)				\
({																\
	long __ret = timeout;										\
	if (!(condition))											\
		__wait_event_timeout(wq, condition, __ret);				\
	__ret;														\
})
#endif

#endif

