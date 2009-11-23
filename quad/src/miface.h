/* miface.h
 *
 * Definitions for Linear Systems Ltd. Master interfaces.
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

#ifndef _MIFACE_H
#define _MIFACE_H

#include <linux/kernel.h> /* container_of () */

#include <linux/types.h> /* uid_t */
#include <linux/list.h> /* list_head */
#include <linux/wait.h> /* wait_queue_head_t */
#include <linux/device.h> /* class_device */
#include <linux/fs.h>
#include <linux/cdev.h> /* cdev */

#include <asm/semaphore.h> /* semaphore */

#include "mdev.h"

#define MASTER_MINORBITS	8

#define MASTER_DIRECTION_TX	0
#define MASTER_DIRECTION_RX	1

/**
 * master_iface - generic Master interface
 * @list: handle for linked list of interfaces for this card
 * @list_all: handle for linked list of all interfaces
 * @direction: direction of data flow
 * @cdev: character device structure
 * @capabilities: capabilities flags
 * @buffers: number of buffers
 * @bufsize: number of bytes in each buffer
 * @clksrc: transmitter clock source
 * @count27: 27 MHz counter flag
 * @granularity: buffer size granularity in bytes
 * @mode: operating mode
 * @null_packets: null packet insertion flag
 * @timestamps: packet timestamping flag
 * @transport: transport type
 * @class_dev: pointer to the class device structure
 * @users: usage count
 * @owner: UID of owner
 * @events: events flags
 * @dma: pointer to the DMA buffer management structure
 * @dma_done: DMA done flag
 * @queue: wait queue
 * @buf_sem: lock for cpu_buffer
 * @card: pointer to the board information structure
 **/
struct master_iface {
	struct list_head list;
	struct list_head list_all;
	unsigned int direction;
	struct cdev cdev;
	unsigned int capabilities;
	unsigned int buffers;
	unsigned int bufsize;
	unsigned int clksrc;
	unsigned int count27;
	unsigned int granularity;
	unsigned int mode;
	unsigned int standard;
	unsigned int null_packets;
	unsigned int timestamps;
	unsigned int transport;
	struct class_device *class_dev;
	unsigned int users;
	uid_t owner;
	volatile unsigned long events;
	void *dma;
	volatile unsigned long dma_done;
	wait_queue_head_t queue;
	struct semaphore buf_sem;
	struct master_dev *card;
};

/* External function prototypes */
ssize_t miface_show_version (struct class *cls, char *buf);
ssize_t miface_show_buffers (struct class_device *cd, char *buf);
ssize_t miface_show_bufsize (struct class_device *cd, char *buf);
ssize_t miface_show_clksrc (struct class_device *cd, char *buf);
ssize_t miface_show_count27 (struct class_device *cd, char *buf);
ssize_t miface_show_granularity (struct class_device *cd, char *buf);
ssize_t miface_show_mode (struct class_device *cd, char *buf);
ssize_t miface_show_standard (struct class_device *cd, char *buf);
ssize_t miface_show_null_packets (struct class_device *cd, char *buf);
ssize_t miface_show_timestamps (struct class_device *cd, char *buf);
ssize_t miface_show_transport (struct class_device *cd, char *buf);
void miface_set_boolean_minmaxmult (struct master_iface *iface,
	unsigned long *min, unsigned long *max, unsigned long *mult);
ssize_t miface_store (struct master_iface *iface,
	unsigned int *var,
	const char *buf,
	size_t count,
	unsigned long min,
	unsigned long max,
	unsigned long mult);

#endif

