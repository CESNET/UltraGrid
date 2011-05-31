/* miface.h
 *
 * Definitions for Linear Systems Ltd. Master interfaces.
 *
 * Copyright (C) 2005-2010 Linear Systems Ltd.
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

#include <linux/version.h> /* LINUX_VERSION_CODE */

#include <linux/types.h> /* uid_t */
#include <linux/list.h> /* list_head */
#include <linux/wait.h> /* wait_queue_head_t */
#include <linux/device.h> /* device */
#include <linux/fs.h>
#include <linux/cdev.h> /* cdev */
#include <linux/mutex.h> /* mutex */
#include <linux/poll.h> /* poll_table */
#include <linux/mm.h> /* vm_area_struct */

#include "mdev.h"
#include "mdma.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,35))
#define FSYNC_HANDLER(name,filp,datasync) \
	name (struct file *filp, struct dentry *dentry, int datasync)
#else
#define FSYNC_HANDLER(name,filp,datasync) \
	name (struct file *filp, int datasync)
#endif

#define MASTER_MINORBITS	8

#define MASTER_DIRECTION_TX	0
#define MASTER_DIRECTION_RX	1

struct master_iface;

/**
 * master_iface_operations - Master interface helper functions
 * @init: Master interface initialization function
 * @start: Master interface activation function
 * @stop: Master interface deactivation function
 * @exit: Master interface cleanup function
 * @start_tx: transmit DMA start function
 **/
struct master_iface_operations {
	void (*init) (struct master_iface *iface);
	void (*start) (struct master_iface *iface);
	void (*stop) (struct master_iface *iface);
	void (*exit) (struct master_iface *iface);
	void (*start_tx_dma) (struct master_iface *iface);
};

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
 * @frmode: frame mode
 * @vanc: vertical ancillary space
 * @sample_size: audio sample size
 * @sample_rate: audio sample rate
 * @channels: enable audio channel
 * @vb1cnt: vertical blanking interval 1 count
 * @vb1ln1: vertical blanking interval 1 line 1
 * @vb2cnt: vertical blanking interval 2 count
 * @vb2ln1: vertical blanking interval 1 line 1
 * @nonaudio: other than linear PCM samples
 * @null_packets: null packet insertion flag
 * @timestamps: packet timestamping flag
 * @transport: transport type
 * @dev: pointer to the device structure
 * @ops: pointer to Master interface helper functions
 * @users: usage count
 * @owner: UID of owner
 * @events: events flags
 * @dma_ops: pointer to DMA helper functions
 * @data_addr: local bus address of the FIFO
 * @dma_flags: DMA buffer allocation flags
 * @dma: pointer to the DMA buffer management structure
 * @dma_done: DMA done flag
 * @queue: wait queue
 * @buf_mutex: mutex for cpu_buffer
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
	unsigned int frmode;
	unsigned int vanc;
	unsigned int sample_size;
	unsigned int sample_rate;
	unsigned int channels;
	unsigned int vb1cnt;
	unsigned int vb1ln1;
	unsigned int vb2cnt;
	unsigned int vb2ln1;
	unsigned int nonaudio;
	unsigned int null_packets;
	unsigned int timestamps;
	unsigned int transport;
	struct device *dev;
	struct master_iface_operations *ops;
	unsigned int users;
	uid_t owner;
	volatile unsigned long events;
	struct master_dma_operations *dma_ops;
	u32 data_addr;
	unsigned int dma_flags;
	void *dma;
	volatile unsigned long dma_done;
	wait_queue_head_t queue;
	struct mutex buf_mutex;
	struct master_dev *card;
};

/* External function prototypes */

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,34))
#define miface_show_version(cls,attr,buf) miface_show_version(cls,buf)
#endif
ssize_t miface_show_version (struct class *cls,
	struct class_attribute *attr,
	char *buf);
ssize_t miface_show_buffers (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_bufsize (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_clksrc (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_count27 (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_granularity (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_mode (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_vanc (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_frmode (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_sample_size (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_sample_rate (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_channels (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_vb1cnt (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_vb1ln1 (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_vb2cnt (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_vb2ln1 (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_nonaudio (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_null_packets (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_timestamps (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_show_transport (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t miface_store (struct master_iface *iface,
	unsigned int *var,
	unsigned long val);
int miface_open (struct inode *inode, struct file *filp);
ssize_t miface_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset);
ssize_t miface_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset);
unsigned int miface_txpoll (struct file *filp, poll_table *wait);
unsigned int miface_rxpoll (struct file *filp, poll_table *wait);
int miface_mmap (struct file *filp, struct vm_area_struct *vma);
int miface_release (struct inode *inode, struct file *filp);
int miface_txdqbuf (struct file *filp, unsigned int arg);
int miface_txqbuf (struct file *filp, unsigned int arg);
int miface_rxdqbuf (struct file *filp, unsigned int arg);
int miface_rxqbuf (struct file *filp, unsigned int arg);

#endif

