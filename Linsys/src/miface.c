/* miface.c
 *
 * Support functions for Linear Systems Ltd. Master interfaces.
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

#include <linux/kernel.h> /* snprintf () */
#include <linux/version.h> /* LINUX_VERSION_CODE */

#include <linux/fs.h> /* file */
#include <linux/sched.h> /* current_uid () */
#include <linux/poll.h> /* poll_wait () */
#include <linux/dma-mapping.h> /* DMA_FROM_DEVICE */
#include <linux/mutex.h> /* mutex_lock () */

#include "../include/master.h"
#include "miface.h"
#include "mdev.h"
#include "mdma.h"

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27)
#define current_uid() (current->uid)
#define current_euid() (current->euid)
#endif

#ifndef IRQF_SHARED
#define IRQF_SHARED SA_SHIRQ
#endif

/* Static function prototypes */
static unsigned int total_users (struct master_dev *card);

static const char fmt_u[] = "%u\n";
static const char fmt_x[] = "0x%04X\n";

/**
 * miface_show_version - class attribute read handler
 * @cls: class being read
 * @attr: class attribute
 * @buf: output buffer
 **/
ssize_t
miface_show_version (struct class *cls,
	struct class_attribute *attr,
	char *buf)
{
	return snprintf (buf, PAGE_SIZE, "%s\n", MASTER_DRIVER_VERSION);
}

/**
 * miface_show_* - interface attribute read handler
 * @dev: device being read
 * @attr: device attribute
 * @buf: output buffer
 **/
#define MIFACE_SHOW(var,format) \
	ssize_t miface_show_##var (struct device *dev, \
		struct device_attribute *attr, \
		char *buf) \
	{ \
		struct master_iface *iface = dev_get_drvdata (dev); \
		return snprintf (buf, PAGE_SIZE, format, iface->var); \
	}
MIFACE_SHOW(buffers,fmt_u)
MIFACE_SHOW(bufsize,fmt_u)
MIFACE_SHOW(clksrc,fmt_u)
MIFACE_SHOW(count27,fmt_u)
MIFACE_SHOW(granularity,fmt_u)
MIFACE_SHOW(mode,fmt_u)
MIFACE_SHOW(vanc,fmt_u)
MIFACE_SHOW(null_packets,fmt_u)
MIFACE_SHOW(timestamps,fmt_u)
MIFACE_SHOW(transport,fmt_u)
MIFACE_SHOW(frmode,fmt_u)
MIFACE_SHOW(sample_size,fmt_u)
MIFACE_SHOW(sample_rate,fmt_u)
MIFACE_SHOW(channels,fmt_u)
MIFACE_SHOW(vb1cnt,fmt_u)
MIFACE_SHOW(vb1ln1,fmt_u)
MIFACE_SHOW(vb2cnt,fmt_u)
MIFACE_SHOW(vb2ln1,fmt_u)
MIFACE_SHOW(nonaudio,fmt_x)

/**
 * miface_store - generic Master interface attribute write handler
 * @iface: interface being written
 * @var: pointer to variable being written
 * @val: new attribute value
 **/
ssize_t
miface_store (struct master_iface *iface,
	unsigned int *var,
	unsigned long val)
{
	int retcode = 0;

	mutex_lock (&iface->card->users_mutex);
	if (iface->users) {
		retcode = -EBUSY;
		goto OUT;
	}
	*var = val;
OUT:
	mutex_unlock (&iface->card->users_mutex);
	return retcode;
}

/**
 * total_users - return the total usage count for a device
 * @card: Master device
 *
 * Call this with card->users_mutex held!
 **/
static unsigned int
total_users (struct master_dev *card)
{
	struct list_head *p;
	struct master_iface *iface;
	unsigned int total_users = 0;

	list_for_each (p, &card->iface_list) {
		iface = list_entry (p, struct master_iface, list);
		total_users += iface->users;
	}
	return total_users;
}

/**
 * miface_open - Master interface open() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
miface_open (struct inode *inode, struct file *filp)
{
	struct master_iface *iface =
		container_of(inode->i_cdev,struct master_iface,cdev);
	struct master_dev *card = iface->card;

	filp->private_data = iface;
	mutex_lock (&card->users_mutex);
	if (iface->users) {
		if (((iface->owner != current_uid()) &&
			(iface->owner != current_euid()) &&
			!capable (CAP_DAC_OVERRIDE))) {
			mutex_unlock (&card->users_mutex);
			return -EBUSY;
		}
	} else {
		/* Reset flags */
		iface->events = 0;
		__set_bit (0, &iface->dma_done);

		/* Create a DMA buffer management structure */
		if ((iface->dma = mdma_alloc (card->parent,
			iface->dma_ops,
			iface->data_addr,
			iface->buffers,
			iface->bufsize,
			(iface->direction == MASTER_DIRECTION_TX) ?
			DMA_TO_DEVICE : DMA_FROM_DEVICE,
			iface->dma_flags)) == NULL) {
			mutex_unlock (&card->users_mutex);
			return -ENOMEM;
		}

		/* Initialize the interface */
		if (iface->ops->init) {
			iface->ops->init (iface);
		}

		/* If we are the first user, install the interrupt handler */
		if (!total_users (card) &&
			(request_irq (card->irq,
				card->irq_handler,
				IRQF_SHARED,
				filp->f_op->owner->name,
				card) != 0)) {
			mdma_free (iface->dma);
			mutex_unlock (&card->users_mutex);
			return -EBUSY;
		}

		/* Activate the interface */
		if (iface->ops->start) {
			iface->ops->start (iface);
		}

		iface->owner = current_uid();
	}
	iface->users++;
	mutex_unlock (&card->users_mutex);
	return nonseekable_open (inode, filp);
}

/**
 * miface_write - Master interface write() method
 * @filp: file
 * @data: data
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
ssize_t
miface_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;
	ssize_t ret;

	mutex_lock (&iface->buf_mutex);
	if ((filp->f_flags & O_NONBLOCK) && mdma_tx_isfull (dma)) {
		mutex_unlock (&iface->buf_mutex);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !mdma_tx_isfull (dma))) {
		mutex_unlock (&iface->buf_mutex);
		return -ERESTARTSYS;
	}

	ret = iface->dma_ops->write (dma, data, length);

	/* If DMA is stopped and the buffer queue is half full,
	 * enable and start DMA */
	if (test_bit (0, &iface->dma_done) &&
		mdma_tx_buflevel (dma) >= iface->buffers / 2) {
		iface->ops->start_tx_dma (iface);
	}

	mutex_unlock (&iface->buf_mutex);
	return ret;
}

/**
 * miface_read - Master interface read() method
 * @filp: file
 * @data: read buffer
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
ssize_t
miface_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;
	ssize_t ret;

	mutex_lock (&iface->buf_mutex);
	if ((filp->f_flags & O_NONBLOCK) && mdma_rx_isempty (dma)) {
		mutex_unlock (&iface->buf_mutex);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !mdma_rx_isempty (dma))) {
		mutex_unlock (&iface->buf_mutex);
		return -ERESTARTSYS;
	}

	ret = iface->dma_ops->read (dma, data, length);

	mutex_unlock (&iface->buf_mutex);
	return ret;
}

/**
 * miface_txpoll - transmitter poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
miface_txpoll (struct file *filp, poll_table *wait)
{
	struct master_iface *iface = filp->private_data;
	unsigned int mask = 0;

	poll_wait (filp, &iface->queue, wait);
	if (!mdma_tx_isfull (iface->dma)) {
		mask |= POLLOUT | POLLWRNORM;
	}
	if (iface->events) {
		mask |= POLLPRI;
	}
	return mask;
}

/**
 * miface_rxpoll - receiver poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
miface_rxpoll (struct file *filp, poll_table *wait)
{
	struct master_iface *iface = filp->private_data;
	unsigned int mask = 0;

	poll_wait (filp, &iface->queue, wait);
	if (!mdma_rx_isempty (iface->dma)) {
		mask |= POLLIN | POLLRDNORM;
	}
	if (iface->events) {
		mask |= POLLPRI;
	}
	return mask;
}

/**
 * miface_mmap - Master interface mmap() method
 * @filp: file
 * @vma: VMA
 **/
int
miface_mmap (struct file *filp, struct vm_area_struct *vma)
{
	struct master_iface *iface = filp->private_data;

	vma->vm_ops = &mdma_vm_ops;
	vma->vm_flags |= VM_RESERVED;
	vma->vm_private_data = iface->dma;
	return 0;
}

/**
 * miface_release - Master interface release() method
 * @inode: inode
 * @filp: file
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
miface_release (struct inode *inode, struct file *filp)
{
	struct master_iface *iface = filp->private_data;
	struct master_dev *card = iface->card;

	mutex_lock (&card->users_mutex);
	if (iface->users == 1) {
		if (iface->ops->stop) {
			iface->ops->stop (iface);
		}
		if (iface->ops->exit) {
			iface->ops->exit (iface);
		}

		/* If we are the last user, uninstall the interrupt handler */
		if (total_users (card) == 1) {
			free_irq (card->irq, card);
		}

		/* Destroy the DMA buffer management structure */
		mdma_free (iface->dma);
	}
	iface->users--;
	mutex_unlock (&card->users_mutex);
	return 0;
}

/**
 * miface_txdqbuf - transmitter dqbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Dequeue a transmitter buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
miface_txdqbuf (struct file *filp, unsigned int arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	if ((filp->f_flags & O_NONBLOCK) && mdma_tx_isfull (dma)) {
		mutex_unlock (&iface->buf_mutex);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !mdma_tx_isfull (dma))) {
		mutex_unlock (&iface->buf_mutex);
		return -ERESTARTSYS;
	}
	if (iface->dma_ops->txdqbuf (dma, arg) < 0) {
		mutex_unlock (&iface->buf_mutex);
		return -EINVAL;
	}
	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * miface_txqbuf - transmitter qbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Queue a transmitter buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
miface_txqbuf (struct file *filp, unsigned int arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	if (iface->dma_ops->txqbuf (dma, arg) < 0) {
		mutex_unlock (&iface->buf_mutex);
		return -EINVAL;
	}

	/* If DMA is stopped and the buffer queue is half full,
	 * enable and start DMA */
	if (test_bit (0, &iface->dma_done) &&
		mdma_tx_buflevel (dma) >= iface->buffers / 2) {
		iface->ops->start_tx_dma (iface);
	}

	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * miface_rxdqbuf - receiver dqbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Dequeue a receiver buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
miface_rxdqbuf (struct file *filp, unsigned int arg)
{
	struct master_iface *iface = filp->private_data;
	struct master_dma *dma = iface->dma;

	mutex_lock (&iface->buf_mutex);
	if ((filp->f_flags & O_NONBLOCK) && mdma_rx_isempty (dma)) {
		mutex_unlock (&iface->buf_mutex);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !mdma_rx_isempty (dma))) {
		mutex_unlock (&iface->buf_mutex);
		return -ERESTARTSYS;
	}
	if (iface->dma_ops->rxdqbuf (dma, arg) < 0) {
		mutex_unlock (&iface->buf_mutex);
		return -EINVAL;
	}
	mutex_unlock (&iface->buf_mutex);
	return 0;
}

/**
 * miface_rxqbuf - receiver qbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Queue a receiver buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
miface_rxqbuf (struct file *filp, unsigned int arg)
{
	struct master_iface *iface = filp->private_data;

	mutex_lock (&iface->buf_mutex);
	if (iface->dma_ops->rxqbuf (iface->dma, arg) < 0) {
		mutex_unlock (&iface->buf_mutex);
		return -EINVAL;
	}
	mutex_unlock (&iface->buf_mutex);
	return 0;
}

