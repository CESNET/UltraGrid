/* masterplx.c
 *
 * Functions related to interfaces which use the PLX PCI 9080.
 *
 * Copyright (C) 2004-2007 Linear Systems Ltd.
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

#include <linux/version.h> /* LINUX_VERSION_CODE */

#include <linux/fs.h> /* file */
#include <linux/poll.h> /* poll_wait () */
#include <linux/errno.h> /* error codes */
#include <linux/sched.h> /* wait_event_interruptible () */
#include <linux/delay.h> /* udelay () */

#include <asm/semaphore.h> /* down_interruptible () */
#include <asm/bitops.h> /* clear_bit () */

#include "mdev.h"
// Temporary fix for Linux kernel 2.6.21
#include "plx9080.c"
#include "masterplx.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,8))
static inline int
nonseekable_open (struct inode *inode, struct file *filp)
{
	return 0;
}
#endif

#ifndef IRQF_SHARED
#define IRQF_SHARED SA_SHIRQ
#endif

/**
 * masterplx_reset_bridge - reset the PCI 9080
 * @card: Master device containing the bridge
 **/
void
masterplx_reset_bridge (struct master_dev *card)
{
	unsigned int cntrl;

	/* Set the PCI Read Command to Memory Read Multiple */
	/* and pulse the PCI 9080 software reset bit */
	cntrl = (readl (card->bridge_addr + PLX_CNTRL) &
		~(PLX_CNTRL_PCIRCCDMA_MASK | PLX_CNTRL_PCIMRCCDM_MASK)) | 0xc0c;
	writel (cntrl | PLX_CNTRL_RESET, card->bridge_addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_CNTRL);
	udelay (100L);
	writel (cntrl, card->bridge_addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_CNTRL);
	udelay (100L);

	/* Reload the PCI 9080 local configuration registers from the EEPROM */
	writel (cntrl | PLX_CNTRL_RECONFIG, card->bridge_addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_CNTRL);
	/* Sleep for at least 6 ms */
	set_current_state (TASK_UNINTERRUPTIBLE);
	schedule_timeout (1 + HZ / 167);
	writel (cntrl, card->bridge_addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (card->bridge_addr + PLX_CNTRL);
}

/**
 * masterplx_open - open() method
 * @inode: inode
 * @filp: file
 * @init: interface initialization function
 * @start: interface activation function
 * @data_addr: local bus address of the data register
 * @flags: allocation flags
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
masterplx_open (struct inode *inode,
	struct file *filp,
	void (*init)(struct master_iface *iface),
	void (*start)(struct master_iface *iface),
	u32 data_addr,
	unsigned int flags)
{
	struct master_iface *iface =
		container_of(inode->i_cdev,struct master_iface,cdev);
	struct master_dev *card = iface->card;

	filp->private_data = iface;
	if (down_interruptible (&card->users_sem)) {
		return -ERESTARTSYS;
	}
	if (iface->users) {
		if (((iface->owner != current->uid) &&
			(iface->owner != current->euid) &&
			!capable (CAP_DAC_OVERRIDE))) {
			up (&card->users_sem);
			return -EBUSY;
		}
	} else {
		/* Reset flags */
		iface->events = 0;
		__set_bit (0, &iface->dma_done);

		/* Create a DMA buffer management structure */
		if ((iface->dma = plx_alloc (card->pdev,
			data_addr,
			iface->buffers,
			iface->bufsize,
			(iface->direction == MASTER_DIRECTION_TX) ?
			PCI_DMA_TODEVICE : PCI_DMA_FROMDEVICE,
			flags)) == NULL) {
			up (&card->users_sem);
			return -ENOMEM;
		}

		/* Initialize the interface */
		if (init) {
			init (iface);
		}

		/* If we are the first user, install the interrupt handler */
		if (!mdev_users (card) &&
			(request_irq (card->pdev->irq,
				card->irq_handler,
				IRQF_SHARED,
				card->name,
				card) != 0)) {
			plx_free (iface->dma);
			up (&card->users_sem);
			return -EBUSY;
		}

		/* Activate the interface */
		if (start) {
			start (iface);
		}

		iface->owner = current->uid;
	}
	iface->users++;
	up (&card->users_sem);
	return nonseekable_open (inode, filp);
}

/**
 * masterplx_write - write() method
 * @filp: file
 * @data: data
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes written on success.
 **/
ssize_t
masterplx_write (struct file *filp,
	const char __user *data,
	size_t length,
	loff_t *offset)
{
	struct master_iface *iface = filp->private_data;
	struct plx_dma *dma = iface->dma;
	ssize_t ret;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,8))
	if (offset != &filp->f_pos) {
		return -ESPIPE;
	}
#endif
	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	if ((filp->f_flags & O_NONBLOCK) && plx_tx_isfull (dma)) {
		up (&iface->buf_sem);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !plx_tx_isfull (dma))) {
		up (&iface->buf_sem);
		return -ERESTARTSYS;
	}

	ret = plx_write (dma, data, length);

	/* If DMA is stopped and the buffer queue is half full,
	 * enable and start DMA */
	if (test_bit (0, &iface->dma_done) &&
		plx_tx_buflevel (dma) >= iface->buffers / 2) {
		struct master_dev *card = iface->card;

		writel (plx_head_desc_bus_addr (dma) |
			PLX_DMADPR_DLOC_PCI,
			card->bridge_addr + PLX_DMADPR0);
		clear_bit (0, &iface->dma_done);
		wmb ();
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
			card->bridge_addr + PLX_DMACSR0);
		/* Dummy read to flush PCI posted writes */
		readb (card->bridge_addr + PLX_DMACSR0);
	}

	up (&iface->buf_sem);
	return ret;
}

/**
 * masterplx_read - read() method
 * @filp: file
 * @data: read buffer
 * @length: size of data
 * @offset:
 *
 * Returns a negative error code on failure and
 * the number of bytes read on success.
 **/
ssize_t
masterplx_read (struct file *filp,
	char __user *data,
	size_t length,
	loff_t *offset)
{
	struct master_iface *iface = filp->private_data;
	struct plx_dma *dma = iface->dma;
	ssize_t ret;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,8))
	if (offset != &filp->f_pos) {
		return -ESPIPE;
	}
#endif
	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	if ((filp->f_flags & O_NONBLOCK) && plx_rx_isempty (dma)) {
		up (&iface->buf_sem);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !plx_rx_isempty (dma))) {
		up (&iface->buf_sem);
		return -ERESTARTSYS;
	}

	ret = plx_read (dma, data, length);

	up (&iface->buf_sem);
	return ret;
}

/**
 * masterplx_txpoll - transmitter poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
masterplx_txpoll (struct file *filp, poll_table *wait)
{
	struct master_iface *iface = filp->private_data;
	unsigned int mask = 0;

	poll_wait (filp, &iface->queue, wait);
	if (!plx_tx_isfull (iface->dma)) {
		mask |= POLLOUT | POLLWRNORM;
	}
	if (iface->events) {
		mask |= POLLPRI;
	}
	return mask;
}

/**
 * masterplx_rxpoll - receiver poll() method
 * @filp: file
 * @wait:
 **/
unsigned int
masterplx_rxpoll (struct file *filp, poll_table *wait)
{
	struct master_iface *iface = filp->private_data;
	unsigned int mask = 0;

	poll_wait (filp, &iface->queue, wait);
	if (!plx_rx_isempty (iface->dma)) {
		mask |= POLLIN | POLLRDNORM;
	}
	if (iface->events) {
		mask |= POLLPRI;
	}
	return mask;
}

/**
 * masterplx_mmap - mmap() method
 * @filp: file
 * @vma: VMA
 **/
int
masterplx_mmap (struct file *filp, struct vm_area_struct *vma)
{
	struct master_iface *iface = filp->private_data;

	vma->vm_ops = &plx_vm_ops;
	vma->vm_flags |= VM_RESERVED;
	vma->vm_private_data = iface->dma;
	return 0;
}

/**
 * masterplx_release - release() method
 * @iface: interface
 * @stop: interface deactivation function
 * @exit: interface shutdown function
 *
 * Returns a negative error code on failure and 0 on success.
 **/
int
masterplx_release (struct master_iface *iface,
	void (*stop)(struct master_iface *iface),
	void (*exit)(struct master_iface *iface))
{
	struct master_dev *card = iface->card;

	if (down_interruptible (&card->users_sem)) {
		return -ERESTARTSYS;
	}
	if (iface->users == 1) {
		if (stop) {
			stop (iface);
		}
		if (exit) {
			exit (iface);
		}

		/* If we are the last user, uninstall the interrupt handler */
		if (mdev_users (card) == 1) {
			free_irq (card->pdev->irq, card);
		}

		/* Destroy the DMA buffer management structure */
		plx_free (iface->dma);
	}
	iface->users--;
	up (&card->users_sem);
	return 0;
}

/**
 * masterplx_txdqbuf - transmitter dqbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Dequeue a transmitter buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
masterplx_txdqbuf (struct file *filp, unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct plx_dma *dma = iface->dma;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	if ((filp->f_flags & O_NONBLOCK) && plx_tx_isfull (dma)) {
		up (&iface->buf_sem);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !plx_tx_isfull (dma))) {
		up (&iface->buf_sem);
		return -ERESTARTSYS;
	}
	if (plx_txdqbuf (dma, arg) < 0) {
		up (&iface->buf_sem);
		return -EINVAL;
	}
	up (&iface->buf_sem);
	return 0;
}

/**
 * masterplx_txqbuf - transmitter qbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Queue a transmitter buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
masterplx_txqbuf (struct file *filp, unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct plx_dma *dma = iface->dma;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	if (plx_txqbuf (dma, arg) < 0) {
		up (&iface->buf_sem);
		return -EINVAL;
	}

	/* If DMA is stopped and the buffer queue is half full,
	 * enable and start DMA */
	if (test_bit (0, &iface->dma_done) &&
		plx_tx_buflevel (dma) >= iface->buffers / 2) {
		struct master_dev *card = iface->card;

		writel (plx_head_desc_bus_addr (dma) |
			PLX_DMADPR_DLOC_PCI,
			card->bridge_addr + PLX_DMADPR0);
		clear_bit (0, &iface->dma_done);
		wmb ();
		writeb (PLX_DMACSR_ENABLE | PLX_DMACSR_START,
			card->bridge_addr + PLX_DMACSR0);
		/* Dummy read to flush PCI posted writes */
		readb (card->bridge_addr + PLX_DMACSR0);
	}

	up (&iface->buf_sem);
	return 0;
}

/**
 * masterplx_rxdqbuf - receiver dqbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Dequeue a receiver buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
masterplx_rxdqbuf (struct file *filp, unsigned long arg)
{
	struct master_iface *iface = filp->private_data;
	struct plx_dma *dma = iface->dma;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	if ((filp->f_flags & O_NONBLOCK) && plx_rx_isempty (dma)) {
		up (&iface->buf_sem);
		return -EAGAIN;
	}
	if (wait_event_interruptible (
		iface->queue, !plx_rx_isempty (dma))) {
		up (&iface->buf_sem);
		return -ERESTARTSYS;
	}
	if (plx_rxdqbuf (dma, arg) < 0) {
		up (&iface->buf_sem);
		return -EINVAL;
	}
	up (&iface->buf_sem);
	return 0;
}

/**
 * masterplx_rxqbuf - receiver qbuf ioctl handler
 * @filp: file
 * @arg: ioctl argument
 *
 * Queue a receiver buffer.
 * Returns a negative error code on failure and 0 on success.
 **/
int
masterplx_rxqbuf (struct file *filp, unsigned long arg)
{
	struct master_iface *iface = filp->private_data;

	if (down_interruptible (&iface->buf_sem)) {
		return -ERESTARTSYS;
	}
	if (plx_rxqbuf (iface->dma, arg) < 0) {
		up (&iface->buf_sem);
		return -EINVAL;
	}
	up (&iface->buf_sem);
	return 0;
}

