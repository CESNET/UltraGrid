/* mdma.h
 *
 * Header file for mdma.c.
 *
 * Copyright (C) 2007-2009 Linear Systems Ltd.
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

#ifndef _MDMA_H
#define _MDMA_H

#include <linux/types.h> /* size_t */
#include <linux/device.h> /* device */
#include <linux/spinlock.h> /* spinlock_t */
#include <linux/mm.h> /* vm_operations_struct */

#define MDMA_MMAP 0x00000001

struct master_dma;

/**
 * master_dma_operations - structure of pointers to DMA helper functions
 * @alloc_descriptors: descriptor array allocation function
 * @free_descriptors: descriptor array deallocation function
 * @map_buffers: buffer mapping function
 * @unmap_buffers: buffer unmapping function
 * @read: buffer reading function
 * @write: buffer writing function
 * @txdqbuf: transmit buffer dequeueing function
 * @txqbuf: transmit buffer queueing function
 * @rxdqbuf: receive buffer dequeueing function
 * @rxqbuf: receive buffer queueing function
 **/
struct master_dma_operations {
	int (*alloc_descriptors) (struct master_dma *dma);
	void (*free_descriptors) (struct master_dma *dma);
	int (*map_buffers) (struct master_dma *dma, u32 data_addr);
	void (*unmap_buffers) (struct master_dma *dma);
	ssize_t (*read) (struct master_dma *dma,
		char __user *data,
		size_t length);
	ssize_t (*write) (struct master_dma *dma,
		const char __user *data,
		size_t length);
	ssize_t (*txdqbuf) (struct master_dma *dma, size_t bufnum);
	ssize_t (*txqbuf) (struct master_dma *dma, size_t bufnum);
	ssize_t (*rxdqbuf) (struct master_dma *dma, size_t bufnum);
	ssize_t (*rxqbuf) (struct master_dma *dma, size_t bufnum);
};

/**
 * master_dma - DMA information structure
 * @dev: device
 * @ops: pointer to a structure of pointers to DMA helper functions
 * @buffers: number of buffers
 * @bufsize: number of bytes in each buffer
 * @pointers_per_buf: number of descriptors per buffer
 * @direction: direction of data flow
 * @desc_pool: DMA-coherent memory pool
 * @desc: pointer to an array of pointers to DMA descriptors
 * @page: pointer to an array of pointers to memory pages
 * @vpage: pointer to an array of pointers to DMA buffer fragments
 * @lock: spinlock to protect dev_buffer
 * @dev_buffer: buffer being accessed by the device
 * @cpu_buffer: buffer being accessed by the CPU
 * @cpu_offset: offset of the CPU access point within cpu_buffer
 * @flags: allocation flags
 **/
struct master_dma {
	struct device *dev;
	struct master_dma_operations *ops;
	unsigned int buffers;
	unsigned int bufsize;
	unsigned int pointers_per_buf;
	unsigned int direction;
	struct dma_pool *desc_pool;
	void **desc;
	unsigned long *page;
	unsigned char **vpage;
	spinlock_t lock;
	size_t dev_buffer;
	size_t cpu_buffer;
	size_t cpu_offset;
	unsigned int flags;
};

/* External variables */

extern struct vm_operations_struct mdma_vm_ops;

/* External function prototypes */

struct master_dma *mdma_alloc (struct device *dev,
	struct master_dma_operations *ops,
	u32 data_addr,
	unsigned int buffers,
	unsigned int bufsize,
	unsigned int direction,
	unsigned int flags);
void mdma_free (struct master_dma *dma);

/* Inline functions */

/**
 * mdma_desc_to_dma - return a dma address from a descriptor address
 * @desc_low: low 32 bits of descriptor pointer
 * @desc_high: high 32 bits of descriptor pointer
 *
 * Handles 32 and 64 bit addresses.
 **/
static inline dma_addr_t
mdma_desc_to_dma (u32 desc_low, u32 desc_high)
{
	if (sizeof (dma_addr_t) == 4) {
		return (dma_addr_t)desc_low;
	} else {
		return (dma_addr_t)((((u64)desc_high) << 32) | desc_low);
	}
}

/**
 * mdma_dma_to_desc_low - return the low 32 bits of a dma address
 * @dma_addr: dma address
 *
 * Handles 32 and 64 bit addresses.
 **/
static inline u32
mdma_dma_to_desc_low (dma_addr_t dma_addr)
{
	return (u32)(dma_addr & 0xffffffff);
}

/**
 * mdma_dma_to_desc_high - return the high 32 bits of a dma address
 * @dma_addr: dma address
 *
 * Handles 32 and 64 bit addresses.
 **/
static inline u32
mdma_dma_to_desc_high (dma_addr_t dma_addr)
{
	return (u32)((((u64)dma_addr) >> 32) & 0xffffffff);
}

/**
 * mdma_tx_buflevel - return the number of transmit buffers in use
 * @dma: DMA information structure
 **/
static inline int
mdma_tx_buflevel (struct master_dma *dma)
{
	int ret;

	spin_lock_irq (&dma->lock);
	ret = ((dma->cpu_buffer + dma->buffers - dma->dev_buffer) %
		dma->buffers);
	spin_unlock_irq (&dma->lock);
	return ret;
}

/**
 * mdma_tx_isfull - return true if the transmit buffers are full
 * @dma: DMA information structure
 **/
static inline int
mdma_tx_isfull (struct master_dma *dma)
{
	int ret;

	spin_lock_irq (&dma->lock);
	ret = (dma->dev_buffer == ((dma->cpu_buffer + 1) % dma->buffers));
	spin_unlock_irq (&dma->lock);
	return ret;
}

/**
 * mdma_rx_buflevel - return the number of receive buffers in use
 * @dma: DMA information structure
 **/
static inline int
mdma_rx_buflevel (struct master_dma *dma)
{
	int ret;

	spin_lock_irq (&dma->lock);
	ret = ((dma->dev_buffer + dma->buffers - dma->cpu_buffer) %
		dma->buffers);
	spin_unlock_irq (&dma->lock);
	return ret;
}

/**
 * mdma_rx_isempty - return true if the receive buffers are empty
 * @dma: DMA information structure
 **/
static inline int
mdma_rx_isempty (struct master_dma *dma)
{
	int ret;

	spin_lock_irq (&dma->lock);
	ret = (dma->cpu_buffer == dma->dev_buffer);
	spin_unlock_irq (&dma->lock);
	return ret;
}

/**
 * mdma_advance - increment the device buffer pointer
 * @dma: DMA information structure
 **/
static inline void
mdma_advance (struct master_dma *dma)
{
	spin_lock_irq (&dma->lock);
	dma->dev_buffer = (dma->dev_buffer + 1) % dma->buffers;
	spin_unlock_irq (&dma->lock);
	return;
}

#endif

