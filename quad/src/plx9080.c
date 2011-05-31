/* plx9080.c
 *
 * DMA linked-list buffer management for the PLX PCI 9080.
 *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2010 Linear Systems Ltd.
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
#include <linux/slab.h> /* kmalloc () */
#include <linux/dma-mapping.h> /* dma_map_page () */
#include <linux/dmapool.h> /* dma_pool_create () */
#include <linux/errno.h> /* error codes */
#include <linux/delay.h> /* msleep () */

#include <asm/bitops.h> /* clear_bit () */
#include <asm/uaccess.h> /* copy_from_user () */

#include "plx9080.h"
#include "mdma.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define dma_mapping_error(dev,addr) dma_mapping_error(addr)
#endif

/* Static function prototypes */
static int plx_alloc_descriptors (struct master_dma *dma);
static void plx_free_descriptors (struct master_dma *dma);
static int plx_map_buffers (struct master_dma *dma, u32 data_addr);
static void plx_unmap_buffers (struct master_dma *dma);
static ssize_t plx_read (struct master_dma *dma,
	char __user *data,
	size_t length);
static ssize_t plx_write (struct master_dma *dma,
	const char __user *data,
	size_t length);
static ssize_t plx_txdqbuf (struct master_dma *dma, size_t bufnum);
static ssize_t plx_txqbuf (struct master_dma *dma, size_t bufnum);
static ssize_t plx_rxdqbuf (struct master_dma *dma, size_t bufnum);
static ssize_t plx_rxqbuf (struct master_dma *dma, size_t bufnum);

struct master_dma_operations plx_dma_ops = {
	.alloc_descriptors = plx_alloc_descriptors,
	.free_descriptors = plx_free_descriptors,
	.map_buffers = plx_map_buffers,
	.unmap_buffers = plx_unmap_buffers,
	.read = plx_read,
	.write = plx_write,
	.txdqbuf = plx_txdqbuf,
	.txqbuf = plx_txqbuf,
	.rxdqbuf = plx_rxdqbuf,
	.rxqbuf = plx_rxqbuf
};

/**
 * plx_reset_bridge - reset the PCI 9080
 * @addr: mapped address of the bridge
 **/
void
plx_reset_bridge (void __iomem *addr)
{
	unsigned int cntrl;

	/* Set the PCI Read Command to Memory Read Multiple */
	/* and pulse the PCI 9080 software reset bit */
	cntrl = (readl (addr + PLX_CNTRL) &
		~(PLX_CNTRL_PCIRCCDMA_MASK | PLX_CNTRL_PCIMRCCDM_MASK)) | 0xc0c;
	writel (cntrl | PLX_CNTRL_RESET, addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (addr + PLX_CNTRL);
	udelay (100L);
	writel (cntrl, addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (addr + PLX_CNTRL);
	udelay (100L);

	/* Reload the PCI 9080 local configuration registers from the EEPROM */
	writel (cntrl | PLX_CNTRL_RECONFIG, addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (addr + PLX_CNTRL);
	/* Sleep for at least 6 ms */
	msleep (6);
	writel (cntrl, addr + PLX_CNTRL);
	/* Dummy read to flush PCI posted writes */
	readl (addr + PLX_CNTRL);
}

/**
 * plx_alloc_descriptors - allocate DMA descriptors
 * @dma: DMA information structure
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
plx_alloc_descriptors (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	dma_addr_t dma_addr, first_dma_addr;
	struct plx_desc *desc;
	unsigned int i;

	/* Allocate an array of pointers to descriptors */
	if ((dma->desc = (void **)kmalloc (
		total_pointers * sizeof (*dma->desc),
		GFP_KERNEL)) == NULL) {
		goto NO_DESC_PTR;
	}

	/* Allocate the DMA descriptors */
	if ((dma->desc_pool = dma_pool_create ("plx",
		dma->dev,
		sizeof (struct plx_desc),
		sizeof (struct plx_desc),
		0)) == NULL) {
		goto NO_PCI_POOL;
	}
	if ((desc = dma->desc[0] = dma_pool_alloc (dma->desc_pool,
		GFP_KERNEL, &first_dma_addr)) == NULL) {
		goto NO_DESC;
	}
	for (i = 1; i < total_pointers; i++) {
		if ((dma->desc[i] = dma_pool_alloc (dma->desc_pool,
			GFP_KERNEL,
			&dma_addr)) == NULL) {
			unsigned int j;

			for (j = i - 1; j > 0; j--) {
				desc = dma->desc[j - 1];
				dma_pool_free (dma->desc_pool,
					dma->desc[j],
					desc->next_desc);
			}
			dma_pool_free (dma->desc_pool,
				dma->desc[0],
				first_dma_addr);
			goto NO_DESC;
		}
		desc->next_desc = dma_addr;
		desc = dma->desc[i];
	}
	desc->next_desc = first_dma_addr;

	return 0;

NO_DESC:
	dma_pool_destroy (dma->desc_pool);
NO_PCI_POOL:
	kfree (dma->desc);
NO_DESC_PTR:
	return -ENOMEM;
}

/**
 * plx_free_descriptors - free DMA descriptors
 * @dma: DMA information structure
 **/
static void
plx_free_descriptors (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct plx_desc *prev_desc = dma->desc[total_pointers - 1];
	dma_addr_t dma_addr = prev_desc->next_desc;
	unsigned int i;

	for (i = total_pointers - 1; i > 0; i--) {
		prev_desc = dma->desc[i - 1];
		dma_pool_free (dma->desc_pool,
			dma->desc[i],
			prev_desc->next_desc);
	}
	dma_pool_free (dma->desc_pool,
		dma->desc[0], dma_addr);
	dma_pool_destroy (dma->desc_pool);
	kfree (dma->desc);
	return;
}

/**
 * plx_map_buffers - map DMA buffers
 * @dma: DMA information structure
 * @data_addr: local bus address of the data register
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
plx_map_buffers (struct master_dma *dma, u32 data_addr)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int last_block_size = dma->bufsize -
		((dma->pointers_per_buf - 1) * PAGE_SIZE);
	struct plx_desc *desc;
	unsigned int i;

	for (i = 0; i < total_pointers; i++) {
		desc = dma->desc[i];
		desc->local_addr = data_addr;
		if ((i % dma->pointers_per_buf) ==
			(dma->pointers_per_buf - 1)) {
			/* This is the last descriptor for this buffer */
			desc->bytes = last_block_size;
			desc->next_desc |= PLX_DMADPR_DLOC_PCI |
				((dma->direction == DMA_TO_DEVICE) ?
				PLX_DMADPR_EOC : PLX_DMADPR_LB2PCI) |
				PLX_DMADPR_INT;
		} else {
			desc->bytes = PAGE_SIZE;
			desc->next_desc |= PLX_DMADPR_DLOC_PCI |
				((dma->direction == DMA_TO_DEVICE) ?
				0 : PLX_DMADPR_LB2PCI);
		}
		desc->pci_addr = dma_map_page (dma->dev,
			virt_to_page (dma->vpage[i]),
			offset_in_page (dma->vpage[i]),
			desc->bytes,
			dma->direction);
		if (dma_mapping_error (dma->dev, desc->pci_addr)) {
			unsigned int j;

			for (j = 0; j < i; j++) {
				desc = dma->desc[j];
				dma_unmap_page (dma->dev,
					desc->pci_addr,
					desc->bytes,
					dma->direction);
			}
			return -ENOMEM;
		}
	}
	return 0;
}

/**
 * plx_unmap_buffers - unmap DMA buffers
 * @dma: DMA information structure
 **/
static void
plx_unmap_buffers (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct plx_desc *desc;
	unsigned int i;

	for (i = 0; i < total_pointers; i++) {
		desc = dma->desc[i];
		dma_unmap_page (dma->dev,
			desc->pci_addr,
			desc->bytes,
			dma->direction);
	}
	return;
}

/**
 * plx_read - copy data from driver buffer to user buffer
 * @dma: DMA information structure
 * @data: user buffer
 * @length: size of data
 *
 * Returns a negative error code on failure and
 * the number of bytes copied on success.
 **/
static ssize_t
plx_read (struct master_dma *dma, char __user *data, size_t length)
{
	/* Amount of data remaining in the current buffer */
	const size_t max_length = dma->bufsize - dma->cpu_offset;
	size_t chunkvpage, chunkoffset, chunksize;
	size_t copied = 0; /* Number of bytes copied */
	struct plx_desc *desc;

	/* Copy the rest of this buffer or the requested amount,
	 * whichever is less */
	if (length > max_length) {
		length = max_length;
	}
	while (length > 0) {
		chunkvpage = dma->cpu_buffer * dma->pointers_per_buf +
			dma->cpu_offset / PAGE_SIZE;
		chunkoffset = dma->cpu_offset % PAGE_SIZE;
		desc = dma->desc[chunkvpage];
		chunksize = desc->bytes - chunkoffset;
		if (chunksize > length) {
			chunksize = length;
		}
		dma_sync_single_for_cpu (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_FROM_DEVICE);
		if (copy_to_user (data + copied,
			dma->vpage[chunkvpage] + chunkoffset,
			chunksize)) {
			dma_sync_single_for_device (dma->dev,
				desc->pci_addr,
				desc->bytes,
				DMA_FROM_DEVICE);
			return -EFAULT;
		}
		dma_sync_single_for_device (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_FROM_DEVICE);
		dma->cpu_offset += chunksize;
		copied += chunksize;
		length -= chunksize;
	}

	if (copied == max_length) {
		dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
		dma->cpu_offset = 0;
	}
	return copied;
}

/**
 * plx_write - copy data from user buffer to driver buffer
 * @dma: DMA information structure
 * @data: user buffer
 * @length: size of data
 *
 * Returns a negative error code on failure and
 * the number of bytes copied on success.
 **/
static ssize_t
plx_write (struct master_dma *dma, const char __user *data, size_t length)
{
	/* Amount of free space remaining in the current buffer */
	const size_t max_length = dma->bufsize - dma->cpu_offset;
	size_t chunkvpage = 0, chunkoffset, max_chunk, chunksize;
	size_t copied = 0; /* Number of bytes copied */
	struct plx_desc *desc;

	/* Copy the rest of this buffer or the requested amount,
	 * whichever is less */
	if (length > max_length) {
		length = max_length;
	}
	while (length > 0) {
		chunkvpage = dma->cpu_buffer * dma->pointers_per_buf +
			dma->cpu_offset / PAGE_SIZE;
		chunkoffset = dma->cpu_offset % PAGE_SIZE;
		desc = dma->desc[chunkvpage];
		chunksize = max_chunk = desc->bytes - chunkoffset;
		if (chunksize > length) {
			chunksize = length;
		}
		dma_sync_single_for_cpu (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_TO_DEVICE);
		if (copy_from_user (
			dma->vpage[chunkvpage] + chunkoffset,
			data + copied, chunksize)) {
			dma_sync_single_for_device (dma->dev,
				desc->pci_addr,
				desc->bytes,
				DMA_TO_DEVICE);
			return -EFAULT;
		}
		dma_sync_single_for_device (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_TO_DEVICE);
		dma->cpu_offset += chunksize;
		copied += chunksize;
		length -= chunksize;
	}

	/* If we've filled a buffer,
	 * link the descriptors for the buffer into the DMA chain */
	if (copied == max_length) {
		const unsigned int prevbuffer =
			(dma->cpu_buffer + dma->buffers - 1) %
			dma->buffers;
		const unsigned int oldlastvpage =
			prevbuffer * dma->pointers_per_buf +
			dma->pointers_per_buf - 1;

		desc = dma->desc[chunkvpage];
		desc->next_desc |= (1 << PLX_DMADPR_EOC_ORDER);
		dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
		dma->cpu_offset = 0;
		desc = dma->desc[oldlastvpage];
		clear_bit (PLX_DMADPR_EOC_ORDER,
			(unsigned long *)&desc->next_desc);
	}
	return copied;
}

/**
 * plx_head_desc_bus_addr - return the bus address of the head descriptor
 * @dma: DMA information structure
 **/
u32
plx_head_desc_bus_addr (struct master_dma *dma)
{
	const unsigned int prevbuffer =
		(dma->dev_buffer + dma->buffers - 1) % dma->buffers;
	const unsigned int oldlastvpage =
		prevbuffer * dma->pointers_per_buf +
		dma->pointers_per_buf - 1;
	struct plx_desc *desc = dma->desc[oldlastvpage];

	return (desc->next_desc & ~0x0000000f);
}

/**
 * plx_tx_link_all - flush the transmit buffers
 * @dma: DMA information structure
 *
 * If there are any complete dwords in the transmit buffer
 * currently being written to, add them to the end of the
 * DMA chain.
 **/
void
plx_tx_link_all (struct master_dma *dma)
{
	const unsigned int ptr = dma->cpu_buffer * dma->pointers_per_buf +
		dma->cpu_offset / PAGE_SIZE;
	const unsigned int actual_bytes =
		(dma->cpu_offset % PAGE_SIZE) & ~0x3;
	const unsigned int prevbuffer =
		(dma->cpu_buffer + dma->buffers - 1) %
		dma->buffers;
	const unsigned int oldlastptr = prevbuffer * dma->pointers_per_buf +
		dma->pointers_per_buf - 1;
	struct plx_desc *desc = dma->desc[ptr];
	struct plx_desc *oldlastdesc = dma->desc[oldlastptr];

	desc->bytes = actual_bytes;
	desc->next_desc |= PLX_DMADPR_EOC | PLX_DMADPR_INT;
	clear_bit (PLX_DMADPR_EOC_ORDER,
		(unsigned long *)&oldlastdesc->next_desc);
	clear_bit (PLX_DMADPR_INT_ORDER,
		(unsigned long *)&oldlastdesc->next_desc);
	return;
}

/**
 * plx_reset - reset the descriptor chain
 * @dma: DMA information structure
 **/
void
plx_reset (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int last_block_size = dma->bufsize -
		((dma->pointers_per_buf - 1) * PAGE_SIZE);
	unsigned int i;
	struct plx_desc *desc;

	for (i = 0; i < total_pointers; i++) {
		desc = dma->desc[i];
		if ((i % dma->pointers_per_buf) ==
			(dma->pointers_per_buf - 1)) {
			/* This is the last descriptor for this buffer */
			desc->bytes = last_block_size;
			desc->next_desc |= PLX_DMADPR_INT;
			if (dma->direction == DMA_TO_DEVICE) {
				desc->next_desc |= PLX_DMADPR_EOC;
			} else {
				desc->next_desc &= ~PLX_DMADPR_EOC;
			}
		} else {
			desc->bytes = PAGE_SIZE;
			desc->next_desc &= ~PLX_DMADPR_EOC & ~PLX_DMADPR_INT;
		}
	}
	dma->cpu_buffer = 0;
	dma->cpu_offset = 0;
	dma->dev_buffer = 0;
	return;
}

/**
 * plx_txdqbuf - Dequeue a write buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally precedes a copy from a user buffer
 * to a driver buffer.
 **/
static ssize_t
plx_txdqbuf (struct master_dma *dma, size_t bufnum)
{
	unsigned int i;
	struct plx_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		desc = dma->desc[dma->cpu_buffer * dma->pointers_per_buf + i];
		dma_sync_single_for_cpu (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_TO_DEVICE);
	}
	return dma->bufsize;
}

/**
 * plx_txqbuf - Queue a write buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally follows a copy from a user buffer
 * to a driver buffer.
 **/
static ssize_t
plx_txqbuf (struct master_dma *dma, size_t bufnum)
{
	const unsigned int prevbuffer =
		(dma->cpu_buffer + dma->buffers - 1) %
		dma->buffers;
	const unsigned int oldlastvpage =
		prevbuffer * dma->pointers_per_buf +
		dma->pointers_per_buf - 1;
	unsigned int i, vpage = 0;
	struct plx_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		vpage = dma->cpu_buffer * dma->pointers_per_buf + i;
		desc = dma->desc[vpage];
		dma_sync_single_for_device (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_TO_DEVICE);
	}
	desc = dma->desc[vpage];
	desc->next_desc |= (1 << PLX_DMADPR_EOC_ORDER);
	dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
	dma->cpu_offset = 0;
	desc = dma->desc[oldlastvpage];
	clear_bit (PLX_DMADPR_EOC_ORDER,
		(unsigned long *)&desc->next_desc);
	return dma->bufsize;
}

/**
 * plx_rxdqbuf - Dequeue a read buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally precedes a copy from a driver buffer
 * to a user buffer.
 **/
static ssize_t
plx_rxdqbuf (struct master_dma *dma, size_t bufnum)
{
	unsigned int i;
	struct plx_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		desc = dma->desc[dma->cpu_buffer * dma->pointers_per_buf + i];
		dma_sync_single_for_cpu (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_FROM_DEVICE);
	}
	return dma->bufsize;
}

/**
 * plx_rxqbuf - Queue a read buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally follows a copy from a driver buffer
 * to a user buffer.
 **/
static ssize_t
plx_rxqbuf (struct master_dma *dma, size_t bufnum)
{
	unsigned int i;
	struct plx_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		desc = dma->desc[dma->cpu_buffer * dma->pointers_per_buf + i];
		dma_sync_single_for_device (dma->dev,
			desc->pci_addr,
			desc->bytes,
			DMA_FROM_DEVICE);
	}
	dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
	dma->cpu_offset = 0;
	return dma->bufsize;
}

