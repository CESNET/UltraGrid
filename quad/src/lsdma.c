/* lsdma.c
 *
 * DMA linked-list buffer management for the Linear Systems DMA controller.
 *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2009 Linear Systems Ltd.
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

#include <asm/bitops.h> /* clear_bit () */
#include <asm/uaccess.h> /* copy_from_user () */

#include "lsdma.h"
#include "mdma.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define dma_mapping_error(dev,addr) dma_mapping_error(addr)
#endif

/* Static function prototypes */
static int lsdma_alloc_descriptors (struct master_dma *dma);
static void lsdma_free_descriptors (struct master_dma *dma);
static int lsdma_map_buffers (struct master_dma *dma, u32 data_addr);
static void lsdma_unmap_buffers (struct master_dma *dma);
static ssize_t lsdma_read (struct master_dma *dma,
	char __user *data,
	size_t length);
static ssize_t lsdma_write (struct master_dma *dma,
	const char __user *data,
	size_t length);
static ssize_t lsdma_txdqbuf (struct master_dma *dma, size_t bufnum);
static ssize_t lsdma_txqbuf (struct master_dma *dma, size_t bufnum);
static ssize_t lsdma_rxdqbuf (struct master_dma *dma, size_t bufnum);
static ssize_t lsdma_rxqbuf (struct master_dma *dma, size_t bufnum);

struct master_dma_operations lsdma_dma_ops = {
	.alloc_descriptors = lsdma_alloc_descriptors,
	.free_descriptors = lsdma_free_descriptors,
	.map_buffers = lsdma_map_buffers,
	.unmap_buffers = lsdma_unmap_buffers,
	.read = lsdma_read,
	.write = lsdma_write,
	.txdqbuf = lsdma_txdqbuf,
	.txqbuf = lsdma_txqbuf,
	.rxdqbuf = lsdma_rxdqbuf,
	.rxqbuf = lsdma_rxqbuf
};

/**
 * lsdma_alloc_descriptors - allocate DMA descriptors
 * @dma: DMA information structure
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsdma_alloc_descriptors (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	dma_addr_t dma_addr, first_dma_addr;
	struct lsdma_desc *desc;
	unsigned int i;

	/* Allocate an array of pointers to descriptors */
	if ((dma->desc = (void **)kmalloc (
		total_pointers * sizeof (*dma->desc),
		GFP_KERNEL)) == NULL) {
		goto NO_DESC_PTR;
	}

	/* Allocate the DMA descriptors */
	if ((dma->desc_pool = dma_pool_create ("lsdma",
		dma->dev,
		sizeof (struct lsdma_desc),
		32,
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
				dma_addr = mdma_desc_to_dma (desc->next_desc,
					desc->next_desc_h);
				dma_pool_free (dma->desc_pool,
					dma->desc[j],
					dma_addr);
			}
			dma_pool_free (dma->desc_pool,
				dma->desc[0],
				first_dma_addr);
			goto NO_DESC;
		}

		desc->next_desc = mdma_dma_to_desc_low (dma_addr);
		desc->next_desc_h = mdma_dma_to_desc_high (dma_addr);

		desc = dma->desc[i];
	}

	desc->next_desc = mdma_dma_to_desc_low (first_dma_addr);
	desc->next_desc_h = mdma_dma_to_desc_high (first_dma_addr);

	return 0;

NO_DESC:
	dma_pool_destroy (dma->desc_pool);
NO_PCI_POOL:
	kfree (dma->desc);
NO_DESC_PTR:
	return -ENOMEM;
}

/**
 * lsdma_free_descriptors - free DMA descriptors
 * @dma: DMA information structure
 **/
static void
lsdma_free_descriptors (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct lsdma_desc *prev_desc = dma->desc[total_pointers - 1];
	dma_addr_t last_dma_addr = mdma_desc_to_dma (prev_desc->next_desc,
		prev_desc->next_desc_h);
	unsigned int i;

	for (i = total_pointers - 1; i > 0; i--) {
		prev_desc = dma->desc[i - 1];
		dma_pool_free (dma->desc_pool,
			dma->desc[i],
			mdma_desc_to_dma (prev_desc->next_desc,
				prev_desc->next_desc_h));
	}
	dma_pool_free (dma->desc_pool,
		dma->desc[0], last_dma_addr);
	dma_pool_destroy (dma->desc_pool);
	kfree (dma->desc);
	return;
}

/**
 * lsdma_map_buffers - map DMA buffers
 * @dma: DMA information structure
 * @data_addr: local bus address of the data register
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
lsdma_map_buffers (struct master_dma *dma, u32 data_addr)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int last_block_size = dma->bufsize -
		((dma->pointers_per_buf - 1) * PAGE_SIZE);
	struct lsdma_desc *desc;
	dma_addr_t dma_addr;
	unsigned int i;

	if (dma->direction == DMA_TO_DEVICE) {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			desc->dest_addr = data_addr;
			if ((i % dma->pointers_per_buf) ==
				(dma->pointers_per_buf - 1)) {
				/* This is the last descriptor for this buffer */
				desc->csr = LSDMA_DESC_CSR_INT |
					LSDMA_DESC_CSR_EOL |
					last_block_size;
			} else {
				desc->csr = PAGE_SIZE;
			}
			dma_addr = dma_map_page (dma->dev,
				virt_to_page (dma->vpage[i]),
				offset_in_page (dma->vpage[i]),
				(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
				dma->direction);
			if (dma_mapping_error (dma->dev, dma_addr)) {
				unsigned int j;

				for (j = 0; j < i; j++) {
					desc = dma->desc[j];
					dma_unmap_page (dma->dev,
						mdma_desc_to_dma (desc->src_addr,
							desc->src_addr_h),
						(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
						dma->direction);
				}
				return -ENOMEM;
			}
			desc->src_addr = mdma_dma_to_desc_low (dma_addr);
			desc->src_addr_h = mdma_dma_to_desc_high (dma_addr);
		}
	} else {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			desc->src_addr = data_addr;
			if ((i % dma->pointers_per_buf) ==
				(dma->pointers_per_buf - 1)) {
				/* This is the last descriptor for this buffer */
				desc->csr = LSDMA_DESC_CSR_INT |
					LSDMA_DESC_CSR_DIRECTION |
					last_block_size;
			} else {
				desc->csr = LSDMA_DESC_CSR_DIRECTION |
					PAGE_SIZE;
			}
			dma_addr = dma_map_page (dma->dev,
				virt_to_page (dma->vpage[i]),
				offset_in_page (dma->vpage[i]),
				(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
				dma->direction);
			if (dma_mapping_error (dma->dev, dma_addr)) {
				unsigned int j;

				for (j = 0; j < i; j++) {
					desc = dma->desc[j];
					dma_unmap_page (dma->dev,
						mdma_desc_to_dma (desc->dest_addr,
							desc->dest_addr_h),
						(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
						dma->direction);
				}
				return -ENOMEM;
			}
			desc->dest_addr = mdma_dma_to_desc_low (dma_addr);
			desc->dest_addr_h = mdma_dma_to_desc_high (dma_addr);
		}
	}
	return 0;
}

/**
 * lsdma_unmap_buffers - unmap DMA buffers
 * @dma: DMA information structure
 **/
static void
lsdma_unmap_buffers (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct lsdma_desc *desc;
	dma_addr_t dma_addr;
	unsigned int i;

	for (i = 0; i < total_pointers; i++) {
		desc = dma->desc[i];
		if (dma->direction == DMA_TO_DEVICE) {
			dma_addr = mdma_desc_to_dma (desc->src_addr,
				desc->src_addr_h);
		} else {
			dma_addr = mdma_desc_to_dma (desc->dest_addr,
				desc->dest_addr_h);
		}
		dma_unmap_page (dma->dev,
			dma_addr,
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			dma->direction);
	}
	return;
}

/**
 * lsdma_read - copy data from driver buffer to user buffer
 * @dma: DMA information structure
 * @data: user buffer
 * @length: size of data
 *
 * Returns a negative error code on failure and
 * the number of bytes copied on success.
 **/
static ssize_t
lsdma_read (struct master_dma *dma, char __user *data, size_t length)
{
	/* Amount of data remaining in the current buffer */
	const size_t max_length = dma->bufsize - dma->cpu_offset;
	size_t chunkvpage, chunkoffset, chunksize;
	size_t copied = 0; /* Number of bytes copied */
	struct lsdma_desc *desc;

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
		chunksize = (desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE) - chunkoffset;
		if (chunksize > length) {
			chunksize = length;
		}
		dma_sync_single_for_cpu (dma->dev,
			mdma_desc_to_dma (desc->dest_addr, desc->dest_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			DMA_FROM_DEVICE);
		if (copy_to_user (data + copied,
			dma->vpage[chunkvpage] + chunkoffset,
			chunksize)) {
			dma_sync_single_for_device (dma->dev,
				mdma_desc_to_dma (desc->dest_addr,
					desc->dest_addr_h),
				(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
				DMA_FROM_DEVICE);
			return -EFAULT;
		}
		dma_sync_single_for_device (dma->dev,
			mdma_desc_to_dma (desc->dest_addr, desc->dest_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
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
 * lsdma_write - copy data from user buffer to driver buffer
 * @dma: DMA information structure
 * @data: user buffer
 * @length: size of data
 *
 * Returns a negative error code on failure and
 * the number of bytes copied on success.
 **/
static ssize_t
lsdma_write (struct master_dma *dma, const char __user *data, size_t length)
{
	/* Amount of free space remaining in the current buffer */
	const size_t max_length = dma->bufsize - dma->cpu_offset;
	size_t chunkvpage = 0, chunkoffset, max_chunk, chunksize;
	size_t copied = 0; /* Number of bytes copied */
	struct lsdma_desc *desc;

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
		chunksize = max_chunk = (desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE) - chunkoffset;
		if (chunksize > length) {
			chunksize = length;
		}
		dma_sync_single_for_cpu (dma->dev,
			mdma_desc_to_dma (desc->src_addr, desc->src_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			DMA_TO_DEVICE);
		if (copy_from_user (
			dma->vpage[chunkvpage] + chunkoffset,
			data + copied, chunksize)) {
			dma_sync_single_for_device (dma->dev,
				mdma_desc_to_dma (desc->src_addr, desc->src_addr_h),
				(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
				DMA_TO_DEVICE);
			return -EFAULT;
		}
		dma_sync_single_for_device (dma->dev,
			mdma_desc_to_dma (desc->src_addr, desc->src_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
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
		desc->csr |= LSDMA_DESC_CSR_EOL;
		dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
		dma->cpu_offset = 0;
		desc = dma->desc[oldlastvpage];
		clear_bit (LSDMA_DESC_CSR_EOL_ORDER,
			(unsigned long *)&desc->csr);
	}
	return copied;
}

/**
 * lsdma_head_desc_bus_addr - return the bus address of the head descriptor
 * @dma: DMA information structure
 **/
dma_addr_t
lsdma_head_desc_bus_addr (struct master_dma *dma)
{
	const unsigned int prevbuffer =
		(dma->dev_buffer + dma->buffers - 1) % dma->buffers;
	const unsigned int oldlastvpage =
		prevbuffer * dma->pointers_per_buf +
		dma->pointers_per_buf - 1;
	struct lsdma_desc *desc = dma->desc[oldlastvpage];

	return (mdma_desc_to_dma (desc->next_desc, desc->next_desc_h));
}

/**
 * lsdma_tx_link_all - flush the transmit buffers
 * @dma: DMA information structure
 *
 * If there are any complete dwords in the transmit buffer
 * currently being written to, add them to the end of the
 * DMA chain.
 **/
void
lsdma_tx_link_all (struct master_dma *dma)
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
	struct lsdma_desc *desc = dma->desc[ptr];
	struct lsdma_desc *oldlastdesc = dma->desc[oldlastptr];

	desc->csr = actual_bytes |
		LSDMA_DESC_CSR_EOL | LSDMA_DESC_CSR_INT;
	clear_bit (LSDMA_DESC_CSR_EOL_ORDER,
		(unsigned long *)&oldlastdesc->csr);
	clear_bit (LSDMA_DESC_CSR_INT_ORDER,
		(unsigned long *)&oldlastdesc->csr);
	return;
}

/**
 * lsdma_reset - reset the descriptor chain
 * @dma: DMA information structure
 **/
void
lsdma_reset (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int last_block_size = dma->bufsize -
		((dma->pointers_per_buf - 1) * PAGE_SIZE);
	unsigned int i;
	struct lsdma_desc *desc;

	if (dma->direction == DMA_TO_DEVICE) {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			if ((i % dma->pointers_per_buf) ==
				(dma->pointers_per_buf - 1)) {
				/* This is the last descriptor for this buffer */
				desc->csr = LSDMA_DESC_CSR_INT |
					LSDMA_DESC_CSR_EOL |
					last_block_size;
			} else {
				desc->csr = PAGE_SIZE;
			}
		}
	}
	dma->cpu_buffer = 0;
	dma->cpu_offset = 0;
	dma->dev_buffer = 0;
	return;
}

/**
 * lsdma_txdqbuf - Dequeue a write buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally precedes a copy from a user buffer
 * to a driver buffer.
 **/
static ssize_t
lsdma_txdqbuf (struct master_dma *dma, size_t bufnum)
{
	unsigned int i;
	struct lsdma_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		desc = dma->desc[dma->cpu_buffer * dma->pointers_per_buf + i];

		dma_sync_single_for_cpu (dma->dev,
			mdma_desc_to_dma (desc->src_addr, desc->src_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			DMA_TO_DEVICE);
	}
	return dma->bufsize;
}

/**
 * lsdma_txqbuf - Queue a write buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally follows a copy from a user buffer
 * to a driver buffer.
 **/
static ssize_t
lsdma_txqbuf (struct master_dma *dma, size_t bufnum)
{
	const unsigned int prevbuffer =
		(dma->cpu_buffer + dma->buffers - 1) %
		dma->buffers;
	const unsigned int oldlastvpage =
		prevbuffer * dma->pointers_per_buf +
		dma->pointers_per_buf - 1;
	unsigned int i, vpage = 0;
	struct lsdma_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		vpage = dma->cpu_buffer * dma->pointers_per_buf + i;
		desc = dma->desc[vpage];
		dma_sync_single_for_device (dma->dev,
			mdma_desc_to_dma (desc->src_addr, desc->src_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			DMA_TO_DEVICE);
	}
	desc = dma->desc[vpage];
	desc->csr |= LSDMA_DESC_CSR_EOL;
	dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
	dma->cpu_offset = 0;
	desc = dma->desc[oldlastvpage];
	clear_bit (LSDMA_DESC_CSR_EOL_ORDER,
		(unsigned long *)&desc->csr);
	return dma->bufsize;
}

/**
 * lsdma_rxdqbuf - Dequeue a read buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally precedes a copy from a driver buffer
 * to a user buffer.
 **/
static ssize_t
lsdma_rxdqbuf (struct master_dma *dma, size_t bufnum)
{
	unsigned int i;
	struct lsdma_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		desc = dma->desc[dma->cpu_buffer * dma->pointers_per_buf + i];
		dma_sync_single_for_cpu (dma->dev,
			mdma_desc_to_dma (desc->dest_addr, desc->dest_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			DMA_FROM_DEVICE);
	}
	return dma->bufsize;
}

/**
 * lsdma_rxqbuf - Queue a read buffer
 * @dma: DMA information structure
 * @bufnum: buffer number
 *
 * Do everything which normally follows a copy from a driver buffer
 * to a user buffer.
 **/
static ssize_t
lsdma_rxqbuf (struct master_dma *dma, size_t bufnum)
{
	unsigned int i;
	struct lsdma_desc *desc;

	if (bufnum != dma->cpu_buffer) {
		return -EINVAL;
	}
	for (i = 0; i < dma->pointers_per_buf; i++) {
		desc = dma->desc[dma->cpu_buffer * dma->pointers_per_buf + i];
		dma_sync_single_for_device (dma->dev,
			mdma_desc_to_dma (desc->dest_addr, desc->dest_addr_h),
			(desc->csr & LSDMA_DESC_CSR_TOTALXFERSIZE),
			DMA_FROM_DEVICE);
	}
	dma->cpu_buffer = (dma->cpu_buffer + 1) % dma->buffers;
	dma->cpu_offset = 0;
	return dma->bufsize;
}

