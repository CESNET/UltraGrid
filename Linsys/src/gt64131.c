/* gt64131.c
 *
 * DMA linked-list buffer management for the Marvell GT-64131.
 *
 * Copyright (C) 2003-2009 Linear Systems Ltd.
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

#include <asm/uaccess.h> /* copy_to_user () */

#include "gt64131.h"
#include "mdma.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,27))
#define dma_mapping_error(dev,addr) dma_mapping_error(addr)
#endif

/* Static function prototypes */
static int gt64_alloc_descriptors (struct master_dma *dma);
static void gt64_free_descriptors (struct master_dma *dma);
static int gt64_map_buffers (struct master_dma *dma, u32 data_addr);
static void gt64_unmap_buffers (struct master_dma *dma);
static ssize_t gt64_read (struct master_dma *dma,
	char __user *data,
	size_t length);

struct master_dma_operations gt64_dma_ops = {
	.alloc_descriptors = gt64_alloc_descriptors,
	.free_descriptors = gt64_free_descriptors,
	.map_buffers = gt64_map_buffers,
	.unmap_buffers = gt64_unmap_buffers,
	.read = gt64_read
};

/**
 * gt64_alloc_descriptors - allocate DMA descriptors
 * @dma: DMA buffer management structure
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
gt64_alloc_descriptors (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	dma_addr_t dma_addr, first_dma_addr;
	struct gt64_desc *desc;
	unsigned int i;

	/* Allocate an array of pointers to descriptors */
	if ((dma->desc = (void **)kmalloc (
		total_pointers * sizeof (*dma->desc),
		GFP_KERNEL)) == NULL) {
		goto NO_DESC_PTR;
	}

	/* Allocate the DMA descriptors */
	if ((dma->desc_pool = dma_pool_create ("gt64",
		dma->dev,
		sizeof (struct gt64_desc),
		sizeof (struct gt64_desc),
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
 * gt64_free_descriptors - free DMA descriptors
 * @dma: DMA buffer management structure
 **/
static void
gt64_free_descriptors (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct gt64_desc *prev_desc = dma->desc[total_pointers - 1];
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
 * gt64_map_buffers - map DMA buffers
 * @dma: DMA buffer management structure
 * @data_addr: local bus address of the data register
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
gt64_map_buffers (struct master_dma *dma, u32 data_addr)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int last_block_size = dma->bufsize -
		((dma->pointers_per_buf - 1) * PAGE_SIZE);
	struct gt64_desc *desc;
	unsigned int i;

	if (dma->direction == DMA_TO_DEVICE) {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			desc->dest_addr = data_addr;
			if ((i % dma->pointers_per_buf) ==
				(dma->pointers_per_buf - 1)) {
				/* This is the last page in this buffer */
				desc->bytes = last_block_size;
			} else {
				desc->bytes = PAGE_SIZE;
			}
			desc->src_addr = dma_map_page (dma->dev,
				virt_to_page (dma->vpage[i]),
				offset_in_page (dma->vpage[i]),
				desc->bytes,
				dma->direction);
			if (dma_mapping_error (dma->dev, desc->src_addr)) {
				unsigned int j;

				for (j = 0; j < i; j++) {
					desc = dma->desc[j];
					dma_unmap_page (dma->dev,
						desc->src_addr,
						desc->bytes,
						dma->direction);
				}
				return -ENOMEM;
			}
		}
	} else {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			desc->src_addr = data_addr;
			if ((i % dma->pointers_per_buf) ==
				(dma->pointers_per_buf - 1)) {
				/* This is the last page in this buffer */
				desc->bytes = last_block_size;
			} else {
				desc->bytes = PAGE_SIZE;
			}
			desc->dest_addr = dma_map_page (dma->dev,
				virt_to_page (dma->vpage[i]),
				offset_in_page (dma->vpage[i]),
				desc->bytes,
				dma->direction);
			if (dma_mapping_error (dma->dev, desc->dest_addr)) {
				unsigned int j;

				for (j = 0; j < i; j++) {
					desc = dma->desc[j];
					dma_unmap_page (dma->dev,
						desc->dest_addr,
						desc->bytes,
						dma->direction);
				}
				return -ENOMEM;
			}
		}
	}
	return 0;
}

/**
 * gt64_unmap_buffers - unmap DMA buffers
 * @dma: DMA buffer management structure
 **/
static void
gt64_unmap_buffers (struct master_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct gt64_desc *desc;
	unsigned int i;

	if (dma->direction == DMA_TO_DEVICE) {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			dma_unmap_page (dma->dev,
				desc->src_addr,
				desc->bytes,
				DMA_TO_DEVICE);
		}
	} else {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			dma_unmap_page (dma->dev,
				desc->dest_addr,
				desc->bytes,
				DMA_FROM_DEVICE);
		}
	}
	return;
}

/**
 * gt64_read - copy data from driver buffer to user buffer
 * @dma: DMA buffer management structure
 * @data: user buffer
 * @length: size of data
 *
 * Returns a negative error code on failure and
 * the number of bytes copied on success.
 **/
static ssize_t
gt64_read (struct master_dma *dma, char __user *data, size_t length)
{
	/* Amount of data remaining in the current buffer */
	const size_t max_length = dma->bufsize - dma->cpu_offset;
	size_t chunkvpage, chunkoffset, chunksize;
	size_t copied = 0; /* Number of bytes copied */
	struct gt64_desc *desc;

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
			desc->dest_addr,
			desc->bytes,
			DMA_FROM_DEVICE);
		if (copy_to_user (data + copied,
			dma->vpage[chunkvpage] + chunkoffset,
			chunksize)) {
			dma_sync_single_for_device (dma->dev,
				desc->dest_addr,
				desc->bytes,
				DMA_FROM_DEVICE);
			return -EFAULT;
		}
		dma_sync_single_for_device (dma->dev,
			desc->dest_addr,
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
 * gt64_head_desc_bus_addr - return the bus address of the head descriptor
 * @dma: DMA buffer management structure
 **/
u32
gt64_head_desc_bus_addr (struct master_dma *dma)
{
	const unsigned int prevbuffer =
		(dma->dev_buffer + dma->buffers - 1) % dma->buffers;
	const unsigned int oldlastvpage =
		prevbuffer * dma->pointers_per_buf +
		dma->pointers_per_buf - 1;
	struct gt64_desc *desc = dma->desc[oldlastvpage];

	return desc->next_desc;
}

/**
 * gt64_reset - reset the descriptor chain
 * @dma: DMA buffer management structure
 **/
void
gt64_reset (struct master_dma *dma)
{
	dma->cpu_buffer = 0;
	dma->cpu_offset = 0;
	dma->dev_buffer = 0;
	return;
}

