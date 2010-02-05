/* gt64131.c
 *
 * DMA linked-list buffer management for the Marvell GT-64131.
 *
 * Copyright (C) 2003-2006 Linear Systems Ltd.
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
#include <linux/pci.h> /* pci_pool_create () */
#include <linux/errno.h> /* error codes */

#include <asm/uaccess.h> /* copy_to_user () */

#include "gt64131.h"

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,5))
#define pci_dma_sync_single_for_cpu pci_dma_sync_single
#define pci_dma_sync_single_for_device(w,x,y,z)
#endif

/* Static function prototypes */
static int gt64_alloc_buffers (struct gt64_dma *dma);
static void gt64_free_buffers (struct gt64_dma *dma);
static int gt64_alloc_descriptors (struct gt64_dma *dma);
static void gt64_free_descriptors (struct gt64_dma *dma);
static int gt64_map_buffers (struct gt64_dma *dma, u32 data_addr);
static void gt64_unmap_buffers (struct gt64_dma *dma);

/**
 * gt64_alloc_buffers - allocate DMA buffers
 * @dma: DMA buffer management structure
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
gt64_alloc_buffers (struct gt64_dma *dma)
{
	const unsigned int tail_size = dma->bufsize % PAGE_SIZE;
	const unsigned int tails_per_buf = (tail_size > 0) ? 1 : 0;
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int full_pages =
		dma->bufsize / PAGE_SIZE * dma->buffers;
	const unsigned int tails_per_page = (tail_size > 0) ?
		PAGE_SIZE / tail_size : 0;
	const unsigned int tail_pages = (tail_size > 0) ?
		dma->buffers / tails_per_page +
		((dma->buffers % tails_per_page) ? 1 : 0) : 0;
	const unsigned int total_pages = full_pages + tail_pages;
	unsigned int i, bufnum;

	/* Allocate an array of pointers to pages */
	if ((dma->page = (unsigned long *)kmalloc (
		total_pages * sizeof (*dma->page),
		GFP_KERNEL)) == NULL) {
		goto NO_PAGE_PTR;
	}

	/* Allocate an array of pointers to locations in the page array */
	if ((dma->vpage = (unsigned char **)kmalloc (
		total_pointers * sizeof (*dma->vpage),
		GFP_KERNEL)) == NULL) {
		goto NO_VPAGE_PTR;
	}

	/* Allocate pages */
	for (i = 0; i < total_pages; i++) {
		if ((dma->page[i] = get_zeroed_page (GFP_KERNEL)) == 0) {
			int j;

			for (j = 0; j < i; j++) {
				free_page (dma->page[j]);
			}
			goto NO_BUF;
		}
	}

	/* Fill dma->vpage[] with pointers to pages and parts of pages
	 * in dma->page[] */
	for (i = 0; i < total_pointers; i++) {
		bufnum = i / dma->pointers_per_buf;
		dma->vpage[i] = (unsigned char *)(((tail_size > 0) &&
			((i % dma->pointers_per_buf) ==
			(dma->pointers_per_buf - 1))) ?
			dma->page[full_pages + bufnum / tails_per_page] +
			(bufnum % tails_per_page) * tail_size :
			dma->page[i - bufnum * tails_per_buf]);
	}

	return 0;

NO_BUF:
	kfree (dma->vpage);
NO_VPAGE_PTR:
	kfree (dma->page);
NO_PAGE_PTR:
	return -ENOMEM;
}

/**
 * gt64_free_buffers - free DMA buffers
 * @dma: DMA buffer management structure
 **/
static void
gt64_free_buffers (struct gt64_dma *dma)
{
	const unsigned int tail_size = dma->bufsize % PAGE_SIZE;
	const unsigned int full_pages =
		dma->bufsize / PAGE_SIZE * dma->buffers;
	const unsigned int tails_per_page = (tail_size > 0) ?
		PAGE_SIZE / tail_size : 0;
	const unsigned int tail_pages = (tail_size > 0) ?
		dma->buffers / tails_per_page +
		((dma->buffers % tails_per_page) ? 1 : 0) : 0;
	const unsigned int total_pages = full_pages + tail_pages;
	unsigned int i;

	for (i = 0; i < total_pages; i++) {
		free_page (dma->page[i]);
	}
	kfree (dma->vpage);
	kfree (dma->page);
	return;
}

/**
 * gt64_alloc_descriptors - allocate DMA descriptors
 * @dma: DMA buffer management structure
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
gt64_alloc_descriptors (struct gt64_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	dma_addr_t dma_addr, first_dma_addr;
	struct gt64_desc *desc;
	unsigned int i;

	/* Allocate an array of pointers to descriptors */
	if ((dma->desc = (struct gt64_desc **)kmalloc (
		total_pointers * sizeof (*dma->desc),
		GFP_KERNEL)) == NULL) {
		goto NO_DESC_PTR;
	}

	/* Allocate the DMA descriptors */
	if ((dma->desc_pool = pci_pool_create ("gt64",
		dma->pdev,
		sizeof (struct gt64_desc),
		sizeof (struct gt64_desc),
		0)) == NULL) {
		goto NO_PCI_POOL;
	}
	if ((desc = dma->desc[0] = pci_pool_alloc (dma->desc_pool,
		GFP_KERNEL, &first_dma_addr)) == NULL) {
		goto NO_DESC;
	}
	for (i = 1; i < total_pointers; i++) {
		if ((dma->desc[i] = pci_pool_alloc (dma->desc_pool,
			GFP_KERNEL,
			&dma_addr)) == NULL) {
			unsigned int j;

			for (j = i - 1; j > 0; j--) {
				desc = dma->desc[j - 1];
				pci_pool_free (dma->desc_pool,
					dma->desc[j],
					desc->next_desc);
			}
			pci_pool_free (dma->desc_pool,
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
	pci_pool_destroy (dma->desc_pool);
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
gt64_free_descriptors (struct gt64_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct gt64_desc *prev_desc = dma->desc[total_pointers - 1];
	dma_addr_t dma_addr = prev_desc->next_desc;
	unsigned int i;

	for (i = total_pointers - 1; i > 0; i--) {
		prev_desc = dma->desc[i - 1];
		pci_pool_free (dma->desc_pool,
			dma->desc[i],
			prev_desc->next_desc);
	}
	pci_pool_free (dma->desc_pool,
		dma->desc[0], dma_addr);
	pci_pool_destroy (dma->desc_pool);
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
gt64_map_buffers (struct gt64_dma *dma, u32 data_addr)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int last_block_size = dma->bufsize -
		((dma->pointers_per_buf - 1) * PAGE_SIZE);
	struct gt64_desc *desc;
	unsigned int i;

	if (dma->direction == PCI_DMA_TODEVICE) {
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
			if ((desc->src_addr = pci_map_page (dma->pdev,
				virt_to_page (dma->vpage[i]),
				offset_in_page (dma->vpage[i]),
				desc->bytes,
				dma->direction)) == 0) {
				unsigned int j;

				for (j = 0; j < i; j++) {
					desc = dma->desc[j];
					pci_unmap_page (dma->pdev,
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
			if ((desc->dest_addr = pci_map_page (dma->pdev,
				virt_to_page (dma->vpage[i]),
				offset_in_page (dma->vpage[i]),
				desc->bytes,
				dma->direction)) == 0) {
				unsigned int j;

				for (j = 0; j < i; j++) {
					desc = dma->desc[j];
					pci_unmap_page (dma->pdev,
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
gt64_unmap_buffers (struct gt64_dma *dma)
{
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	struct gt64_desc *desc;
	unsigned int i;

	if (dma->direction == PCI_DMA_TODEVICE) {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			pci_unmap_page (dma->pdev,
				desc->src_addr,
				desc->bytes,
				PCI_DMA_TODEVICE);
		}
	} else {
		for (i = 0; i < total_pointers; i++) {
			desc = dma->desc[i];
			pci_unmap_page (dma->pdev,
				desc->dest_addr,
				desc->bytes,
				PCI_DMA_FROMDEVICE);
		}
	}
	return;
}

/**
 * gt64_alloc - Create a DMA buffer management structure
 * @pdev: PCI device
 * @data_addr: local bus address of the data register
 * @buffers: number of buffers
 * @bufsize: number of bytes in each buffer
 * @direction: direction of data flow
 *
 * Returns %NULL on failure and a pointer to
 * the DMA buffer management structure on success.
 **/
struct gt64_dma *
gt64_alloc (struct pci_dev *pdev,
	u32 data_addr,
	unsigned int buffers,
	unsigned int bufsize,
	unsigned int direction)
{
	const unsigned int tail_size = bufsize % PAGE_SIZE;
	const unsigned int tails_per_buf = (tail_size > 0) ? 1 : 0;
	struct gt64_dma *dma;

	/* Allocate and initialize a DMA buffer management structure */
	dma = (struct gt64_dma *)kmalloc (sizeof (*dma), GFP_KERNEL);
	if (dma == NULL) {
		goto NO_DMA;
	}
	memset (dma, 0, sizeof (*dma));
	dma->pdev = pdev;
	dma->buffers = buffers;
	dma->bufsize = bufsize;
	dma->pointers_per_buf = bufsize / PAGE_SIZE + tails_per_buf;
	dma->direction = direction;

	/* Allocate DMA buffers */
	if (gt64_alloc_buffers (dma) < 0) {
		goto NO_BUF;
	}

	/* Allocate DMA descriptors */
	if (gt64_alloc_descriptors (dma) < 0) {
		goto NO_DESC;
	}

	/* Map DMA buffers */
	if (gt64_map_buffers (dma, data_addr) < 0) {
		goto NO_MAP;
	}

	return dma;

NO_MAP:
	gt64_free_descriptors (dma);
NO_DESC:
	gt64_free_buffers (dma);
NO_BUF:
	kfree (dma);
NO_DMA:
	return NULL;
}

/**
 * gt64_free - Destroy a DMA buffer management structure
 * @dma: DMA buffer management structure
 **/
void
gt64_free (struct gt64_dma *dma)
{
	gt64_unmap_buffers (dma);
	gt64_free_descriptors (dma);
	gt64_free_buffers (dma);
	kfree (dma);
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
ssize_t
gt64_read (struct gt64_dma *dma, char *data, size_t length)
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
		pci_dma_sync_single_for_cpu (dma->pdev,
			desc->dest_addr,
			desc->bytes,
			PCI_DMA_FROMDEVICE);
		if (copy_to_user (data + copied,
			dma->vpage[chunkvpage] + chunkoffset,
			chunksize)) {
			pci_dma_sync_single_for_device (dma->pdev,
				desc->dest_addr,
				desc->bytes,
				PCI_DMA_FROMDEVICE);
			return -EFAULT;
		}
		pci_dma_sync_single_for_device (dma->pdev,
			desc->dest_addr,
			desc->bytes,
			PCI_DMA_FROMDEVICE);
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
gt64_head_desc_bus_addr (struct gt64_dma *dma)
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
gt64_reset (struct gt64_dma *dma)
{
	dma->cpu_buffer = 0;
	dma->cpu_offset = 0;
	dma->dev_buffer = 0;
	return;
}

