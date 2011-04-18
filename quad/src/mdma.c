/* mdma.c
 *
 * DMA linked-list buffer management for Linear Systems Ltd. Master devices.
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

#include <linux/version.h> /* LINUX_VERSION_CODE */
#include <linux/slab.h> /* kmalloc () */
#include <linux/spinlock.h> /* spin_lock_init () */
#include <linux/errno.h> /* error codes */
#include <linux/mm.h> /* get_page () */

#include <asm/bitops.h> /* clear_bit () */
#include <asm/uaccess.h> /* copy_from_user () */

#include "mdma.h"

/* Static function prototypes */
static int mdma_alloc_buffers (struct master_dma *dma);
static void mdma_free_buffers (struct master_dma *dma);
#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,26))
static struct page *mdma_nopage (struct vm_area_struct *vma,
	unsigned long address,
	int *type);
struct vm_operations_struct mdma_vm_ops = {
	.nopage = mdma_nopage
};
#else
static int mdma_fault (struct vm_area_struct *vma,
	struct vm_fault *vmf);
struct vm_operations_struct mdma_vm_ops = {
	.fault = mdma_fault
};
#endif

/**
 * mdma_alloc_buffers - allocate DMA buffers
 * @dma: DMA information structure
 *
 * Returns a negative error code on failure and 0 on success.
 **/
static int
mdma_alloc_buffers (struct master_dma *dma)
{
	const unsigned int tail_size = dma->bufsize % PAGE_SIZE;
	const unsigned int tails_per_buf = (tail_size > 0) ? 1 : 0;
	const unsigned int total_pointers =
		dma->pointers_per_buf * dma->buffers;
	const unsigned int full_pages =
		dma->bufsize / PAGE_SIZE * dma->buffers;
	const unsigned int tails_per_page = (tail_size > 0) ?
		((dma->flags & MDMA_MMAP) ? 1 : (PAGE_SIZE / tail_size)) : 0;
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
 * mdma_free_buffers - free DMA buffers
 * @dma: DMA information structure
 **/
static void
mdma_free_buffers (struct master_dma *dma)
{
	const unsigned int tail_size = dma->bufsize % PAGE_SIZE;
	const unsigned int full_pages =
		dma->bufsize / PAGE_SIZE * dma->buffers;
	const unsigned int tails_per_page = (tail_size > 0) ?
		((dma->flags & MDMA_MMAP) ? 1 : (PAGE_SIZE / tail_size)) : 0;
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
 * mdma_alloc - Create a DMA information structure
 * @dev: device
 * @ops: pointer to a structure of pointers to DMA helper functions
 * @data_addr: local bus address of the data register
 * @buffers: number of buffers
 * @bufsize: number of bytes in each buffer
 * @direction: direction of data flow
 * @flags: allocation flags
 *
 * Returns %NULL on failure and a pointer to
 * the DMA information structure on success.
 **/
struct master_dma *
mdma_alloc (struct device *dev,
	struct master_dma_operations *ops,
	u32 data_addr,
	unsigned int buffers,
	unsigned int bufsize,
	unsigned int direction,
	unsigned int flags)
{
	const unsigned int tail_size = bufsize % PAGE_SIZE;
	const unsigned int tails_per_buf = (tail_size > 0) ? 1 : 0;
	struct master_dma *dma;

	/* Allocate and initialize a DMA information structure */
	dma = (struct master_dma *)kzalloc (sizeof (*dma), GFP_KERNEL);
	if (dma == NULL) {
		goto NO_DMA;
	}
	dma->dev = dev;
	dma->ops = ops;
	dma->buffers = buffers;
	dma->bufsize = bufsize;
	dma->pointers_per_buf = bufsize / PAGE_SIZE + tails_per_buf;
	dma->direction = direction;
	spin_lock_init (&dma->lock);
	dma->flags = flags;

	/* Allocate DMA buffers */
	if (mdma_alloc_buffers (dma) < 0) {
		goto NO_BUF;
	}

	/* Allocate DMA descriptors */
	if (ops->free_descriptors &&
		ops->alloc_descriptors &&
		ops->alloc_descriptors (dma) < 0) {
		goto NO_DESC;
	}

	/* Map DMA buffers */
	if (ops->unmap_buffers &&
		ops->map_buffers &&
		ops->map_buffers (dma, data_addr) < 0) {
		goto NO_MAP;
	}

	return dma;

NO_MAP:
	ops->free_descriptors (dma);
NO_DESC:
	mdma_free_buffers (dma);
NO_BUF:
	kfree (dma);
NO_DMA:
	return NULL;
}

/**
 * mdma_free - Destroy a DMA information structure
 * @dma: DMA information structure
 **/
void
mdma_free (struct master_dma *dma)
{
	dma->ops->unmap_buffers (dma);
	dma->ops->free_descriptors (dma);
	mdma_free_buffers (dma);
	kfree (dma);
	return;
}

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,26))
/**
 * mdma_nopage - page fault handler
 * @vma: VMA
 * @address: address of the faulted page
 * @type: pointer to returned fault type
 *
 * Returns a pointer to the page if it is available,
 * NOPAGE_SIGBUS otherwise.
 **/
static struct page *
mdma_nopage (struct vm_area_struct *vma,
	unsigned long address,
	int *type)
{
	struct master_dma *dma = vma->vm_private_data;
	unsigned long offset = address - vma->vm_start +
		(vma->vm_pgoff << PAGE_SHIFT);
	void *pageptr;
	struct page *pg = NOPAGE_SIGBUS;

	if (offset >= dma->pointers_per_buf * dma->buffers * PAGE_SIZE) {
		goto OUT;
	}
	pageptr = dma->vpage[offset >> PAGE_SHIFT];
	pg = virt_to_page (pageptr);
	get_page (pg);
	if (type) {
		*type = VM_FAULT_MINOR;
	}
OUT:
	return pg;
}
#else
/**
 * mdma_fault - page fault handler
 * @vma: VMA
 * @vmf: vm_fault structure
 *
 * Arrange for a missing page to exist and return its address.
 **/
static int
mdma_fault (struct vm_area_struct *vma,
	struct vm_fault *vmf)
{
	struct master_dma *dma = vma->vm_private_data;

	if (vmf->pgoff >= dma->pointers_per_buf * dma->buffers) {
		return VM_FAULT_SIGBUS;
	}
	vmf->page = virt_to_page (dma->vpage[vmf->pgoff]);
	get_page (vmf->page);
	return 0;
}
#endif

