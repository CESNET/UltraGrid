/* lsdma.h
 *
 * Header file for Linear Systems DMA controller.
  *
 * Copyright (C) 1999 Tony Bolger <d7v@indigo.ie>
 * Copyright (C) 2000-2007 Linear Systems Ltd.
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

#ifndef _LSDMA_H
#define _LSDMA_H

/* Linear Systems DMA Control/Status register addresses */
#define LSDMA_INTMSK		0x04
#define LSDMA_INTSRC		0x0C

/* Linear Systems DMA Channel register addresses */
#define LSDMA_CSR(i)		(0x20 + 0x20 * (i))
#define LSDMA_SIZE(i)		(0x24 + 0x20 * (i))
#define LSDMA_ADDR(i)		(0x28 + 0x20 * (i))
#define LSDMA_ADDR_H(i)		(0x2c + 0x20 * (i))
#define LSDMA_DESC(i)		(0x38 + 0x20 * (i))
#define LSDMA_DESC_H(i)		(0x3c + 0x20 * (i))

/* Linear Systems DMA Interrupt Mask register bit locations */
#define LSDMA_INTMSK_CH(i)	(1 << (i))

/* Linear Systems DMA Interrupt Source register bit locations */
#define LSDMA_INTSRC_CH(i)	(1 << (i))

/* Linear Systems DMA Channel Control/Status register bit locations */
#define LSDMA_CH_CSR_INTSRCSTOP		0x01000000	/* DMA Abort Completed Int Src*/
#define LSDMA_CH_CSR_INTSRCBUFFER	0x00800000	/* Descriptor Buffer Int Src */
#define LSDMA_CH_CSR_INTSRCDESC		0x00400000	/* Descriptor Done Int Src */
#define LSDMA_CH_CSR_INTSRCDONE		0x00200000	/* DMA Done (EOL) Int Src */
#define LSDMA_CH_CSR_INTSRCERR		0x00100000	/* DMA Error Int Src*/
#define LSDMA_CH_CSR_INTDESCENABLE	0x00080000	/* Descriptor Done Int Enable */
#define LSDMA_CH_CSR_INTDONEENABLE	0x00040000	/* DMA Done Int Enable */
#define LSDMA_CH_CSR_INTERRENABLE	0x00020000	/* DMA Error Int Enable */
#define LSDMA_CH_CSR_INTSTOPENABLE	0x00010000	/* DMA Abort Int Enable */
#define LSDMA_CH_CSR_ERR		0x00001000	/* DMA Error Status */
#define LSDMA_CH_CSR_DONE		0x00000800	/* DMA Done Status */
#define LSDMA_CH_CSR_STOP		0x00000200	/* DMA Abort */
#define LSDMA_CH_CSR_DIRECTION		0x00000002	/* Xfer Direction - Card to System when set */
#define LSDMA_CH_CSR_ENABLE		0x00000001	/* DMA Enable */

/* Linear Systems DMA Channel Size register bit locations */
#define LSDMA_CH_SZ_TOTALXFERSIZE	0x0000FFFF

/* Linear Systems DMA Descriptor CSR bit locations */
#define LSDMA_DESC_CSR_INT		0x00200000	/* Buffer Interrupt */
#define LSDMA_DESC_CSR_EOL		0x00100000	/* End of Link */
#define LSDMA_DESC_CSR_DIRECTION	0x00010000	/* Xfer Direction - Card to System when set */
#define LSDMA_DESC_CSR_TOTALXFERSIZE	0x0000FFFF	/* Size of data */

#define LSDMA_DESC_CSR_EOL_ORDER	20
#define LSDMA_DESC_CSR_INT_ORDER	21

#include <linux/types.h> /* size_t, u32 */
#include <linux/pci.h> /* pci_dev, pci_pool */
#include <linux/mm.h> /* vm_operations_struct */

#define LSDMA_MMAP	0x00000001

/**
 * lsdma_dma - DMA information structure
 * @pdev: PCI device
 * @buffers: number of buffers
 * @bufsize: number of bytes in each buffer
 * @pointers_per_buf: number of descriptors per buffer
 * @direction: direction of data flow
 * @desc_pool: DMA-coherent memory pool
 * @desc: pointer to an array of pointers to DMA descriptors
 * @page: pointer to an array of pointers to memory pages
 * @vpage: pointer to an array of pointers to DMA buffer fragments
 * @dev_buffer: buffer being accessed by the device
 * @cpu_buffer: buffer being accessed by the CPU
 * @cpu_offset: offset of the CPU access point within cpu_buffer
 * @flags: allocation flags
 **/
struct lsdma_dma {
	struct pci_dev *pdev;
	unsigned int buffers;
	unsigned int bufsize;
	unsigned int pointers_per_buf;
	unsigned int direction;
	struct pci_pool *desc_pool;
	struct lsdma_desc **desc;
	unsigned long *page;
	unsigned char **vpage;
	volatile size_t dev_buffer;
	size_t cpu_buffer;
	size_t cpu_offset;
	unsigned int flags;
};

/**
 * lsdma_desc - Linear Systems DMA descriptor
 * @csr: configuration/size register
 * @src_addr: source address
 * @dest_addr: destination address
 * @next_desc: address of next descriptor, and control flags
 **/
struct lsdma_desc {
	u32 csr;
	u32 src_addr;
	u32 dest_addr;
	u32 next_desc;

	/* High Order Bits for 64bit Addresses */
	u32 src_addr_h;
	u32 dest_addr_h;
	u32 next_desc_h;
};

/* External variables */

extern struct vm_operations_struct lsdma_vm_ops;

/* External function prototypes */

struct lsdma_dma *lsdma_alloc (struct pci_dev *pdev,
	u32 data_addr,
	unsigned int buffers,
	unsigned int bufsize,
	unsigned int direction,
	unsigned int flags);
void lsdma_free (struct lsdma_dma *dma);
ssize_t lsdma_read (struct lsdma_dma *dma, char *data, size_t length);
ssize_t lsdma_write (struct lsdma_dma *dma, const char *data, size_t length);
dma_addr_t lsdma_head_desc_bus_addr (struct lsdma_dma *dma);
void lsdma_tx_link_all (struct lsdma_dma *dma);
void lsdma_reset (struct lsdma_dma *dma);
ssize_t lsdma_txdqbuf (struct lsdma_dma *dma, size_t bufnum);
ssize_t lsdma_txqbuf (struct lsdma_dma *dma, size_t bufnum);
ssize_t lsdma_rxdqbuf (struct lsdma_dma *dma, size_t bufnum);
ssize_t lsdma_rxqbuf (struct lsdma_dma *dma, size_t bufnum);

/* Inline functions */

/**
 * lsdma_tx_buflevel - return the number of transmit buffers in use
 * @dma: DMA information structure
 *
 * We don't lock dma->dev_buffer here because
 * simple reads and writes should be atomic.
 **/
static inline int
lsdma_tx_buflevel (struct lsdma_dma *dma)
{
	return ((dma->cpu_buffer + dma->buffers - dma->dev_buffer) %
		dma->buffers);
}

/**
 * lsdma_tx_isfull - return true if the transmit buffers are full
 * @dma: DMA information structure
 *
 * We don't lock dma->dev_buffer here because
 * simple reads and writes should be atomic.
 **/
static inline int
lsdma_tx_isfull (struct lsdma_dma *dma)
{
	return (dma->dev_buffer ==
		((dma->cpu_buffer + 1) % dma->buffers));
}

/**
 * lsdma_rx_buflevel - return the number of receive buffers in use
 * @dma: DMA information structure
 *
 * We don't lock dma->dev_buffer here because
 * simple reads and writes should be atomic.
 **/
static inline int
lsdma_rx_buflevel (struct lsdma_dma *dma)
{
	return ((dma->dev_buffer + dma->buffers - dma->cpu_buffer) %
		dma->buffers);
}

/**
 * lsdma_rx_isempty - return true if the receive buffers are empty
 * @dma: DMA information structure
 *
 * We don't lock dma->dev_buffer here because
 * simple reads and writes should be atomic.
 **/
static inline int
lsdma_rx_isempty (struct lsdma_dma *dma)
{
	return (dma->cpu_buffer == dma->dev_buffer);
}

/**
 * lsdma_advance - increment the device buffer pointer
 * @dma: DMA information structure
 *
 * We don't lock because this function is the only
 * dev_buffer writer, it should only be called from
 * a single interrupt service routine,
 * and dev_buffer reads should be atomic.
 **/
static inline void
lsdma_advance (struct lsdma_dma *dma)
{
	dma->dev_buffer = (dma->dev_buffer + 1) % dma->buffers;
	return;
}

/**
 * lsdma_desc_to_dma - return a dma address from a descriptor address
 * @desc_low: low 32 bits of descriptor pointer
 * @desc_high: high 32 bits of descriptor pointer
 *
 * Handles 32 and 64 bit addresses.
 **/
static inline dma_addr_t
lsdma_desc_to_dma (u32 desc_low, u32 desc_high)
{
	if (sizeof (dma_addr_t) == 4) {
		return (dma_addr_t)desc_low;
	} else {
		return (dma_addr_t)((((u64)desc_high) << 32) | desc_low);
	}
}

/**
 * lsdma_dma_to_desc_low - return the low 32 bits of a dma address
 * @dma_addr: dma address
 *
 * Handles 32 and 64 bit addresses.
 **/
static inline u32
lsdma_dma_to_desc_low (dma_addr_t dma_addr)
{
	return (u32)(dma_addr & 0xffffffff);
}

/**
 * lsdma_dma_to_desc_high - return the high 32 bits of a dma address
 * @dma_addr: dma address
 *
 * Handles 32 and 64 bit addresses.
 **/
static inline u32
lsdma_dma_to_desc_high (dma_addr_t dma_addr)
{
	return (u32)((((u64)dma_addr) >> 32) & 0xffffffff);
}

#endif

