/* lsdma.h
 *
 * Header file for Linear Systems DMA controller.
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
#define LSDMA_CH_CSR_INTSRCSTOP		0x01000000	/* DMA Abort Completed Int Src */
#define LSDMA_CH_CSR_INTSRCBUFFER	0x00800000	/* Descriptor Buffer Int Src */
#define LSDMA_CH_CSR_INTSRCDESC		0x00400000	/* Descriptor Done Int Src */
#define LSDMA_CH_CSR_INTSRCDONE		0x00200000	/* DMA Done (EOL) Int Src */
#define LSDMA_CH_CSR_INTSRCERR		0x00100000	/* DMA Error Int Src */
#define LSDMA_CH_CSR_INTDESCENABLE	0x00080000	/* Descriptor Done Int Enable */
#define LSDMA_CH_CSR_INTDONEENABLE	0x00040000	/* DMA Done Int Enable */
#define LSDMA_CH_CSR_INTERRENABLE	0x00020000	/* DMA Error Int Enable */
#define LSDMA_CH_CSR_INTSTOPENABLE	0x00010000	/* DMA Abort Int Enable */
#define LSDMA_CH_CSR_ERR		0x00001000	/* DMA Error Status */
#define LSDMA_CH_CSR_DONE		0x00000800	/* DMA Done Status */
#define LSDMA_CH_CSR_STOP		0x00000200	/* DMA Abort */
#define LSDMA_CH_CSR_64BIT		0x00000080	/* 64-bit DMA */
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

#include <linux/types.h> /* u32 */

#include "mdma.h"

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

extern struct master_dma_operations lsdma_dma_ops;

/* External function prototypes */

dma_addr_t lsdma_head_desc_bus_addr (struct master_dma *dma);
void lsdma_tx_link_all (struct master_dma *dma);
void lsdma_reset (struct master_dma *dma);

#endif

