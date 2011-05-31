/* plx9080.h
 *
 * Header file for plx9080.c.
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

#ifndef _PLX9080_H
#define _PLX9080_H

/* PLX 9080 local configuration register addresses */
#define PLX_LAS0RR	0x00
#define PLX_LAS0BA	0x04
#define PLX_MARBR	0x08
#define PLX_BIGEND	0x0C
#define PLX_EROMRR	0x10
#define PLX_EROMBA	0x14
#define PLX_LBRD0	0x18
#define PLX_DMRR	0x1C
#define PLX_DMLBAM	0x20
#define PLX_DMLBAI	0x24
#define PLX_DMPBAM	0x28
#define PLX_DMCFGA	0x2C
#define PLX_LAS1RR	0xF0
#define PLX_LAS1BA	0xF4
#define PLX_LBRD1	0xF8

/* PLX 9080 Local Address Space Range Register bit locations */
#define PLX_LASRR_IO	0x00000001 /* PCI I/O Space */

/* PLX 9080 runtime register addresses */
#define PLX_MBOX0	0x40
#define PLX_MBOX1	0x44
#define PLX_MBOX2	0x48
#define PLX_MBOX3	0x4C
#define PLX_MBOX4	0x50
#define PLX_MBOX5	0x54
#define PLX_MBOX6	0x58
#define PLX_MBOX7	0x5C
#define PLX_P2LDBELL	0x60
#define PLX_L2PDBELL	0x64
#define PLX_INTCSR	0x68
#define PLX_CNTRL	0x6C
#define PLX_PCIHIDR	0x70
#define PLX_PCIHREV	0x74

/* PLX 9080 Control Register bit locations */
#define PLX_CNTRL_EESK		0x01000000 /* Serial EEPROM Clock */
#define PLX_CNTRL_EECS		0x02000000 /* Serial EEPROM Chip Select */
#define PLX_CNTRL_EEW		0x04000000 /* Serial EEPROM Write Bit */
#define PLX_CNTRL_EER		0x08000000 /* Serial EEPROM Read Bit */
#define PLX_CNTRL_RECONFIG	0x20000000 /* Reload Configuration Registers */
#define PLX_CNTRL_RESET		0x40000000 /* PCI Adapter Software Reset */
#define PLX_CNTRL_EEDHIZ	0x80000000 /* Serial EEPROM Tristate Data */

#define PLX_CNTRL_PCIRCCDMA_MASK	0x0000000f
#define PLX_CNTRL_PCIRCCDMA_SHIFT	0
#define PLX_CNTRL_PCIWCCDMA_MASK	0x000000f0
#define PLX_CNTRL_PCIWCCDMA_SHIFT	4
#define PLX_CNTRL_PCIMRCCDM_MASK	0x00000f00
#define PLX_CNTRL_PCIMRCCDM_SHIFT	8
#define PLX_CNTRL_PCIMWCCDM_MASK	0x0000f000
#define PLX_CNTRL_PCIMWCCDM_SHIFT	12

/* PLX 9080 Interrupt Control / Status Register bit locations */
#define PLX_INTCSR_LOCMBINT_ENABLE	0x00000008 /* Mailbox Int. Enable */
#define PLX_INTCSR_PCIINT_ENABLE	0x00000100 /* PCI Int. Enable */
#define PLX_INTCSR_PCIDBINT_ENABLE	0x00000200 /* PCI Drbell Int. Enable */
#define PLX_INTCSR_PCIABINT_ENABLE	0x00000400 /* PCI Abort Int. Enable */
#define PLX_INTCSR_PCILOCINT_ENABLE	0x00000800 /* PCI Local Int. Enable */
#define PLX_INTCSR_PCIRAEINT_ENABLE	0x00001000 /* Retry Abort Enable */
#define PLX_INTCSR_PCIDBINT_ACTIVE	0x00002000 /* PCI Drbell Int. Active */
#define PLX_INTCSR_PCIABINT_ACTIVE	0x00004000 /* PCI Abort Int. Active */
#define PLX_INTCSR_PCILOCINT_ACTIVE	0x00008000 /* PCI Local Int. Active */
#define PLX_INTCSR_LOCOUTINT_ENABLE	0x00010000 /* Local Int. Output En. */
#define PLX_INTCSR_LOCDBINT_ENABLE	0x00020000 /* Local Drbell Int. En. */
#define PLX_INTCSR_DMA0INT_ENABLE	0x00040000 /* Local DMA0 Int. Enable */
#define PLX_INTCSR_DMA1INT_ENABLE	0x00080000 /* Local DMA1 Int. Enable */
#define PLX_INTCSR_LOCDBINT_ACTIVE	0x00100000 /* Local Drbell Int. Actv. */
#define PLX_INTCSR_DMA0INT_ACTIVE	0x00200000 /* DMA0 Int. Active */
#define PLX_INTCSR_DMA1INT_ACTIVE	0x00400000 /* DMA1 Int. Active */
#define PLX_INTCSR_PCIBIST_ACTIVE	0x00800000 /* BIST Int. Active */

/* PLX 9080 DMA register addresses */
#define PLX_DMAMODE0	0x80
#define PLX_DMAPADR0	0x84
#define PLX_DMALADR0	0x88
#define PLX_DMASIZ0	0x8C
#define PLX_DMADPR0	0x90
#define PLX_DMAMODE1	0x94
#define PLX_DMAPADR1	0x98
#define PLX_DMALADR1	0x9C
#define PLX_DMASIZ1	0xA0
#define PLX_DMADPR1	0xA4
#define PLX_DMACSR0	0xA8
#define PLX_DMACSR1	0xA9
#define PLX_DMAARB	0xAC /* Shadow of PLX_MARBR */
#define PLX_DMATHR	0xB0

/* PLX 9080 DMA Mode Register bit locations */
#define PLX_DMAMODE_READY	0x00000040 /* Ready Input Enable */
#define PLX_DMAMODE_BTERM	0x00000080 /* ~BTERM Input Enable */
#define PLX_DMAMODE_LOCALBURST	0x00000100 /* Local Burst Enable */
#define PLX_DMAMODE_CHAINED	0x00000200 /* Chaining */
#define PLX_DMAMODE_INT		0x00000400 /* Done Interrupt Enable */
#define PLX_DMAMODE_CLOC	0x00000800 /* Local Addressing Mode */
#define PLX_DMAMODE_DEMAND	0x00001000 /* Demand Mode */
#define PLX_DMAMODE_INV		0x00002000 /* Write and Invalidate Mode */
#define PLX_DMAMODE_EOT		0x00004000 /* End of Transfer Enable */
#define PLX_DMAMODE_SMODE	0x00008000 /* Stop Data Transfer Mode */
#define PLX_DMAMODE_CCOUNT	0x00010000 /* Clear Count Mode */
#define PLX_DMAMODE_INTPCI	0x00020000 /* Interrupt Select */

#define PLX_DMAMODE_8BIT	0x00000000
#define PLX_DMAMODE_16BIT	0x00000001
#define PLX_DMAMODE_32BIT	0x00000003

/* DMA Command / Status Register bit locations */
#define PLX_DMACSR_ENABLE	0x01 /* Enable */
#define PLX_DMACSR_START	0x02 /* Start */
#define PLX_DMACSR_ABORT	0x04 /* Abort */
#define PLX_DMACSR_CLINT	0x08 /* Clear Interrupt */
#define PLX_DMACSR_DONE		0x10 /* Done */

/* Descriptor Pointer Register bit locations */
#define PLX_DMADPR_DLOC_PCI	0x1 /* Descriptor Location */
#define PLX_DMADPR_EOC		0x2 /* End of Chain */
#define PLX_DMADPR_INT		0x4 /* Interrupt after Terminal Count */
#define PLX_DMADPR_LB2PCI	0x8 /* Direction of Transfer */

#define PLX_DMADPR_EOC_ORDER	1 /* log2 (PLX_DMADPR_EOC) */
#define PLX_DMADPR_INT_ORDER	2 /* log2 (PLX_DMADPR_INT) */

#include <linux/types.h> /* u32 */

#include "mdma.h"

/**
 * plx_desc - PLX 9080 DMA descriptor
 * @pci_addr: PCI address
 * @local_addr: local bus address
 * @bytes: number of bytes to transfer
 * @next_desc: address of next descriptor, and control flags
 **/
struct plx_desc {
	u32 pci_addr;
	u32 local_addr;
	u32 bytes;
	u32 next_desc;
};

/* External variables */

extern struct master_dma_operations plx_dma_ops;

/* External function prototypes */

void plx_reset_bridge (void __iomem *addr);
u32 plx_head_desc_bus_addr (struct master_dma *dma);
void plx_tx_link_all (struct master_dma *dma);
void plx_reset (struct master_dma *dma);

#endif

