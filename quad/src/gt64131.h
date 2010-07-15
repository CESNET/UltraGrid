/* gt64131.h
 *
 * Header file for gt64131.c.
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

#ifndef _GT64131_H
#define _GT64131_H

/* GT-64131 CPU Configuration register addresses */
#define GT64_CPU		0x000
#define GT64_MULTIGT		0x120

/* GT-64131 CPU Address Decode register addresses */
#define GT64_SCS10LDA		0x008
#define GT64_SCS10HDA		0x010
#define GT64_SCS32LDA		0x018
#define GT64_SCS32HDA		0x020
#define GT64_CS20LDA		0x028
#define GT64_CS20HDA		0x030
#define GT64_CS3BOOTCSLDA	0x038
#define GT64_CS3BOOTCSHDA	0x040
#define GT64_PCIIOLDA		0x048
#define GT64_PCIMEM0LDA		0x058
#define GT64_PCIMEM0HDA		0x060
#define GT64_ISD		0x068
#define GT64_BEA		0x070
#define GT64_PCIMEM1LDA		0x080
#define GT64_PCIMEM1HDA		0x088
#define GT64_SCS10AR		0x0d0
#define GT64_SCS32AR		0x0d8
#define GT64_CS20R		0x0e0
#define GT64_CS3BOOTCSR		0x0e8
#define GT64_PCIIOR		0x0f0
#define GT64_PCIMEM0R		0x0f8
#define GT64_PCIMEM1R		0x100

/* GT-64131 CPU Sync Barrier register addresses */
#define GT64_PCISBV		0x0c0

/* GT-64131 SDRAM and Device Address Decode register addresses */
#define GT64_SCS0LDA		0x400
#define GT64_SCS0HDA		0x404
#define GT64_SCS1LDA		0x408
#define GT64_SCS1HDA		0x40c
#define GT64_SCS2LDA		0x410
#define GT64_SCS2HDA		0x414
#define GT64_SCS3LDA		0x418
#define GT64_SCS3HDA		0x41c
#define GT64_CS0LDA		0x420
#define GT64_CS0HDA		0x424
#define GT64_CS1LDA		0x428
#define GT64_CS1HDA		0x42c
#define GT64_CS2LDA		0x430
#define GT64_CS2HDA		0x434
#define GT64_CS3LDA		0x438
#define GT64_CS3HDA		0x43c
#define GT64_BOOTCSLDA		0x440
#define GT64_BOOTCSHDA		0x444
#define GT64_ADE		0x470

/* GT-64131 SDRAM Configuration register addresses */
#define GT64_SDRAMCFG		0x448
#define GT64_SDRAMOM		0x474
#define GT64_SDRAMBM		0x478
#define GT64_SDRAMAD		0x47c

/* GT-64131 SDRAM Parameters register addresses */
#define GT64_SDRAM(b)		(0x44c+(b)*4)

/* GT-64131 Device Parameters register addresses */
#define GT64_DEVICE(b)		(0x45c+(b)*4)
#define GT64_DEVICEBOOT		0x46c

/* GT-64131 ECC register addresses */
#define GT64_ECCUD		0x480
#define GT64_ECCLD		0x484
#define GT64_ECCFM		0x488
#define GT64_ECCC		0x48c
#define GT64_ECCER		0x490

/* GT-64131 DMA Record register addresses */
#define GT64_DMABC(c)		(0x800+(c)*4)
#define GT64_DMASA(c)		(0x810+(c)*4)
#define GT64_DMADA(c)		(0x820+(c)*4)
#define GT64_DMANRP(c)		(0x830+(c)*4)
#define GT64_DMACDP(c)		(0x870+(c)*4)

/* GT-64131 DMA Channel Control register addresses */
#define GT64_DMACTL(c)		(0x840+(c)*4)

/* GT-64131 DMA Channel Control bit locations */
#define GT64_DMACTL_DMAREQSRC	0x10000000
#define GT64_DMACTL_RLP		0x02000000
#define GT64_DMACTL_DLP		0x00800000
#define GT64_DMACTL_SLP		0x00200000
#define GT64_DMACTL_ABR		0x00100000
#define GT64_DMACTL_EOTIE	0x00080000
#define GT64_DMACTL_EOTE	0x00040000
#define GT64_DMACTL_CDE		0x00020000
#define GT64_DMACTL_MDREQ	0x00010000
#define GT64_DMACTL_SDA		0x00008000
#define GT64_DMACTL_DMAACTST	0x00004000
#define GT64_DMACTL_FETNEXREC	0x00002000
#define GT64_DMACTL_CHANEN	0x00001000
#define GT64_DMACTL_TRANSMOD	0x00000800
#define GT64_DMACTL_INTMODE	0x00000400
#define GT64_DMACTL_CHAINMOD	0x00000200
#define GT64_DMACTL_RDWRFLY	0x00000002
#define GT64_DMACTL_FLYBYEN	0x00000001

#define GT64_DMACTL_DATTRANSLIM1	0x00000140
#define GT64_DMACTL_DATTRANSLIM2	0x00000180
#define GT64_DMACTL_DATTRANSLIM4	0x00000080
#define GT64_DMACTL_DATTRANSLIM8	0x00000000
#define GT64_DMACTL_DATTRANSLIM16	0x00000040
#define GT64_DMACTL_DATTRANSLIM32	0x000000c0
#define GT64_DMACTL_DATTRANSLIM64	0x000001c0

#define GT64_DMACTL_DESTDIRINC	0x00000000
#define GT64_DMACTL_DESTDIRDEC	0x00000010
#define GT64_DMACTL_DESTDIRHOLD	0x00000020

#define GT64_DMACTL_SRCDIRINC	0x00000000
#define GT64_DMACTL_SRCDIRDEC	0x00000004
#define GT64_DMACTL_SRCDIRHOLD	0x00000008

/* GT-64131 DMA Arbiter register addresses */
#define GT64_DMAA		0x860

/* GT-64131 Timer/Counter register addresses */
#define GT64_TC(tc)		(0x850+(tc)*4)
#define GT64_TCC		0x864

/* GT-64131 PCI Internal register addresses */
#define GT64_PCICMD		0xc00
#define GT64_PCITOR		0xc04
#define GT64_PCISCS10BS		0xc08
#define GT64_PCISCS32BS		0xc0c
#define GT64_PCICS20BS		0xc10
#define GT64_PCICS3BOOTCSBS	0xc14
#define GT64_PCIBARE		0xc3c
#define GT64_PCIPMBS		0xc40
#define GT64_PCISCS10BAR	0xc48
#define GT64_PCISCS32BAR	0xc4c
#define GT64_PCICS20BAR		0xc50
#define GT64_PCICS3BOOTCSAR	0xc54
#define GT64_PCISSCS32BAR	0xc5c
#define GT64_PCISCS3BOOTCSBAR	0xc64
#define GT64_PCICA		0xcf8
#define GT64_PCICDR		0xcfc
#define GT64_PCIIAV		0xc34

/* GT-64131 Interrupts register addresses */
#define GT64_ICR		0xc18
#define GT64_CPUIMR		0xc1c
#define GT64_PCIIMR		0xc24
#define GT64_PCISERRM		0xc28

#define GT64_ICR_DMACOMP(c)	(0x00000010<<(c))

#include <linux/types.h> /* u32 */

#include "mdma.h"

/**
 * gt64_desc - GT-64131 DMA descriptor
 * @bytes: number of bytes to transfer
 * @src_addr: source address
 * @dest_addr: destination address
 * @next_desc: address of the next descriptor
 **/
struct gt64_desc {
	u32 bytes;
	u32 src_addr;
	u32 dest_addr;
	u32 next_desc;
};

/* External variables */

extern struct master_dma_operations gt64_dma_ops;

/* External function prototypes */

u32 gt64_head_desc_bus_addr (struct master_dma *dma);
void gt64_reset (struct master_dma *dma);

#endif

