/* dvbm_qi.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Q/i.
 *
 * Copyright (C) 2003-2010 Linear Systems Ltd.
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

#ifndef _DVBM_QI_H
#define _DVBM_QI_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQI 0x0069
#define DVBM_NAME_DVBQI "DVB Master Q/i"

/* DVB Master Q/i configuration */
#define DVBM_QI_TURNOFF		0x1
#define DVBM_QI_ACCTOFIRST	0x3
#define DVBM_QI_ACCTONEXT	0x1
#define DVBM_QI_ALETOWR		0x3
#define DVBM_QI_WRACTIVE	0x1
#define DVBM_QI_WRHIGH		0x1
#define DVBM_QI_DEVWIDTH	0x2
#define DVBM_QI_BOOTDMAFLYBY	0x1
#define DVBM_QI_DEVLOC		0x0
#define DVBM_QI_DMAFLYBY	0xe
#define DVBM_QI_TIMEOUT0	0xff
#define DVBM_QI_TIMEOUT1	0x07
#define DVBM_QI_RETRYCTR	0x00
#define DVBM_QI_LDA		0x00
#define DVBM_QI_HDA		0xff
#define DVBM_QI_DMATL		0x008

/* Register addresses */
#define DVBM_QI_FIFO(c)		((c)*0x100+0x000) /* FIFO */
#define DVBM_QI_CSR(c)		((c)*0x100+0x004) /* Control/Status */
#define DVBM_QI_DMATLR(c)	((c)*0x100+0x008) /* DMA Trigger Level */
#define DVBM_QI_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define DVBM_QI_ICSR(c)		((c)*0x100+0x010) /* Int. Control/Status */
#define DVBM_QI_BCOUNTR(c)	((c)*0x100+0x014) /* Byte Count */
#define DVBM_QI_PFLUTAR(c)	((c)*0x100+0x018) /* PID Filter LUT Address */
#define DVBM_QI_PFLUTR(c)	((c)*0x100+0x01c) /* PID Filter LUT Data */
#define DVBM_QI_PIDR0(c)	((c)*0x100+0x020) /* PID 0 */
#define DVBM_QI_PIDCOUNTR0(c)	((c)*0x100+0x024) /* PID Count 0 */
#define DVBM_QI_PIDR1(c)	((c)*0x100+0x028) /* PID 1 */
#define DVBM_QI_PIDCOUNTR1(c)	((c)*0x100+0x02c) /* PID Count 1 */
#define DVBM_QI_PIDR2(c)	((c)*0x100+0x030) /* PID 2 */
#define DVBM_QI_PIDCOUNTR2(c)	((c)*0x100+0x034) /* PID Count 2 */
#define DVBM_QI_PIDR3(c)	((c)*0x100+0x038) /* PID 3 */
#define DVBM_QI_PIDCOUNTR3(c)	((c)*0x100+0x03c) /* PID Count 3 */
#define DVBM_QI_DTSR(c)		((c)*0x100+0x040) /* Data Transfer Size */
#define DVBM_QI_FPGAID		0x400 /* FPGA ID */
#define DVBM_QI_HL2CR		0x404 /* HOTLink II Control */
#define DVBM_QI_27COUNTR	0x408 /* 27 MHz Counter */

/* Control/Status Register bit locations */
#define DVBM_QI_CSR_RNP		0x00000200 /* Null Packet Replacement */
#define DVBM_QI_CSR_PTSE	0x00000100 /* Prepended Timestamp Enable */
#define DVBM_QI_CSR_TSE		0x00000080 /* Timestamp Enable */
#define DVBM_QI_CSR_PFE		0x00000040 /* PID Filter Enable */
#define DVBM_QI_CSR_DSYNC	0x00000020 /* Double Packet Sync. */
#define DVBM_QI_CSR_EN		0x00000010 /* Enable */
#define DVBM_QI_CSR_RST		0x00000008 /* Reset */

/* Control/Status Register bitmasks */
#define DVBM_QI_CSR_SYNCMASK	0x00000007

#define DVBM_QI_CSR_204MAKE188	0x00000005 /* Sync. 204 Make 188 */
#define DVBM_QI_CSR_AUTOMAKE188	0x00000004 /* Sync. Auto Make 188 */
#define DVBM_QI_CSR_AUTO	0x00000003 /* Sync. Auto */
#define DVBM_QI_CSR_204		0x00000002 /* Sync. 204 */
#define DVBM_QI_CSR_188		0x00000001 /* Sync. 188 */

/* Int. Control/Status Register bit locations */
#define DVBM_QI_ICSR_204	0x01000000 /* 204-byte packets */
#define DVBM_QI_ICSR_DTIS	0x00200000 /* Data Transfer Int. Status */
#define DVBM_QI_ICSR_DIS	0x00100000 /* Data Int. Status */
#define DVBM_QI_ICSR_CDIS	0x00080000 /* Carrier Detect Int. Status */
#define DVBM_QI_ICSR_AOSIS	0x00040000 /* Acq. of Sync. Int. Status */
#define DVBM_QI_ICSR_LOSIS	0x00020000 /* Loss of Sync. Int. Status */
#define DVBM_QI_ICSR_OIS	0x00010000 /* FIFO Overrun Int. Status */
#define DVBM_QI_ICSR_LFI_N	0x00002000 /* Not Line Fault Indicator */
#define DVBM_QI_ICSR_DATA	0x00001000 /* Data */
#define DVBM_QI_ICSR_CD		0x00000800 /* Carrier Detect Status */
#define DVBM_QI_ICSR_PASSING	0x00000200 /* Passing Data Status */
#define DVBM_QI_ICSR_O		0x00000100 /* FIFO Overrun Status */
#define DVBM_QI_ICSR_DTIE	0x00000020 /* Data Transfer Int. Enable */
#define DVBM_QI_ICSR_DIE	0x00000010 /* Data Int. Enable */
#define DVBM_QI_ICSR_CDIE	0x00000008 /* Carrier Detect Int. Enable */
#define DVBM_QI_ICSR_AOSIE	0x00000004 /* Acq. of Sync. Int. Enable */
#define DVBM_QI_ICSR_LOSIE	0x00000002 /* Loss of Sync. Int. Enable */
#define DVBM_QI_ICSR_OIE	0x00000001 /* FIFO Overrun Int. Enable */

/* Int. Control/Status Register bitmasks */
#define DVBM_QI_ICSR_ISMASK	0x003f0000
#define DVBM_QI_ICSR_IEMASK	0x0000003f

/* HOTLink II Control Register bit locations */
#define DVBM_QI_HL2CR_BOE(c)	(0x00000100<<((c)*2)) /* Receive Enable */
#define DVBM_QI_HL2CR_MDREQ	0x00000080 /* Mask DMA Requests */
#define DVBM_QI_HL2CR_MIE	0x00000020
#define DVBM_QI_HL2CR_LPEN	0x00000010 /* Loop-Back-Enable */
#define DVBM_QI_HL2CR_BISTLE	0x00000008 /* BIST Latch Enable */
#define DVBM_QI_HL2CR_OELE	0x00000004 /* Output Enable Latch Enable */
#define DVBM_QI_HL2CR_RXLE	0x00000002 /* Power-Control Latch Enable */
#define DVBM_QI_HL2CR_TRSTZ_N	0x00000001 /* Not Reset */

/* External function prototypes */

int dvbm_qi_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_qi_pci_remove (struct pci_dev *pdev);

#endif

