/* dvbm_qlf.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Q/i RoHS.
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

#ifndef _DVBM_QLF_H
#define _DVBM_QLF_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF 0x0077
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQLF4 0x00B5
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQIE 0x0084
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF 0x00B9
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_MINIBNC 0x00BA
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQLF_HEADER 0x00BB

#define DVBM_NAME_QLF "DVB Master Q/i"
#define DVBM_NAME_QIE "DVB Master Q/i PCIe"
#define DVBM_NAME_LPQLF "DVB Master Q/i PCIe LP"

/* DVB Master Q/i RoHS configuration */
#define DVBM_QLF_RDMATL		0x020 /* Receiver DMA Trigger Level */

/* Register addresses */
//#define DVBM_QLF_FIFO(c)	((c)*0x100+0x000) /* FIFO */
#define DVBM_QLF_RCR(c)		((c)*0x100+0x004) /* Receiver Control */
#define DVBM_QLF_RDMATLR(c)	((c)*0x100+0x008) /* DMA Trigger Level */
#define DVBM_QLF_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define DVBM_QLF_ICSR(c)	((c)*0x100+0x010) /* Int. Control/Status */
#define DVBM_QLF_RXBCOUNTR(c)	((c)*0x100+0x014) /* Byte Count */
#define DVBM_QLF_PFLUTAR(c)	((c)*0x100+0x018) /* PID Filter LUT Address */
#define DVBM_QLF_PFLUTR(c)	((c)*0x100+0x01c) /* PID Filter LUT Data */
#define DVBM_QLF_PIDR0(c)	((c)*0x100+0x020) /* PID 0 */
#define DVBM_QLF_PIDCOUNTR0(c)	((c)*0x100+0x024) /* PID Count 0 */
#define DVBM_QLF_PIDR1(c)	((c)*0x100+0x028) /* PID 1 */
#define DVBM_QLF_PIDCOUNTR1(c)	((c)*0x100+0x02c) /* PID Count 1 */
#define DVBM_QLF_PIDR2(c)	((c)*0x100+0x030) /* PID 2 */
#define DVBM_QLF_PIDCOUNTR2(c)	((c)*0x100+0x034) /* PID Count 2 */
#define DVBM_QLF_PIDR3(c)	((c)*0x100+0x038) /* PID 3 */
#define DVBM_QLF_PIDCOUNTR3(c)	((c)*0x100+0x03c) /* PID Count 3 */
#define DVBM_QLF_FPGAID		0x400 /* FPGA ID */
#define DVBM_QLF_CSR		0x404 /* Control/Status */
#define DVBM_QLF_27COUNTR	0x408 /* 27 MHz Counter */
#define DVBM_QLF_UIDR_HI	0x40c /* Unique ID, High Dword */
#define DVBM_QLF_UIDR_LO	0x410 /* Unique ID, Low Dword */
#define DVBM_QLF_ASMIR		0x414 /* ASMI */
#define DVBM_QLF_JTAGR		0x414 /* JTAG */

/* Control/Status Register bit locations */
#define DVBM_QLF_RCR_RNP	0x00002000 /* Null Packet Replacement */
#define DVBM_QLF_RCR_PFE	0x00001000 /* PID Filter Enable */
#define DVBM_QLF_RCR_PTSE	0x00000200 /* Prepended Timestamp Enable */
#define DVBM_QLF_RCR_TSE	0x00000100 /* Appended Timestamp Enable */
#define DVBM_QLF_RCR_INVSYNC	0x00000080 /* Inverted Packet Sync. */
#define DVBM_QLF_RCR_RST	0x00000020 /* Reset */
#define DVBM_QLF_RCR_EN		0x00000010 /* Enable */
#define DVBM_QLF_RCR_RSS	0x00000008 /* Reed-Solomon Strip */

/* Receive Control/Status Register bitmasks */
#define DVBM_QLF_RCR_SYNC_MASK	0x00000003

#define DVBM_QLF_RCR_AUTO	0x00000003 /* Sync. Auto */
#define DVBM_QLF_RCR_204	0x00000002 /* Sync. 204 */
#define DVBM_QLF_RCR_188	0x00000001 /* Sync. 188 */

/* Interrupt Control/Status Register bit locations */
#define DVBM_QLF_ICSR_RX204	0x01000000 /* Rx 204-byte packets */
#define DVBM_QLF_ICSR_RXDIS	0x00100000 /* Rx Data Int. Status */
#define DVBM_QLF_ICSR_RXCDIS	0x00080000 /* Rx Carrier Detect Int. Status */
#define DVBM_QLF_ICSR_RXAOSIS	0x00040000 /* Rx Acq. of Sync. Int. Status */
#define DVBM_QLF_ICSR_RXLOSIS	0x00020000 /* Rx Loss of Sync. Int. Status */
#define DVBM_QLF_ICSR_RXOIS	0x00010000 /* Rx FIFO Overrun Int. Status */
#define DVBM_QLF_ICSR_RXD	0x00001000 /* Rx Data */
#define DVBM_QLF_ICSR_RXCD	0x00000800 /* Rx Carrier Detect Status */
#define DVBM_QLF_ICSR_RXPASSING	0x00000200 /* Rx Passing Data Status */
#define DVBM_QLF_ICSR_RXO	0x00000100 /* Rx FIFO Overrun Status */
#define DVBM_QLF_ICSR_RXDIE	0x00000010 /* Rx Data Int. Enable */
#define DVBM_QLF_ICSR_RXCDIE	0x00000008 /* Rx Carrier Detect Int. Enable */
#define DVBM_QLF_ICSR_RXAOSIE	0x00000004 /* Rx Acq. of Sync. Int. Enable */
#define DVBM_QLF_ICSR_RXLOSIE	0x00000002 /* Rx Loss of Sync. Int. Enable */
#define DVBM_QLF_ICSR_RXOIE	0x00000001 /* Rx FIFO Overrun Int. Enable */

/* Interrupt Control/Status Register bitmasks */
#define DVBM_QLF_ICSR_ISMASK	0x003f0000
#define DVBM_QLF_ICSR_IEMASK	0x0000003f

/* Control/Status Register bitmasks */
#define DVBM_QLF_CSR_TSCLKSRC_EXT	0x00000001

/* External function prototypes */
int dvbm_qlf_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_qlf_pci_remove (struct pci_dev *pdev);

#endif

