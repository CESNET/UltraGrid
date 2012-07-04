/* dvbm_lpfd.h
 *
 * Header file for the Linear Systems Ltd. DVB Master LP FD,
 * DVB Master III Tx LP PCIe, and DVB Master III Rx LP PCIe.
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

#ifndef _DVBM_LPFD_H
#define _DVBM_LPFD_H

#include <linux/fs.h> /* file_operations */
#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#include "miface.h"

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFD 0x0075
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPFDE 0x008f
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPTXE 0x00c0
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPRXE 0x00bf
#define DVBM_NAME_LPFD "DVB Master LP"
#define DVBM_NAME_LPFDE "DVB Master LP PCIe"
#define DVBM_NAME_LPTXE "DVB Master III Tx LP PCIe"
#define DVBM_NAME_LPRXE "DVB Master III Rx LP PCIe"

/* DVB Master LP FD configuration */
#define DVBM_LPFD_TFSL			0x200 /* Transmit FIFO Start Level */

/* Register addresses */
#define DVBM_LPFD_FIFO			0x04 /* FIFO */
#define DVBM_LPFD_FTR			0x0c /* Finetuning */
#define DVBM_LPFD_FSR			0x14 /* FIFO Status */
#define DVBM_LPFD_ICSR			0x18 /* Interrupt Control/Status */
#define DVBM_LPFD_TXBCOUNTR		0x20 /* Transmit Byte Count */
#define DVBM_LPFD_RXBCOUNTR		0x24 /* Receive Byte Count */
#define DVBM_LPFD_PFLUTAR		0x28 /* PID Filter Lookup Table Address */
#define DVBM_LPFD_PFLUTR		0x2c /* PID Filter Lookup Table Data */
#define DVBM_LPFD_27COUNTR		0x50 /* 27 MHz Counter */
#define DVBM_LPFD_CSR			0x54 /* Control/Status */
#define DVBM_LPFD_TCSR			0x58 /* Transmit Control/Status */
#define DVBM_LPFD_RCSR			0x5c /* Receive Control/Status */
#define DVBM_LPFD_IBSTR			0x60 /* Interbyte Stuffing */
#define DVBM_LPFD_IPSTR			0x64 /* Interpacket Stuffing */
#define DVBM_LPFD_TFCR			0x68 /* Transmit FIFO Control */
#define DVBM_LPFD_RFCR			0x6c /* Receive FIFO Control */
#define DVBM_LPFD_JTAGR			0x70 /* JTAG */
#define DVBM_LPFD_WDTLR			0x74 /* Watchdog Timer Limit */
#define DVBM_LPFD_UIDR_HI		0x78 /* Unique ID, High Dword */
#define DVBM_LPFD_UIDR_LO		0x7c /* Unique ID, Low Dword */

/* Finetuning Register bit locations */
#define DVBM_LPFD_FTR_ILBIG_MASK		0x0f000000
#define DVBM_LPFD_FTR_ILBIG_SHIFT		24
#define DVBM_LPFD_FTR_BIGIP_MASK		0x00ff0000
#define DVBM_LPFD_FTR_BIGIP_SHIFT		16
#define DVBM_LPFD_FTR_ILNORMAL_MASK		0x00000f00
#define DVBM_LPFD_FTR_ILNORMAL_SHIFT	8
#define DVBM_LPFD_FTR_NORMALIP_MASK		0x000000ff

/* Interrupt Control/Status Register bit locations */
#define DVBM_LPFD_ICSR_NOSIG	0x08000000 /* Tx Ref. Status */
#define DVBM_LPFD_ICSR_RX204	0x01000000 /* Rx 204-byte packets */
#define DVBM_LPFD_ICSR_TXDIS	0x00400000 /* Tx Data Int. Status */
#define DVBM_LPFD_ICSR_RXDIS	0x00200000 /* Rx Data Int. Status */
#define DVBM_LPFD_ICSR_RXCDIS	0x00100000 /* Rx Carrier Detect Int. Status */
#define DVBM_LPFD_ICSR_RXAOSIS	0x00080000 /* Rx Acq. of Sync. Int. Status */
#define DVBM_LPFD_ICSR_RXLOSIS	0x00040000 /* Rx Loss of Sync. Int. Status */
#define DVBM_LPFD_ICSR_RXOIS	0x00020000 /* Rx FIFO Overrun Int. Status */
#define DVBM_LPFD_ICSR_TXUIS	0x00010000 /* Tx FIFO Underrun Int. Status */
#define DVBM_LPFD_ICSR_BYPASS	0x00008000 /* Bypass Status */
#define DVBM_LPFD_ICSR_TXD		0x00004000 /* Tx Data */
#define DVBM_LPFD_ICSR_RXD		0x00002000 /* Rx Data */
#define DVBM_LPFD_ICSR_RXCD		0x00001000 /* Rx Carrier Detect Status */
#define DVBM_LPFD_ICSR_RXPASSING	0x00000400 /* Rx Passing Data Status */
#define DVBM_LPFD_ICSR_RXO		0x00000200 /* Rx FIFO Overrun Status */
#define DVBM_LPFD_ICSR_TXU		0x00000100 /* Tx FIFO Underrun Status */
#define DVBM_LPFD_ICSR_TXDIE	0x00000040 /* Tx Data Int. Enable */
#define DVBM_LPFD_ICSR_RXDIE	0x00000020 /* Rx Data Int. Enable */
#define DVBM_LPFD_ICSR_RXCDIE	0x00000010 /* Rx Carrier Detect Int. Enable */
#define DVBM_LPFD_ICSR_RXAOSIE	0x00000008 /* Rx Acq. of Sync. Int. Enable */
#define DVBM_LPFD_ICSR_RXLOSIE	0x00000004 /* Rx Loss of Sync. Int. Enable */
#define DVBM_LPFD_ICSR_RXOIE	0x00000002 /* Rx FIFO Overrun Int. Enable */
#define DVBM_LPFD_ICSR_TXUIE	0x00000001 /* Tx FIFO Underrun Int. Enable */

#define DVBM_LPFD_ICSR_TXCTRL_MASK	0x00000041
#define DVBM_LPFD_ICSR_TXSTAT_MASK	0x08414100

#define DVBM_LPFD_ICSR_RXCTRL_MASK	0x0000003e
#define DVBM_LPFD_ICSR_RXSTAT_MASK	0x113e3600

/* Control/Status Register bitmasks */
#define DVBM_LPFD_CSR_BYPASS_MASK	0x00000003 /* Bypass Control */

/* Transmit Control/Status Register bit locations */
#define DVBM_LPFD_TCSR_PAL		0x00002000 /* PAL External Clock */
#define DVBM_LPFD_TCSR_PRC		0x00000400 /* Packet Release Control */
#define DVBM_LPFD_TCSR_NP		0x00000200 /* Null Packet Insertion */
#define DVBM_LPFD_TCSR_TSS		0x00000100 /* Timestamp Strip */
#define DVBM_LPFD_TCSR_RXCLK	0x00000080 /* Recovered Rx Clock */
#define DVBM_LPFD_TCSR_EXTCLK	0x00000040 /* External Clock */
#define DVBM_LPFD_TCSR_RST		0x00000020 /* Reset */
#define DVBM_LPFD_TCSR_EN		0x00000010 /* Enable */
#define DVBM_LPFD_TCSR_MAKE204	0x00000002 /* Make 204 */
#define DVBM_LPFD_TCSR_204		0x00000001 /* 204-byte packets */

/* Transmit Control/Status Register bitmasks */
#define DVBM_LPFD_TCSR_CLK_MASK		0x000020c0
#define DVBM_LPFD_TCSR_MODE_MASK	0x00000003

/* Receive Control/Status Register bit locations */
#define DVBM_LPFD_RCSR_NP		0x00002000 /* Null Packet Replacement */
#define DVBM_LPFD_RCSR_PFE		0x00001000 /* PID Filter Enable */
#define DVBM_LPFD_RCSR_PTSE		0x00000200 /* Prepended Timestamp Enable */
#define DVBM_LPFD_RCSR_TSE		0x00000100 /* Timestamp Enable */
#define DVBM_LPFD_RCSR_INVSYNC	0x00000080 /* Inverted Packet Sync. */
#define DVBM_LPFD_RCSR_RF		0x00000040 /* Reframe */
#define DVBM_LPFD_RCSR_RST		0x00000020 /* Reset */
#define DVBM_LPFD_RCSR_EN		0x00000010 /* Enable */
#define DVBM_LPFD_RCSR_RSS		0x00000008 /* Reed-Solomon Strip */

/* Receive Control/Status Register bitmasks */
#define DVBM_LPFD_RCSR_SYNC_MASK	0x00000003

#define DVBM_LPFD_RCSR_AUTO		0x00000003 /* Sync. Auto */
#define DVBM_LPFD_RCSR_204		0x00000002 /* Sync. 204 */
#define DVBM_LPFD_RCSR_188		0x00000001 /* Sync. 188 */

/* External variables */

extern struct file_operations dvbm_lpfd_txfops;
extern struct file_operations dvbm_lpfd_rxfops;
extern struct master_iface_operations dvbm_lpfd_txops;
extern struct master_iface_operations dvbm_lpfd_rxops;

/* External function prototypes */

ssize_t dvbm_lpfd_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t dvbm_lpfd_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
ssize_t dvbm_lpfd_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
int dvbm_lpfd_pci_probe (struct pci_dev *pdev) __devinit;
int dvbm_lptxe_pci_probe (struct pci_dev *pdev) __devinit;
int dvbm_lprxe_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_lpfd_pci_remove (struct pci_dev *pdev);

#define dvbm_lptxe_pci_remove(pdev) dvbm_lpfd_pci_remove(pdev)
#define dvbm_lprxe_pci_remove(pdev) dvbm_lpfd_pci_remove(pdev)

#endif
