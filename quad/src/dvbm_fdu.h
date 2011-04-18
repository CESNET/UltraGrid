/* dvbm_fdu.h
 *
 * Header file for the Linear Systems Ltd. DVB Master FD-U.
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

#ifndef _DVBM_FDU_H
#define _DVBM_FDU_H

#include <linux/fs.h> /* file_operations */
#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#include "miface.h"

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU 0x0065
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDU_R 0x0066
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBTXU 0x0067
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBRXU 0x0068
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB 0x006d
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDB_R 0x006e
#define DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD 0x0073
#define DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_R 0x0070
#define DVBM_PCI_DEVICE_ID_LINSYS_DVB2FD_RS 0x0072
#define ATSCM_PCI_DEVICE_ID_LINSYS_2FD 0x0074
#define ATSCM_PCI_DEVICE_ID_LINSYS_2FD_R 0x0071
#define ATSCM_PCI_DEVICE_ID_LINSYS_2FD_RS 0x00a3
#define DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE 0x008C
#define DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_R 0x0089
#define DVBM_PCI_DEVICE_ID_LINSYS_DVB2FDE_RS 0x008B
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE 0x0099
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDE_R 0x009A
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB 0x009D
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBFDEB_R 0x009E
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBTXE 0x009B
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBRXE 0x009C
#define ATSCM_PCI_DEVICE_ID_LINSYS_2FDE 0x008D
#define ATSCM_PCI_DEVICE_ID_LINSYS_2FDE_R 0x008A

#define DVBM_NAME_FDU "DVB Master FD-U"
#define DVBM_NAME_FDU_R "DVB Master FD-UR"
#define DVBM_NAME_TXU "DVB Master III Tx"
#define DVBM_NAME_RXU "DVB Master III Rx"
#define DVBM_NAME_FDB "DVB Master FD-B"
#define DVBM_NAME_FDB_R "DVB Master FD-BR"
#define DVBM_NAME_2FD "DVB Master II FD"
#define DVBM_NAME_2FD_R "DVB Master II FD-R"
#define DVBM_NAME_2FD_RS "DVB Master Dual In"
#define ATSCM_NAME_2FD "ATSC Master II FD"
#define ATSCM_NAME_2FD_R "ATSC Master II FD-R"
#define ATSCM_NAME_2FD_RS "ATSC Master II FD-RS"
#define DVBM_NAME_2FDE "DVB Master II FD PCIe"
#define DVBM_NAME_2FDE_R "DVB Master II FD-R PCIe"
#define DVBM_NAME_2FDE_RS "DVB Master Dual In PCIe"
#define DVBM_NAME_FDE "DVB Master FD PCIe"
#define DVBM_NAME_FDE_R "DVB Master FD-R PCIe"
#define DVBM_NAME_FDEB "DVB Master FD-B PCIe"
#define DVBM_NAME_FDEB_R "DVB Master FD-BR PCIe"
#define DVBM_NAME_TXE "DVB Master III Tx PCIe"
#define DVBM_NAME_RXE "DVB Master III Rx PCIe"
#define ATSCM_NAME_2FDE "ATSC Master II FD PCIe"
#define ATSCM_NAME_2FDE_R "ATSC Master II FD-R PCIe"

/* DVB Master FD-U configuration */
#define DVBM_FDU_RDMATL	0x010
#define DVBM_FDU_TDMATL	0x1ef
#define DVBM_FDU_TFSL	0x100

/* Register addresses */
#define DVBM_FDU_FIFO		0x04 /* FIFO */
#define DVBM_FDU_FTR		0x0c /* Finetuning */
#define DVBM_FDU_FSR		0x14 /* FIFO Status */
#define DVBM_FDU_ICSR		0x18 /* Interrupt Control/Status */
#define DVBM_FDU_TXBCOUNTR	0x20 /* Transmit Byte Count */
#define DVBM_FDU_RXBCOUNTR	0x24 /* Receive Byte Count */
#define DVBM_FDU_PFLUTAR	0x28 /* PID Filter Lookup Table Address */
#define DVBM_FDU_PFLUTR		0x2c /* PID Filter Lookup Table Data */
#define DVBM_FDU_PIDR0		0x30 /* PID 0 */
#define DVBM_FDU_PIDCOUNTR0	0x34 /* PID Count 0 */
#define DVBM_FDU_PIDR1		0x38 /* PID 1 */
#define DVBM_FDU_PIDCOUNTR1	0x3c /* PID Count 1 */
#define DVBM_FDU_PIDR2		0x40 /* PID 2 */
#define DVBM_FDU_PIDCOUNTR2	0x44 /* PID Count 2 */
#define DVBM_FDU_PIDR3		0x48 /* PID 3 */
#define DVBM_FDU_PIDCOUNTR3	0x4c /* PID Count 3 */
#define DVBM_FDU_27COUNTR	0x50 /* 27 MHz Counter */
#define DVBM_FDU_CSR		0x54 /* Control/Status */
#define DVBM_FDU_TCSR		0x58 /* Transmit Control/Status */
#define DVBM_FDU_RCSR		0x5c /* Receive Control/Status */
#define DVBM_FDU_IBSTR		0x60 /* Interbyte Stuffing */
#define DVBM_FDU_IPSTR		0x64 /* Interpacket Stuffing */
#define DVBM_FDU_TFCR		0x68 /* Transmit FIFO Control */
#define DVBM_FDU_RFCR		0x6c /* Receive FIFO Control */
#define DVBM_FDU_ASMIR		0x70 /* ASMI */
#define DVBM_FDU_JTAGR		0x70 /* JTAG */
#define DVBM_FDU_WDTLR		0x74 /* Watchdog Timer Limit */
#define DVBM_FDU_UIDR_HI	0x78 /* Unique ID, High Dword */
#define DVBM_FDU_UIDR_LO	0x7c /* Unique ID, Low Dword */
#define DVBM_FDU_TPIDR		0x80 /* Transmit PID */
#define DVBM_FDU_TPCRR_HI	0x84 /* Transmit PCR, High Dword */
#define DVBM_FDU_TPCRR_LO	0x88 /* Transmit PCR, Low Dword */
#define DVBM_FDU_TSTAMPR_HI	0x8c /* Transmit Timestamp, High Dword */
#define DVBM_FDU_TSTAMPR_LO	0x90 /* Transmit Timestamp, Low Dword */

/* Finetuning Register bit locations */
#define DVBM_FDU_FTR_ILBIG_MASK		0x0f000000
#define DVBM_FDU_FTR_ILBIG_SHIFT	24
#define DVBM_FDU_FTR_BIGIP_MASK		0x00ff0000
#define DVBM_FDU_FTR_BIGIP_SHIFT	16
#define DVBM_FDU_FTR_ILNORMAL_MASK	0x00000f00
#define DVBM_FDU_FTR_ILNORMAL_SHIFT	8
#define DVBM_FDU_FTR_NORMALIP_MASK	0x000000ff

/* Interrupt Control/Status Register bit locations */
#define DVBM_FDU_ICSR_RXRS	0x10000000 /* Rx Redundant Sync. Status */
#define DVBM_FDU_ICSR_NOSIG	0x08000000 /* Tx Ref. Status */
#define DVBM_FDU_ICSR_RX204	0x01000000 /* Rx 204-byte packets */
#define DVBM_FDU_ICSR_TXDIS	0x00400000 /* Tx Data Int. Status */
#define DVBM_FDU_ICSR_RXDIS	0x00200000 /* Rx Data Int. Status */
#define DVBM_FDU_ICSR_RXCDIS	0x00100000 /* Rx Carrier Detect Int. Status */
#define DVBM_FDU_ICSR_RXAOSIS	0x00080000 /* Rx Acq. of Sync. Int. Status */
#define DVBM_FDU_ICSR_RXLOSIS	0x00040000 /* Rx Loss of Sync. Int. Status */
#define DVBM_FDU_ICSR_RXOIS	0x00020000 /* Rx FIFO Overrun Int. Status */
#define DVBM_FDU_ICSR_TXUIS	0x00010000 /* Tx FIFO Underrun Int. Status */
#define DVBM_FDU_ICSR_BYPASS	0x00008000 /* Bypass Status */
#define DVBM_FDU_ICSR_TXD	0x00004000 /* Tx Data */
#define DVBM_FDU_ICSR_RXD	0x00002000 /* Rx Data */
#define DVBM_FDU_ICSR_RXCD	0x00001000 /* Rx Carrier Detect Status */
#define DVBM_FDU_ICSR_RXPASSING	0x00000400 /* Rx Passing Data Status */
#define DVBM_FDU_ICSR_RXO	0x00000200 /* Rx FIFO Overrun Status */
#define DVBM_FDU_ICSR_TXU	0x00000100 /* Tx FIFO Underrun Status */
#define DVBM_FDU_ICSR_TXDIE	0x00000040 /* Tx Data Int. Enable */
#define DVBM_FDU_ICSR_RXDIE	0x00000020 /* Rx Data Int. Enable */
#define DVBM_FDU_ICSR_RXCDIE	0x00000010 /* Rx Carrier Detect Int. Enable */
#define DVBM_FDU_ICSR_RXAOSIE	0x00000008 /* Rx Acq. of Sync. Int. Enable */
#define DVBM_FDU_ICSR_RXLOSIE	0x00000004 /* Rx Loss of Sync. Int. Enable */
#define DVBM_FDU_ICSR_RXOIE	0x00000002 /* Rx FIFO Overrun Int. Enable */
#define DVBM_FDU_ICSR_TXUIE	0x00000001 /* Tx FIFO Underrun Int. Enable */
#define DVBM_FDU_ICSR_RXD2	0x20000000 /* Rx Redundant Data */

#define DVBM_FDU_ICSR_TXCTRL_MASK	0x00000041
#define DVBM_FDU_ICSR_TXSTAT_MASK	0x08414100
#define DVBM_FDU_ICSR_RXCTRL_MASK	0x0000003e
#define DVBM_FDU_ICSR_RXSTAT_MASK	0x113e3600

/* Control/Status Register bitmasks */
#define DVBM_FDU_CSR_GPO_MASK		0x00000100 /* General Purpose Outputs */
#define DVBM_FDU_CSR_GPO_SHIFT		8
#define DVBM_FDU_CSR_GPI_MASK		0x000000fc /* General Purpose Inputs */
#define DVBM_FDU_CSR_GPI_SHIFT		2
#define DVBM_FDU_CSR_BYPASS_MASK	0x00000003 /* Bypass Control */

/* Transmit Control/Status Register bit locations */
#define DVBM_FDU_TCSR_PAL	0x00002000 /* PAL External Clock */
#define DVBM_FDU_TCSR_PRC	0x00000400 /* Packet Release Control */
#define DVBM_FDU_TCSR_NP	0x00000200 /* Null Packet Insertion */
#define DVBM_FDU_TCSR_TSS	0x00000100 /* Timestamp Strip */
#define DVBM_FDU_TCSR_RXCLK	0x00000080 /* Recovered Rx Clock */
#define DVBM_FDU_TCSR_EXTCLK	0x00000040 /* External Clock */
#define DVBM_FDU_TCSR_RST	0x00000020 /* Reset */
#define DVBM_FDU_TCSR_EN	0x00000010 /* Enable */
#define DVBM_FDU_TCSR_MAKE204	0x00000002 /* Make 204 */
#define DVBM_FDU_TCSR_204	0x00000001 /* 204-byte packets */

/* Transmit Control/Status Register bitmasks */
#define DVBM_FDU_TCSR_CLK_MASK	0x000020c0
#define DVBM_FDU_TCSR_MODE_MASK	0x00000003

/* Receive Control/Status Register bit locations */
#define DVBM_FDU_RCSR_SEL	0x00004000 /* Redundant Input Select */
#define DVBM_FDU_RCSR_NP	0x00002000 /* Null Packet Replacement */
#define DVBM_FDU_RCSR_PFE	0x00001000 /* PID Filter Enable */
#define DVBM_FDU_RCSR_PTSE	0x00000200 /* Prepended Timestamp Enable */
#define DVBM_FDU_RCSR_TSE	0x00000100 /* Timestamp Enable */
#define DVBM_FDU_RCSR_INVSYNC	0x00000080 /* Inverted Packet Sync. */
#define DVBM_FDU_RCSR_RF	0x00000040 /* Reframe */
#define DVBM_FDU_RCSR_RST	0x00000020 /* Reset */
#define DVBM_FDU_RCSR_EN	0x00000010 /* Enable */
#define DVBM_FDU_RCSR_RSS	0x00000008 /* Reed-Solomon Strip */

/* Receive Control/Status Register bitmasks */
#define DVBM_FDU_RCSR_SYNC_MASK	0x00000003

#define DVBM_FDU_RCSR_AUTO	0x00000003 /* Sync. Auto */
#define DVBM_FDU_RCSR_204	0x00000002 /* Sync. 204 */
#define DVBM_FDU_RCSR_188	0x00000001 /* Sync. 188 */

/* Interpacket Stuffing Register bit locations */
#define DVBM_FDU_IPSTR_DELETE		0x80000000 /* Delete IP Stuffing */
#define DVBM_FDU_IPSTR_CHANGENEXT	0x40000000 /* Change Next */

/* Transmit Timestamp Register bit locations */
#define DVBM_FDU_TSTAMPR_LOCK	0x80000000 /* Timestamp lock */

/* External variables */

extern struct file_operations dvbm_fdu_txfops;
extern struct file_operations dvbm_fdu_rxfops;
extern struct master_iface_operations dvbm_fdu_txops;
extern struct master_iface_operations dvbm_fdu_rxops;

/* External function prototypes */

ssize_t dvbm_fdu_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
int dvbm_fdu_pci_probe (struct pci_dev *pdev) __devinit;
int dvbm_txu_pci_probe (struct pci_dev *pdev) __devinit;
int dvbm_rxu_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_fdu_pci_remove (struct pci_dev *pdev);

#define dvbm_txu_pci_remove(pdev) dvbm_fdu_pci_remove(pdev)
#define dvbm_rxu_pci_remove(pdev) dvbm_fdu_pci_remove(pdev)

#endif

