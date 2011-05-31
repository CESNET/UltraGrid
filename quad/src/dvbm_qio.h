/* dvbm_qio.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Q/io.
 *
 * Copyright (C) 2007-2010 Linear Systems Ltd.
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

#ifndef _DVBM_QIO_H
#define _DVBM_QIO_H

#include <linux/fs.h> /* file_operations */
#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#include "miface.h"

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3IOE 0x0087
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQ3INOE 0x0088
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQO 0x007C
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE 0x0085

#define DVBM_NAME_Q3IOE "DVB Master Quad-1in3out"
#define DVBM_NAME_Q3INOE "DVB Master Quad-3in1out"
#define DVBM_NAME_QO "DVB Master Q/o"
#define DVBM_NAME_QOE "DVB Master Q/o PCIe"

/* DVB Master Q/io configuration */
#define DVBM_QIO_TFSL	0x200 /* Transmit FIFO Start Level */
#define DVBM_QIO_RDMATL	0x020 /* Receiver DMA Trigger Level */
#define DVBM_QIO_TDMATL	0xfdf /* Transmit DMA Trigger Level */

/* Register addresses */
#define DVBM_QIO_FPGAID		0x400 /* FPGA ID */
#define DVBM_QIO_CSR		0x404 /* Control/Status */
#define DVBM_QIO_27COUNTR	0x408 /* 27 MHz Counter */
#define DVBM_QIO_SSN_HI		0x40C /* Silicon serial number, High */
#define DVBM_QIO_SSN_LO		0x410 /* Silicon serial number, low */
#define DVBM_QIO_ASMIR		0x414 /* ASMI */

/* Common per-channel registers */
#define DVBM_QIO_ICSR(c)	((c)*0x100+0x010) /* Interrupt Control/Status */

/* Receiver Registers */
#define DVBM_QIO_RCSR(c)	((c)*0x100+0x004) /* Receive Control/Status */
#define DVBM_QIO_RFCR(c)	((c)*0x100+0x008) /* Receive FIFO Control */
#define DVBM_QIO_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define DVBM_QIO_RXBCOUNT(c)	((c)*0x100+0x014) /* Receive Byte Counter */
#define DVBM_QIO_PFLUTWA(c)	((c)*0x100+0x018) /* PID Filter LUT Address */
#define DVBM_QIO_PFLUT(c)	((c)*0x100+0x01c) /* PID Filter LUT Data */
#define DVBM_QIO_PID0(c)	((c)*0x100+0x020) /* PID Detector Value 0 */
#define DVBM_QIO_PIDCOUNT0(c)	((c)*0x100+0x024) /* PID Count 0 */
#define DVBM_QIO_PID1(c)	((c)*0x100+0x028) /* PID Detector Value 1 */
#define DVBM_QIO_PIDCOUNT1(c)	((c)*0x100+0x02c) /* PID Count 1 */
#define DVBM_QIO_PID2(c)	((c)*0x100+0x030) /* PID Detector Value 2 */
#define DVBM_QIO_PIDCOUNT2(c)	((c)*0x100+0x034) /* PID Count 3 */
#define DVBM_QIO_PID3(c)	((c)*0x100+0x038) /* PID Detector Value 3 */
#define DVBM_QIO_PIDCOUNT3(c)	((c)*0x100+0x03c) /* PID Count 4 */

/* Transmit Registers */
#define DVBM_QIO_FTREG(c)	((c)*0x100+0x084) /* Finetuning */
#define DVBM_QIO_TXBCOUNT(c)	((c)*0x100+0x088) /* Transmit Byte Counter */
#define DVBM_QIO_TCSR(c)	((c)*0x100+0x08c) /* Transmit Control/Status */
#define DVBM_QIO_IBSTREG(c)	((c)*0x100+0x090) /* Interbyte Stuffing */
#define DVBM_QIO_IPSTREG(c)	((c)*0x100+0x094) /* Interpacket Stuffing */
#define DVBM_QIO_TFCR(c)	((c)*0x100+0x098) /* Transmit FIFO Control */
#define DVBM_QIO_TPIDR(c)	((c)*0x100+0x09c) /* Transmit PID */
#define DVBM_QIO_TPCRR_HI(c)	((c)*0x100+0x0a0) /* Transmit PCR, High Word */
#define DVBM_QIO_TPCRR_LO(c)	((c)*0x100+0x0a4) /* Transmit PCR, Low Word */
#define DVBM_QIO_TSTAMPR_HI(c)	((c)*0x100+0x0a8) /* Transmit Timestamp, High Word */
#define DVBM_QIO_TSTAMPR_LO(c)	((c)*0x100+0x0ac) /* Transmit Timestamp, Low Word */

/* Control/Status Register bit locations */
#define DVBM_QIO_CSR_PLLFS	0x00000004 /* PLL Frequency Select */

/* Receiver Control/Status Register (RCSR) bit locations */
#define DVBM_QIO_RCSR_RXMODE	0x00000003 /* Rx Mode */
#define DVBM_QIO_RCSR_RSS	0x00000008 /* Reed-Solomon Strip */
#define DVBM_QIO_RCSR_RXE	0x00000010 /* Receiver Enable */
#define DVBM_QIO_RCSR_RXRST	0x00000020 /* Receiver Reset */
#define DVBM_QIO_RCSR_INVSYNC	0x00000080 /* Inverted Sync Byte Enable */
#define DVBM_QIO_RCSR_RNP	0x00002000 /* Null Packet Replacement */
#define DVBM_QIO_RCSR_PFE	0x00001000 /* PID Filter Enable */
#define DVBM_QIO_RCSR_188	0x00000001 /* 188 Byte Packet */
#define DVBM_QIO_RCSR_204	0x00000002 /* 204 Byte Packet */
#define DVBM_QIO_RCSR_AUTO	0x00000003 /* Auto Byte Packet */
#define DVBM_QIO_RCSR_APPEND	0x00000100 /* Appended Timestamp Mode */
#define DVBM_QIO_RCSR_PREPEND	0x00000200 /* Prepended Timestamp Mode */

/* Interrupt Control/Status Register bit locations */
#define DVBM_QIO_ICSR_TXUIE	0x00000001 /* Tx FIFO Underrun Int. Enable */
#define DVBM_QIO_ICSR_TXDIE	0x00000040 /* Tx Data Int. Enable */
#define DVBM_QIO_ICSR_TXU	0x00000100 /* Tx FIFO Underrun Status */
#define DVBM_QIO_ICSR_TXD	0x00004000 /* Tx Data */
#define DVBM_QIO_ICSR_TXUIS	0x00010000 /* Tx FIFO Underrun Int. Status */
#define DVBM_QIO_ICSR_TXDIS	0x00400000 /* Tx Data Interrupt Status */
#define DVBM_QIO_ICSR_PMS	0x04000000 /* Packet Maturity Status */
#define DVBM_QIO_ICSR_NOSIG	0x08000000 /* No Signal */
#define DVBM_QIO_ICSR_RXOIE	0x00000001 /* Rx FIFO Overrun Int. Enable */
#define DVBM_QIO_ICSR_RXLOSIE	0x00000002 /* Rx Loss of Sync. Int. Enable */
#define DVBM_QIO_ICSR_RXAOSIE	0x00000004 /* Rx Acq. of Sync. Int. Enable */
#define DVBM_QIO_ICSR_RXCDIE	0x00000008 /* Rx Carrier Detect Int. Enable */
#define DVBM_QIO_ICSR_RXDIE	0x00000010 /* Rx Data Int. Enable */
#define DVBM_QIO_ICSR_RXO	0x00000100 /* Rx FIFO Overrun Status */
#define DVBM_QIO_ICSR_RXPASSING  0x00000200 /* Rx sync status same as SYNC */
#define DVBM_QIO_ICSR_RXCD	0x00000800 /* Rx Carrier Detect Status */
#define DVBM_QIO_ICSR_RXD	0x00001000 /* Rx Data */
#define DVBM_QIO_ICSR_RXOIS	0x00010000 /* Rx FIFO Overrun Int. Status */
#define DVBM_QIO_ICSR_RXLOSIS	0x00020000 /* Rx Loss of Sync. Int. Status */
#define DVBM_QIO_ICSR_RXAOSIS	0x00040000 /* Rx Acq. of Sync. Int. Status */
#define DVBM_QIO_ICSR_RXCDIS	0x00080000 /* Rx Carrier Detect Int. Status */
#define DVBM_QIO_ICSR_RXDIS	0x00100000 /* Rx Data Int. Status */
#define DVBM_QIO_ICSR_RX204	0x01000000 /* Rx 204-byte packets */

/* Transmit Control/Status Register (TCSR) bit locations */
#define DVBM_QIO_TCSR_188	0x00000000 /* 188 Byte Packet */
#define DVBM_QIO_TCSR_204	0x00000001 /* 204 Byte Packet */
#define DVBM_QIO_TCSR_MAKE204	0x00000002 /* Make 204 */
#define DVBM_QIO_TCSR_TXE	0x00000010 /* Transmit Enable */
#define DVBM_QIO_TCSR_TXRST	0x00000020 /* Transmit Reset */
#define DVBM_QIO_TCSR_TXCS	0x000000c0 /* Transmit Clock Source add two more */
#define DVBM_QIO_TCSR_EXTCLK	0x00000040 /* External clock Blackburst */
#define DVBM_QIO_TCSR_TTSS	0x00000100 /* Transmit Timestamp Strip */
#define DVBM_QIO_TCSR_TNP	0x00000200 /* Transmit Null Packet */
#define DVBM_QIO_TCSR_TPRC	0x00000400 /* Transmit Packet Release */

/* Finetuning Register bit locations */
#define DVBM_QIO_FTR_ILBIG_MASK		0x0f000000
#define DVBM_QIO_FTR_ILBIG_SHIFT	24
#define DVBM_QIO_FTR_BIGIP_MASK		0x00ff0000
#define DVBM_QIO_FTR_BIGIP_SHIFT	16
#define DVBM_QIO_FTR_ILNORMAL_MASK	0x00000f00
#define DVBM_QIO_FTR_ILNORMAL_SHIFT	8
#define DVBM_QIO_FTR_NORMALIP_MASK	0x000000ff

/* Control/Status Register bitmasks */
#define DVBM_QIO_RCSR_SYNC_MASK	0x00000003

/* Transmit Control/Status Register bitmasks */
#define DVBM_QIO_TCSR_CLK_MASK	0x000020c0
#define DVBM_QIO_TCSR_MODE_MASK	0x00000003

/* External variables */

extern struct file_operations dvbm_qio_txfops;
extern struct file_operations dvbm_qio_rxfops;
extern struct master_iface_operations dvbm_qio_txops;
extern struct master_iface_operations dvbm_qio_rxops;

/* External function prototypes */

ssize_t dvbm_qio_show_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	char *buf);
ssize_t dvbm_qio_store_blackburst_type (struct device *dev,
	struct device_attribute *attr,
	const char *buf,
	size_t count);
ssize_t dvbm_qio_show_uid (struct device *dev,
	struct device_attribute *attr,
	char *buf);
int dvbm_q3io_pci_probe (struct pci_dev *pdev) __devinit;
int dvbm_q3ino_pci_probe (struct pci_dev *pdev) __devinit;
int dvbm_qo_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_qio_pci_remove (struct pci_dev *pdev);

#define dvbm_q3io_pci_remove(pdev) dvbm_qio_pci_remove(pdev)
#define dvbm_q3ino_pci_remove(pdev) dvbm_qio_pci_remove(pdev)
#define dvbm_qo_pci_remove(pdev) dvbm_qio_pci_remove(pdev)

#endif

