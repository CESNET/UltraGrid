/* dvbm_qdual.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Quad-2in2out.
 *
 * Copyright (C) 2006-2010 Linear Systems Ltd.
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

#ifndef _DVBM_QDUAL_H
#define _DVBM_QDUAL_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUAL 0x007d
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQDUALE 0x0086
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE 0x0096
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQDUALE_MINIBNC 0x00ad

#define DVBM_NAME_QDUAL "DVB Master Quad-2in2out"
#define DVBM_NAME_QDUALE "DVB Master Quad-2in2out PCIe"
#define DVBM_NAME_LPQDUALE "DVB Master Quad-2in2out LP PCIe"
#define DVBM_NAME_LPQDUALE_MINIBNC "DVB Master Quad-2in2out LP PCIe Mini BNC"

/* DVB Quad-2in2out configuration */
#define DVBM_QDUAL_TFL			0x200 /* Transmit FIFO Start Level */
#define DVBM_QDUAL_RDMATL		0x020 /* Receiver DMA Trigger Level */
#define DVBM_QDUAL_TDMATL		0xfdf /* Transmit DMA Trigger Level */

/* Register addresses */
#define DVBM_QDUAL_FPGAID		0x400 /* FPGA ID */
#define DVBM_QDUAL_HL2CSR		0x404 /* HOTLink II Control and status */
#define DVBM_QDUAL_27COUNTR		0x408 /* 27 MHz Counter */
#define DVBM_QDUAL_SSN_HI		0x40c /* Silicon serial number, High */
#define DVBM_QDUAL_SSN_LO		0x410 /* Silicon serial number, low */
#define DVBM_QDUAL_ASMIR		0x414 /* ASMI */
#define DVBM_QDUAL_JTAGR		0x414 /* JTAG */

/* Receiver Registers */
#define DVBM_QDUAL_RCSR(c)		((c)*0x100+0x004) /* Receive Control/Status */
#define DVBM_QDUAL_RFCR(c)		((c)*0x100+0x008) /* Receive FIFO Control */
#define DVBM_QDUAL_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define DVBM_QDUAL_ICSR(c)		((c)*0x100+0x010) /* Interrupt Control and Status */
#define DVBM_QDUAL_RXBCOUNT(c)		((c)*0x100+0x014) /* Receive Byte Counter */
#define DVBM_QDUAL_PFLUTWA(c)		((c)*0x100+0x018) /* PID Filter LUT Address */
#define DVBM_QDUAL_PFLUT(c)		((c)*0x100+0x01c) /* PID Filter LUT Data */
#define DVBM_QDUAL_PID0(c)		((c)*0x100+0x020) /* PID Detector Value 0 */
#define DVBM_QDUAL_PIDCOUNT0(c)		((c)*0x100+0x024) /* PID Count 0 */
#define DVBM_QDUAL_PID1(c)		((c)*0x100+0x028) /* PID Detector Value 1 */
#define DVBM_QDUAL_PIDCOUNT1(c)		((c)*0x100+0x02c) /* PID Count 1 */
#define DVBM_QDUAL_PID2(c)		((c)*0x100+0x030) /* PID Detector Value 2 */
#define DVBM_QDUAL_PIDCOUNT2(c)		((c)*0x100+0x034) /* PID Count 3 */
#define DVBM_QDUAL_PID3(c)		((c)*0x100+0x038) /* PID Detector Value 3 */
#define DVBM_QDUAL_PIDCOUNT3(c)		((c)*0x100+0x03c) /* PID Count 4 */

/* Transmit Registers */
#define DVBM_QDUAL_FTREG(c)		((c)*0x100+0x084) /* Finetuning */
#define DVBM_QDUAL_TXBCOUNT(c)		((c)*0x100+0x088) /* Transmit Byte Counter */
#define DVBM_QDUAL_TCSR(c)		((c)*0x100+0x08c) /* Transmit Control and Status */
#define DVBM_QDUAL_IBSTREG(c)		((c)*0x100+0x090) /* Interbyte Stuffing */
#define DVBM_QDUAL_IPSTREG(c)		((c)*0x100+0x094) /* Interpacket Stuffing */
#define DVBM_QDUAL_TFCR(c)		((c)*0x100+0x098) /* Transmit FIFO Control */
#define DVBM_QDUAL_TPIDR(c)		((c)*0x100+0x09c) /* Transmit PID */
#define DVBM_QDUAL_TPCRR_HI(c)		((c)*0x100+0x0a0) /* Transmit PCR, High Word */
#define DVBM_QDUAL_TPCRR_LO(c)		((c)*0x100+0x0a4) /* Transmit PCR, Low Word */
#define DVBM_QDUAL_TSTAMPR_HI(c)	((c)*0x100+0x0a8) /* Transmit Timestamp, High Word */
#define DVBM_QDUAL_TSTAMPR_LO(c)	((c)*0x100+0x0ac) /* Transmit Timestamp, Low Word */

/* Control/Status Register bit locations */
#define DVBM_QDUAL_HL2CSR_PLLFS		0x00000004 /* PLL Frequency Select */

/* Receiver Control/Status Register (RCSR) bit locations */
#define DVBM_QDUAL_RCSR_188		0x00000001 /* 188 Byte Packet */
#define DVBM_QDUAL_RCSR_204		0x00000002 /* 204 Byte Packet */
#define DVBM_QDUAL_RCSR_AUTO		0x00000003 /* Auto Byte Packet */
#define DVBM_QDUAL_RCSR_RSS		0x00000008 /* Reed-Solomon Strip */
#define DVBM_QDUAL_RCSR_RXE		0x00000010 /* Receiver Enable */
#define DVBM_QDUAL_RCSR_RXRST		0x00000020 /* Receiver Reset */
#define DVBM_QDUAL_RCSR_INVSYNC		0x00000080 /* Inverted Sync Byte Enable */
#define DVBM_QDUAL_RCSR_TSE		0x00000100 /* Appended Timestamp Enable */
#define DVBM_QDUAL_RCSR_PTSE		0x00000200 /* Prepended Timestamp Enable */
#define DVBM_QDUAL_RCSR_PFE		0x00001000 /* PID Filter Enable */
#define DVBM_QDUAL_RCSR_RNP		0x00002000 /* Null Packet Replacement */

/* Interrupt Control/Status Register bit locations */
#define DVBM_QDUAL_ICSR_TXUIE	0x00000001 /* Tx FIFO Underrun Int. Enable */
#define DVBM_QDUAL_ICSR_TXDIE	0x00000040 /* Tx Data Int. Enable */
#define DVBM_QDUAL_ICSR_TXU	0x00000100 /* Tx FIFO Underrun Status */
#define DVBM_QDUAL_ICSR_TXD	0x00004000 /* Tx Data */
#define DVBM_QDUAL_ICSR_TXUIS	0x00010000 /* Tx FIFO Underrun Int. Status */
#define DVBM_QDUAL_ICSR_TXDIS	0x00400000 /* Tx Data Interrupt Status */
#define DVBM_QDUAL_ICSR_PMS	0x04000000 /* Packet Maturity Status */
#define DVBM_QDUAL_ICSR_NOSIG	0x08000000 /* No Signal */
#define DVBM_QDUAL_ICSR_RXOIE	0x00000001 /* Rx FIFO Overrun Int. Enable */
#define DVBM_QDUAL_ICSR_RXLOSIE	0x00000002 /* Rx Loss of Sync. Int. Enable */
#define DVBM_QDUAL_ICSR_RXAOSIE	0x00000004 /* Rx Acq. of Sync. Int. Enable */
#define DVBM_QDUAL_ICSR_RXCDIE	0x00000008 /* Rx Carrier Detect Int. Enable */
#define DVBM_QDUAL_ICSR_RXDIE	0x00000010 /* Rx Data Int. Enable */
#define DVBM_QDUAL_ICSR_RXO	0x00000100 /* Rx FIFO Overrun Status */
#define DVBM_QDUAL_ICSR_RXPASSING  0x00000200 /* Rx sync status same as SYNC */
#define DVBM_QDUAL_ICSR_RXCD	0x00000800 /* Rx Carrier Detect Status */
#define DVBM_QDUAL_ICSR_RXD	0x00001000 /* Rx Data */
#define DVBM_QDUAL_ICSR_RXOIS	0x00010000 /* Rx FIFO Overrun Int. Status */
#define DVBM_QDUAL_ICSR_RXLOSIS	0x00020000 /* Rx Loss of Sync. Int. Status */
#define DVBM_QDUAL_ICSR_RXAOSIS	0x00040000 /* Rx Acq. of Sync. Int. Status */
#define DVBM_QDUAL_ICSR_RXCDIS	0x00080000 /* Rx Carrier Detect Int. Status */
#define DVBM_QDUAL_ICSR_RXDIS	0x00100000 /* Rx Data Int. Status */
#define DVBM_QDUAL_ICSR_RX204	0x01000000 /* Rx 204-byte packets */

/* Transmit Control and Status (TCSR) */
#define DVBM_QDUAL_TCSR_188	0x00000000 /* 188 Byte Packet */
#define DVBM_QDUAL_TCSR_204	0x00000001 /* 204 Byte Packet */
#define DVBM_QDUAL_TCSR_MAKE204 0x00000002 /* Make 204 */
#define DVBM_QDUAL_TCSR_TXE	0x00000010 /* Transmit Enable */
#define DVBM_QDUAL_TCSR_TXRST	0x00000020 /* Transmit Reset */
#define DVBM_QDUAL_TCSR_EXTCLK	0x00000040 /* External clock Blackburst */
#define DVBM_QDUAL_TCSR_RXCLK	0x00000080 /* Recovered Rx Clock */
#define DVBM_QDUAL_TCSR_EXTCLK2	0x000008c0 /* External Clock 27 MHz */
#define DVBM_QDUAL_TCSR_TTSS	0x00000100 /* Transmit Timestamp Strip */
#define DVBM_QDUAL_TCSR_TNP	0x00000200 /* Transmit Null Packet */
#define DVBM_QDUAL_TCSR_TPRC	0x00000400 /* Transmit Packet Release */

/* Finetuning Register bit locations */
#define DVBM_QDUAL_FTR_ILBIG_MASK		0x0f000000
#define DVBM_QDUAL_FTR_ILBIG_SHIFT		24
#define DVBM_QDUAL_FTR_BIGIP_MASK		0x00ff0000
#define DVBM_QDUAL_FTR_BIGIP_SHIFT		16
#define DVBM_QDUAL_FTR_ILNORMAL_MASK		0x00000f00
#define DVBM_QDUAL_FTR_ILNORMAL_SHIFT		8
#define DVBM_QDUAL_FTR_NORMALIP_MASK		0x000000ff

/* Control/Status Register bitmasks */
#define DVBM_QDUAL_RCSR_SYNC_MASK	0x00000003

/* Transmit Control/Status Register bitmasks */
#define DVBM_QDUAL_TCSR_CLK_MASK	0x0000d8c0
#define DVBM_QDUAL_TCSR_MODE_MASK	0x00000003

/* External function prototypes */
int dvbm_qdual_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_qdual_pci_remove (struct pci_dev *pdev);

#endif

