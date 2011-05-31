/* dvbm_lpqo.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Q/o LP PCIe.
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

#ifndef _DVBM_LPQO_H
#define _DVBM_LPQO_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE 0x0095
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE_MINIBNC 0x00AC

#define DVBM_NAME_LPQOE		"DVB Master Q/o LP PCIe"
#define DVBM_NAME_LPQOE_MINIBNC	"DVB Master Q/o LP PCIe Mini BNC"

#define DVBM_LPQO_TFSL		0x200 /* FIFO start level */

/* Register addresses */
#define DVBM_LPQO_FIFO(c)	((c)*0x100+0x000) /* FIFO */
#define DVBM_LPQO_DMATLR(c)	((c)*0x100+0x008) /* Receive FIFO control */
#define DVBM_LPQO_FSR(c)	((c)*0x100+0x00c) /* FIFO Status */
#define DVBM_LPQO_ICSR(c)	((c)*0x100+0x010) /* Int. Control/Status */
#define DVBM_LPQO_FTREG(c)	((c)*0x100+0x084) /* Fine Tuning */
#define DVBM_LPQO_TXBCOUNT(c)	((c)*0x100+0x088) /* Transmit Byte Counter */
#define DVBM_LPQO_TCSR(c)	((c)*0x100+0x08c) /* Transmit Control/Status */
#define DVBM_LPQO_IBSTREG(c)	((c)*0x100+0x090) /* Interbyte Stuffing */
#define DVBM_LPQO_IPSTREG(c)	((c)*0x100+0x094) /* Interpacket stuffing */
#define DVBM_LPQO_TFCR(c)	((c)*0x100+0x098) /* Transmit FIFO Control */
#define DVBM_LPQO_TPIDR(c)	((c)*0x100+0x09c) /* Transmit PID */
#define DVBM_LPQO_TPCRR_HI(c)	((c)*0x100+0x0a0) /* Transmit PCR, High Dword */
#define DVBM_LPQO_TPCRR_LO(c)	((c)*0x100+0x0a4) /* Transmit PCR, Low Dword */
#define DVBM_LPQO_TSTAMPR_HI(c)	((c)*0x100+0x0a8) /* Transmit Timestamp, High Dword */
#define DVBM_LPQO_TSTAMPR_LO(c)	((c)*0x100+0x0ac) /* Transmit Timestamp, Low Dword */
#define DVBM_LPQO_FPGAID	0x400 /* FPGA ID */
#define DVBM_LPQO_CSR		0x404 /* Control/Status */
#define DVBM_LPQO_27COUNTR	0x408 /* 27 MHz Counter */
#define DVBM_LPQO_SSN_HI	0x40c /* Silicon serial number, High */
#define DVBM_LPQO_SSN_LO	0x410 /* Silicon serial number, low */
#define DVBM_LPQO_JTAGR		0x414 /* JTAG */

/* Int. Control/Status Register bit locations */
#define DVBM_LPQO_ICSR_TUIE	0x00000001 /* Transmit Underrun Interrupt Enable */
#define DVBM_LPQO_ICSR_TXDIE	0x00000040 /* Transmit Data Interrupt Enable */
#define DVBM_LPQO_ICSR_TU	0x00000100 /* Transmit Underrun Status */
#define DVBM_LPQO_ICSR_TXD	0x00004000 /* Bit goes high when 1st byte of data is transmitted */
#define DVBM_LPQO_ICSR_TUIS	0x00010000 /* Transmitter Underrun Interrupt Status */
#define DVBM_LPQO_ICSR_TXDIS	0x00400000 /* Transmit Data Interrupt Status */
#define DVBM_LPQO_ICSR_PMS	0x04000000 /* Packet Maturity Status */
#define DVBM_LPQO_ICSR_NOSIG	0x08000000 /* No Signal */

/* Interrupt Control/Status Register bitmasks */
#define DVBM_LPQO_ICSR_ISMASK	0x007f0000 /* Interrupt Status BitMask */
#define DVBM_LPQO_ICSR_IEMASK	0x0000007f /* Interrupt Enable BitMask */

/* Transmit Control/Status Register (TCSR) bit locations */
#define DVBM_LPQO_TCSR_TXMODE	0x00000003 /* Tx Mode */
#define DVBM_LPQO_TCSR_TXE	0x00000010 /* Tx Enable */
#define DVBM_LPQO_TCSR_TXRST	0x00000020 /* Tx Reset */
#define DVBM_LPQO_TCSR_TTSS	0x00000100 /* Transmit Timestamp Strip */
#define DVBM_LPQO_TCSR_TNP	0x00000200 /* Transmit Null Packet */
#define DVBM_LPQO_TCSR_TPRC	0x00000400 /* Transmit Packet Release Control */

/* Transmit Control/Status Register bitmasks */
#define DVBM_LPQO_TCSR_204MAKE188	0x00000005 /* Sync. 204 Make 188 */
#define DVBM_LPQO_TCSR_AUTO		0x00000002 /* Sync. Auto */
#define DVBM_LPQO_TCSR_204		0x00000001 /* Sync. 204 */
#define DVBM_LPQO_TCSR_188		0x00000000 /* Sync. 188 */

/* Control/Status Register bit locations */
#define DVBM_LPQO_CSR_PLLFS	0x00000004 /* PLL Frequency Select */
#define DVBM_LPQO_CSR_EXTCLK	0x00000040 /* External Clock */

#define DVBM_LPQO_FTR_ILBIG_SHIFT	24
#define DVBM_LPQO_FTR_BIGIP_SHIFT	16
#define DVBM_LPQO_FTR_ILNORMAL_SHIFT	8

/* External function prototypes */

int dvbm_lpqo_pci_probe (struct pci_dev *pdev) __devinit;
void dvbm_lpqo_pci_remove (struct pci_dev *pdev);

#endif

