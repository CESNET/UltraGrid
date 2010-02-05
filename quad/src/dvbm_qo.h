/* dvbm_qo.h
 *
 * Header file for the Linear Systems Ltd. DVB Master Q/o.
 *
 * Copyright (C) 2006-2008 Linear Systems Ltd.
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

#ifndef _DVBM_QO_H
#define _DVBM_QO_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#include "mdev.h"

#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQO 0x007C
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBQOE 0x0085
#define DVBM_PCI_DEVICE_ID_LINSYS_DVBLPQOE 0x0095

#define DVBM_NAME_QO 		"DVB Master Q/o"
#define DVBM_NAME_QOE 		"DVB Master Q/o PCIe"
#define DVBM_NAME_LPQOE 	"DVB Master Q/o LP PCIe"

#define DVBM_QO_TFSL 		0x200 /* FIFO start level */

/* Register addresses */
#define DVBM_QO_FIFO(c)		((c)*0x100+0x000) /* FIFO */
#define DVBM_QO_CSR(c)		((c)*0x100+0x004) /* Control/Status */
#define DVBM_QO_DMATLR(c)	((c)*0x100+0x008) /* Receive FIFO control */
#define DVBM_QO_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define DVBM_QO_ICSR(c)		((c)*0x100+0x010) /* Int. Control/Status */
#define DVBM_QO_FTREG(c) 	((c)*0x100+0x084) /* Fine Tuning */
#define DVBM_QO_TXBCOUNT(c) 	((c)*0x100+0x088) /* Transmit Byte Counter */
#define DVBM_QO_TCSR(c) 	((c)*0x100+0x08c) /* Transmit Control and Status */
#define DVBM_QO_IBSTREG(c) 	((c)*0x100+0x090) /* Interbyte Stuffing */
#define DVBM_QO_IPSTREG(c) 	((c)*0x100+0x094) /* Interpacket stuffing */
#define DVBM_QO_TFCR(c) 	((c)*0x100+0x098) /* Transmit FIFO Control */
#define DVBM_QO_TPIDR(c) 	((c)*0x100+0x09c) /* Transmit PID */
#define DVBM_QO_TPCRR_HI(c) 	((c)*0x100+0x0a0) /* Transmit PCR, High Dword */
#define DVBM_QO_TPCRR_LO(c) 	((c)*0x100+0x0a4) /* Transmit PCR, Low Dword */
#define DVBM_QO_TSTAMPR_HI(c) 	((c)*0x100+0x0a8) /* Transmit Timestamp, High Dword */
#define DVBM_QO_TSTAMPR_LO(c) 	((c)*0x100+0x0ac) /* Transmit Timestamp, Low Dword */
#define DVBM_QO_FPGAID 		0x400 /* FPGA ID */
#define DVBM_QO_HL2CSR 		0x404 /* HOTLink II Control and status */
#define DVBM_QO_27COUNTR 	0x408 /* 27 MHz Counter */
#define DVBM_QO_SSN_HI 		0x40c /* Silicon serial number, High */
#define DVBM_QO_SSN_LO 		0x410 /* Silicon serial number, low */
#define DVBM_QO_ASMIR 		0x414 /* ASMI */
#define DVBM_QO_JTAGR 		0x414 /* JTAG */

/* Int. Control/Status Register bit locations */
#define DVBM_QO_ICSR_TUIE 	0x00000001 /* Transmit Underrun Interrupt Enable */
#define DVBM_QO_ICSR_TXDIE 	0x00000040 /* Transmit Data Interrupt Enable */
#define DVBM_QO_ICSR_TU 	0x00000100 /* Transmit Underrun Status */
#define DVBM_QO_ICSR_TXD 	0x00004000 /* Bit goes high when 1st byte of data is transmitted */
#define DVBM_QO_ICSR_TUIS 	0x00010000 /* Transmitter Underrun Interrupt Status */
#define DVBM_QO_ICSR_TXDIS 	0x00400000 /* Transmit Data Interrupt Status */
#define DVBM_QO_ICSR_PMS 	0x04000000 /* Packet Maturity Status */
#define DVBM_QO_ICSR_NOSIG 	0x08000000 /* No Signal */

/* Interrupt Control/Status Register bitmasks */
#define DVBM_QO_ICSR_ISMASK 	0x007f0000 /* Interrupt Status BitMask */
#define DVBM_QO_ICSR_IEMASK 	0x0000007f /* Interrupt Enable BitMask */

/* Transmitter Control/Status Register (TCSR) bit locations */
#define DVBM_QO_TCSR_TXMODE 	0x00000003 /* Tx Mode */
#define DVBM_QO_TCSR_TXE 	0x00000010 /* Tx Enable */
#define DVBM_QO_TCSR_TXRST 	0x00000020 /* Tx Reset */
#define DVBM_QO_TCSR_EXTCLK 	0x00000040 /* External Clock */
#define DVBM_QO_TCSR_TTSS 	0x00000100 /* Transmit Timestamp Strip */
#define DVBM_QO_TCSR_TNP 	0x00000200 /* Transmit Null Packet */
#define DVBM_QO_TCSR_TPRC 	0x00000400 /* Transmit Packet Release Control */
#define DVBM_QO_TCSR_PLLFS 	0x00002000 /* PLL Frequency Select */

/* Control/Status Register bitmasks */
#define DVBM_QO_TCSR_204MAKE188	0x00000005 /* Sync. 204 Make 188 */
#define DVBM_QO_TCSR_AUTO	0x2 /* Sync. Auto */
#define DVBM_QO_TCSR_204	0x1 /* Sync. 204 */
#define DVBM_QO_TCSR_188	0x0 /* Sync. 188 */

#define DVBM_QO_FTR_ILBIG_SHIFT		24
#define DVBM_QO_FTR_BIGIP_SHIFT		16
#define DVBM_QO_FTR_ILNORMAL_SHIFT	8

/* External function prototypes */

int dvbm_qo_pci_probe (struct pci_dev *dev) __devinit;
void dvbm_qo_pci_remove (struct master_dev *card);

#endif

