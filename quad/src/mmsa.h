/* mmsa.h
 *
 * Header file for the Linear Systems Ltd. MultiMaster SDI-R.
 *
 * Copyright (C) 2004-2008 Linear Systems Ltd.
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

#ifndef _MMSA_H
#define _MMSA_H

#define MMSA_PCI_DEVICE_ID_LINSYS 0x006a
#define MMSAE_PCI_DEVICE_ID_LINSYS 0x00a0
#define MMSA_NAME "MultiMaster SDI-R"
#define MMSAE_NAME "MultiMaster SDI-R PCIe"

/* MultiMaster SDI-R configuration */
#define MMSA_RDMATL	0x010
#define MMSA_TDMATL	0x1ef
#define MMSA_TFSL	0x100

/* Register addresses */
#define MMSA_FIFO	0x04 /* FIFO */
#define MMSA_FTR	0x0c /* Finetuning */
#define MMSA_FSR	0x14 /* FIFO Status */
#define MMSA_ICSR	0x18 /* Interrupt Control/Status */
#define MMSA_TXBCOUNTR	0x20 /* Transmit Byte Count */
#define MMSA_27COUNTR	0x50 /* 27 MHz Counter */
#define MMSA_CSR	0x54 /* Control/Status */
#define MMSA_TCSR	0x58 /* Transmit Control/Status */
#define MMSA_RCSR	0x5c /* Receive Control/Status */
#define MMSA_IBSTR	0x60 /* Interbyte Stuffing */
#define MMSA_IPSTR	0x64 /* Interpacket Stuffing */
#define MMSA_TFCR	0x68 /* Transmit FIFO Control */
#define MMSA_RFCR	0x6c /* Receive FIFO Control */
#define MMSA_ASMIR	0x70 /* ASMI */
#define MMSA_UIDR_HI	0x78 /* Unique ID, High Dword */
#define MMSA_UIDR_LO	0x7c /* Unique ID, Low Dword */

/* Interrupt Control/Status Register bit locations */
#define MMSA_ICSR_NOSIG		0x08000000 /* Tx Ref. Status */
#define MMSA_ICSR_TXDIS		0x00400000 /* Tx Data Int. Status */
#define MMSA_ICSR_RXDIS		0x00200000 /* Rx Data Int. Status */
#define MMSA_ICSR_RXCDIS	0x00100000 /* Rx Carrier Detect Int. Status */
#define MMSA_ICSR_RXOIS		0x00020000 /* Rx FIFO Overrun Int. Status */
#define MMSA_ICSR_TXUIS		0x00010000 /* Tx FIFO Underrun Int. Status */
#define MMSA_ICSR_TXD		0x00004000 /* Tx Data */
#define MMSA_ICSR_RXD		0x00002000 /* Rx Data */
#define MMSA_ICSR_RXCD		0x00001000 /* Rx Carrier Detect Status */
#define MMSA_ICSR_RXPASSING	0x00000400 /* Rx Passing Data Status */
#define MMSA_ICSR_RXO		0x00000200 /* Rx FIFO Overrun Status */
#define MMSA_ICSR_TXU		0x00000100 /* Tx FIFO Underrun Status */
#define MMSA_ICSR_TXDIE		0x00000040 /* Tx Data Int. Enable */
#define MMSA_ICSR_RXDIE		0x00000020 /* Rx Data Int. Enable */
#define MMSA_ICSR_RXCDIE	0x00000010 /* Rx Carrier Detect Int. Enable */
#define MMSA_ICSR_RXOIE		0x00000002 /* Rx FIFO Overrun Int. Enable */
#define MMSA_ICSR_TXUIE		0x00000001 /* Tx FIFO Underrun Int. Enable */

#define MMSA_ICSR_TXCTRLMASK	0x00000041
#define MMSA_ICSR_TXSTATMASK	0x08414100
#define MMSA_ICSR_RXCTRLMASK	0x00000032
#define MMSA_ICSR_RXSTATMASK	0x00323600

/* Transmit Control/Status Register bit locations */
#define MMSA_TCSR_PAL		0x00002000 /* PAL External Clock */
#define MMSA_TCSR_PRC		0x00000400 /* Packet Release Control */
#define MMSA_TCSR_NP		0x00000200 /* Null Packet Insertion */
#define MMSA_TCSR_TSS		0x00000100 /* Timestamp Strip */
#define MMSA_TCSR_RXCLK		0x00000080 /* Recovered Rx Clock */
#define MMSA_TCSR_EXTCLK	0x00000040 /* External Clock */
#define MMSA_TCSR_RST		0x00000020 /* Reset */
#define MMSA_TCSR_EN		0x00000010 /* Enable */
#define MMSA_TCSR_MAKE204	0x00000002 /* Make 204 */
#define MMSA_TCSR_204		0x00000001 /* 204-byte packets */

/* Transmit Control/Status Register bitmasks */
#define MMSA_TCSR_CLKMASK	0x000020c0
#define MMSA_TCSR_MODEMASK	0x00000003

/* Receive Control/Status Register bit locations */
#define MMSA_RCSR_RF	0x00000040 /* Reframe */
#define MMSA_RCSR_RST	0x00000020 /* Reset */
#define MMSA_RCSR_EN	0x00000010 /* Enable */
#define MMSA_RCSR_10BIT	0x00000001 /* 10-bit data */

#endif

