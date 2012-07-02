/* mmas.h
 *
 * Header file for the Linear Systems Ltd. MultiMaster SDI-T.
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

#ifndef _MMAS_H
#define _MMAS_H

#define MMAS_PCI_DEVICE_ID_LINSYS 0x006c
#define MMASE_PCI_DEVICE_ID_LINSYS 0x00a1
#define MMAS_NAME "MultiMaster SDI-T"
#define MMASE_NAME "MultiMaster SDI-T PCIe"

/* MultiMaster SDI-T configuration */
#define MMAS_RDMATL	0x010
#define MMAS_TDMATL	0x1ef
#define MMAS_TFSL	0x100

/* Register addresses */
#define MMAS_FIFO	0x04 /* FIFO */
#define MMAS_FSR	0x14 /* FIFO Status */
#define MMAS_ICSR	0x18 /* Interrupt Control/Status */
#define MMAS_RXBCOUNTR	0x24 /* Receive Byte Count */
#define MMAS_PFLUTAR	0x28 /* PID Filter Lookup Table Address */
#define MMAS_PFLUTR	0x2c /* PID Filter Lookup Table Data */
#define MMAS_PIDR0	0x30 /* PID 0 */
#define MMAS_PIDCOUNTR0	0x34 /* PID Count 0 */
#define MMAS_PIDR1	0x38 /* PID 1 */
#define MMAS_PIDCOUNTR1	0x3c /* PID Count 1 */
#define MMAS_PIDR2	0x40 /* PID 2 */
#define MMAS_PIDCOUNTR2	0x44 /* PID Count 2 */
#define MMAS_PIDR3	0x48 /* PID 3 */
#define MMAS_PIDCOUNTR3	0x4c /* PID Count 3 */
#define MMAS_27COUNTR	0x50 /* 27 MHz Counter */
#define MMAS_CSR	0x54 /* Control/Status */
#define MMAS_TCSR	0x58 /* Transmit Control/Status */
#define MMAS_RCSR	0x5c /* Receive Control/Status */
#define MMAS_TFCR	0x68 /* Transmit FIFO Control */
#define MMAS_RFCR	0x6c /* Receive FIFO Control */
#define MMAS_ASMIR	0x70 /* ASMI */
#define MMAS_UIDR_HI	0x78 /* Unique ID, High Dword */
#define MMAS_UIDR_LO	0x7c /* Unique ID, Low Dword */

/* Interrupt Control/Status Register bit locations */
#define MMAS_ICSR_NOSIG		0x08000000 /* Tx Ref. Status */
#define MMAS_ICSR_RX204		0x01000000 /* Rx 204-byte packets */
#define MMAS_ICSR_TXDIS		0x00400000 /* Tx Data Int. Status */
#define MMAS_ICSR_RXDIS		0x00200000 /* Rx Data Int. Status */
#define MMAS_ICSR_RXCDIS	0x00100000 /* Rx Carrier Detect Int. Status */
#define MMAS_ICSR_RXAOSIS	0x00080000 /* Rx Acq. of Sync. Int. Status */
#define MMAS_ICSR_RXLOSIS	0x00040000 /* Rx Loss of Sync. Int. Status */
#define MMAS_ICSR_RXOIS		0x00020000 /* Rx FIFO Overrun Int. Status */
#define MMAS_ICSR_TXUIS		0x00010000 /* Tx FIFO Underrun Int. Status */
#define MMAS_ICSR_TXD		0x00004000 /* Tx Data */
#define MMAS_ICSR_RXD		0x00002000 /* Rx Data */
#define MMAS_ICSR_RXCD		0x00001000 /* Rx Carrier Detect Status */
#define MMAS_ICSR_RXPASSING	0x00000400 /* Rx Passing Data Status */
#define MMAS_ICSR_RXO		0x00000200 /* Rx FIFO Overrun Status */
#define MMAS_ICSR_TXU		0x00000100 /* Tx FIFO Underrun Status */
#define MMAS_ICSR_TXDIE		0x00000040 /* Tx Data Int. Enable */
#define MMAS_ICSR_RXDIE		0x00000020 /* Rx Data Int. Enable */
#define MMAS_ICSR_RXCDIE	0x00000010 /* Rx Carrier Detect Int. Enable */
#define MMAS_ICSR_RXAOSIE	0x00000008 /* Rx Acq. of Sync. Int. Enable */
#define MMAS_ICSR_RXLOSIE	0x00000004 /* Rx Loss of Sync. Int. Enable */
#define MMAS_ICSR_RXOIE		0x00000002 /* Rx FIFO Overrun Int. Enable */
#define MMAS_ICSR_TXUIE		0x00000001 /* Tx FIFO Underrun Int. Enable */

#define MMAS_ICSR_TXCTRLMASK	0x00000041
#define MMAS_ICSR_TXSTATMASK	0x08414100
#define MMAS_ICSR_RXCTRLMASK	0x0000003e
#define MMAS_ICSR_RXSTATMASK	0x013e3600

/* Transmit Control/Status Register bit locations */
#define MMAS_TCSR_PAL		0x00000200 /* PAL External Clock */
#define MMAS_TCSR_RXCLK		0x00000080 /* Recovered Rx Clock */
#define MMAS_TCSR_EXTCLK	0x00000040 /* External Clock */
#define MMAS_TCSR_RST		0x00000020 /* Reset */
#define MMAS_TCSR_EN		0x00000010 /* Enable */
#define MMAS_TCSR_10BIT		0x00000001 /* 10-bit data */

/* Transmit Control/Status Register bitmasks */
#define MMAS_TCSR_CLKMASK	0x000002c0

/* Receive Control/Status Register bit locations */
#define MMAS_RCSR_NP		0x00002000 /* Null Packet Replacement */
#define MMAS_RCSR_PFE		0x00001000 /* PID Filter Enable */
#define MMAS_RCSR_PTSE		0x00000200 /* Prepended Timestamp Enable */
#define MMAS_RCSR_TSE		0x00000100 /* Timestamp Enable */
#define MMAS_RCSR_INVSYNC	0x00000080 /* Inverted Packet Sync. */
#define MMAS_RCSR_RF		0x00000040 /* Reframe */
#define MMAS_RCSR_RST		0x00000020 /* Reset */
#define MMAS_RCSR_EN		0x00000010 /* Enable */
#define MMAS_RCSR_RSS		0x00000008 /* Reed-Solomon Strip */

/* Receive Control/Status Register bitmasks */
#define MMAS_RCSR_SYNCMASK	0x00000003

#define MMAS_RCSR_AUTO	0x00000003 /* Sync. Auto */
#define MMAS_RCSR_204	0x00000002 /* Sync. 204 */
#define MMAS_RCSR_188	0x00000001 /* Sync. 188 */

#endif

