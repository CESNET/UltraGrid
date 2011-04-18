/* sdim.h
 *
 * Header file for the Linear Systems Ltd. SDI Master.
 *
 * Copyright (C) 2004-2009 Linear Systems Ltd.
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

#ifndef _SDIM_H
#define _SDIM_H

#define SDIM_PCI_DEVICE_ID_LINSYS 0x006b
#define SDIME_PCI_DEVICE_ID_LINSYS 0x009f
#define SDIM_PCI_DEVICE_ID_LINSYS_SDILPFD 0x0076
#define SDIM_PCI_DEVICE_ID_LINSYS_SDILPFDE 0x0091
#define SDIM_NAME "SDI Master"
#define SDIME_NAME "SDI Master PCIe"
#define SDIM_NAME_LPFD "SDI Master LP"
#define SDIM_NAME_LPFDE "SDI Master LP PCIe"

/* SDI Master configuration */
#define SDIM_RDMATL	0x010
#define SDIM_TDMATL	0x1ef
#define SDIM_TFSL	0x100
#define SDIME_TDMATL	0x7f8
#define SDIME_TFSL	0x400

/* Register addresses */
#define SDIM_FIFO	0x04 /* FIFO */
#define SDIM_FSR	0x14 /* FIFO Status */
#define SDIM_ICSR	0x18 /* Interrupt Control/Status */
#define SDIM_CSR	0x54 /* Control/Status */
#define SDIM_TCSR	0x58 /* Transmit Control/Status */
#define SDIM_RCSR	0x5c /* Receive Control/Status */
#define SDIM_TFCR	0x68 /* Transmit FIFO Control */
#define SDIM_RFCR	0x6c /* Receive FIFO Control */
#define SDIM_ASMIR	0x70 /* ASMI */
#define SDIM_JTAGR	0x70 /* JTAG */
#define SDIM_UIDR_HI	0x78 /* Unique ID, High Dword */
#define SDIM_UIDR_LO	0x7c /* Unique ID, Low Dword */

/* Interrupt Control/Status Register bit locations */
#define SDIM_ICSR_NOSIG		0x08000000 /* Tx Ref. Status */
#define SDIM_ICSR_TXDIS		0x00400000 /* Tx Data Int. Status */
#define SDIM_ICSR_RXDIS		0x00200000 /* Rx Data Int. Status */
#define SDIM_ICSR_RXCDIS	0x00100000 /* Rx Carrier Detect Int. Status */
#define SDIM_ICSR_RXOIS		0x00020000 /* Rx FIFO Overrun Int. Status */
#define SDIM_ICSR_TXUIS		0x00010000 /* Tx FIFO Underrun Int. Status */
#define SDIM_ICSR_TXD		0x00004000 /* Tx Data */
#define SDIM_ICSR_RXD		0x00002000 /* Rx Data */
#define SDIM_ICSR_RXCD		0x00001000 /* Rx Carrier Detect Status */
#define SDIM_ICSR_RXPASSING	0x00000400 /* Rx Passing Data Status */
#define SDIM_ICSR_RXO		0x00000200 /* Rx FIFO Overrun Status */
#define SDIM_ICSR_TXU		0x00000100 /* Tx FIFO Underrun Status */
#define SDIM_ICSR_TXDIE		0x00000040 /* Tx Data Int. Enable */
#define SDIM_ICSR_RXDIE		0x00000020 /* Rx Data Int. Enable */
#define SDIM_ICSR_RXCDIE	0x00000010 /* Rx Carrier Detect Int. Enable */
#define SDIM_ICSR_RXOIE		0x00000002 /* Rx FIFO Overrun Int. Enable */
#define SDIM_ICSR_TXUIE		0x00000001 /* Tx FIFO Underrun Int. Enable */

#define SDIM_ICSR_TXCTRLMASK	0x00000041
#define SDIM_ICSR_TXSTATMASK	0x08414100
#define SDIM_ICSR_RXCTRLMASK	0x00000032
#define SDIM_ICSR_RXSTATMASK	0x00323600

/* Transmit Control/Status Register bit locations */
#define SDIM_TCSR_RP178	0x00004000 /* RP178 SDI Pattern Generation Disable */
#define SDIM_TCSR_PAL		0x00000200 /* PAL External Clock */
#define SDIM_TCSR_RXCLK		0x00000080 /* Recovered Rx Clock */
#define SDIM_TCSR_EXTCLK	0x00000040 /* External Clock */
#define SDIM_TCSR_RST		0x00000020 /* Reset */
#define SDIM_TCSR_EN		0x00000010 /* Enable */
#define SDIM_TCSR_10BIT		0x00000001 /* 10-bit data */

/* Transmit Control/Status Register bitmasks */
#define SDIM_TCSR_CLKMASK	0x000002c0

/* Receive Control/Status Register bit locations */
#define SDIM_RCSR_RST	0x00000020 /* Reset */
#define SDIM_RCSR_EN	0x00000010 /* Enable */
#define SDIM_RCSR_10BIT	0x00000001 /* 10-bit data */

#endif

