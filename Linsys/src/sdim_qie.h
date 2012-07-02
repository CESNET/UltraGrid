/* sdim_qie.h
 *
 * Header file for the Linear Systems Ltd. SDI Master Q/i card.
 *
 * Copyright (C) 2007-2008 Linear Systems Ltd.
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

#ifndef _SDIM_QIE_H
#define _SDIM_QIE_H

#define SDIM_PCI_DEVICE_ID_LINSYS_SDIQIE 0x00a7
#define SDIM_NAME_QIE "SDI Master Q/i PCIe"

#define SDIM_PCI_DEVICE_ID_LINSYS_SDIQI 0x00b1
#define SDIM_NAME_QI "SDI Master Q/i"

/* SDI Master Q/i configuration */
#define SDIM_QIE_RDMATL		0x010 /* Receiver DMA Trigger Level */

/* Register addresses */
#define SDIM_QIE_FIFO(c)	((c)*0x100+0x000) /* FIFO */
#define SDIM_QIE_RCR(c)		((c)*0x100+0x004) /* Receiver Control */
#define SDIM_QIE_RDMATLR(c)	((c)*0x100+0x008) /* DMA Trigger Level */
#define SDIM_QIE_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define SDIM_QIE_ICSR(c)	((c)*0x100+0x010) /* Int. Control/Status */
#define SDIM_QIE_PFLUTAR(c)	((c)*0x100+0x018) /* PID Filter LUT Address */
#define SDIM_QIE_PFLUTR(c)	((c)*0x100+0x01c) /* PID Filter LUT Data */
#define SDIM_QIE_FPGAID		0x400 /* FPGA ID */
#define SDIM_QIE_CSR		0x404 /* Control/Status */
#define SDIM_QIE_27COUNTR	0x408 /* 27 MHz Counter */
#define SDIM_QIE_UIDR_HI	0x40c /* Unique ID, High Dword */
#define SDIM_QIE_UIDR_LO	0x410 /* Unique ID, Low Dword */
#define SDIM_QIE_ASMIR		0x414 /* ASMI */

/* Interrupt Control/Status Register bit locations */
#define SDIM_QIE_ICSR_RX204	0x01000000 /* Rx 204-byte packets */
#define SDIM_QIE_ICSR_RXDIS	0x00100000 /* Rx Data Int. Status */
#define SDIM_QIE_ICSR_RXCDIS	0x00080000 /* Rx Carrier Detect Int. Status */
#define SDIM_QIE_ICSR_RXOIS	0x00010000 /* Rx FIFO Overrun Int. Status */
#define SDIM_QIE_ICSR_RXD	0x00001000 /* Rx Data */
#define SDIM_QIE_ICSR_RXCD	0x00000800 /* Rx Carrier Detect Status */
#define SDIM_QIE_ICSR_RXPASSING	0x00000200 /* Rx Passing Data Status */
#define SDIM_QIE_ICSR_RXO	0x00000100 /* Rx FIFO Overrun Status */
#define SDIM_QIE_ICSR_RXDIE	0x00000010 /* Rx Data Int. Enable */
#define SDIM_QIE_ICSR_RXCDIE	0x00000008 /* Rx Carrier Detect Int. Enable */
#define SDIM_QIE_ICSR_RXOIE	0x00000001 /* Rx FIFO Overrun Int. Enable */

/* Receive Control/Status Register bit locations */
#define SDIM_QIE_RCSR_RST	0x00000020 /* Reset */
#define SDIM_QIE_RCSR_EN	0x00000010 /* Enable */
#define SDIM_QIE_RCSR_10BIT	0x00000001 /* 10-bit data */

#endif

