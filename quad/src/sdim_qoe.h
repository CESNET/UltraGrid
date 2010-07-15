/* sdim_qoe.h
 *
 * Header file for the Linear Systems Ltd. SDI Master Q/o.
 *
 * Copyright (C) 2007-2009 Linear Systems Ltd.
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

#ifndef _SDIM_QOE_H
#define _SDIM_QOE_H

#define SDIM_PCI_DEVICE_ID_LINSYS_SDIQOE 0x00b4
#define SDIM_NAME_QOE	"SDI Master Q/o PCIe"

#define SDIM_QOE_TFSL		0x400 /* FIFO start level */

/* Register addresses */
#define SDIM_QOE_FIFO(c)	((c)*0x100+0x000) /* FIFO */
#define SDIM_QOE_DMATLR(c)	((c)*0x100+0x008) /* Receive FIFO control */
#define SDIM_QOE_FSR(c)		((c)*0x100+0x00c) /* FIFO Status */
#define SDIM_QOE_ICSR(c)	((c)*0x100+0x010) /* Interrupt. Control/Status */
#define SDIM_QOE_BCOUNTR(c)	((c)*0x100+0x014) /* Receive Byte Counter */

#define SDIM_QOE_TXBCOUNT(c)	((c)*0x100+0x088) /* Transmit Byte Counter */
#define SDIM_QOE_TCSR(c)	((c)*0x100+0x08c) /* Transmit Control and Status */
#define SDIM_QOE_TFCR(c)	((c)*0x100+0x098) /* Transmit FIFO Control */
#define SDIM_QOE_FPGAID		0x400 /* FPGA ID */
#define SDIM_QOE_CSR		0x404 /* CSR */
#define SDIM_QOE_27COUNTR	0x408 /* 27 MHz Counter */
#define SDIM_QOE_SSN_HI		0x40c /* Silicon serial number, High */
#define SDIM_QOE_SSN_LO		0x410 /* Silicon serial number, low */
#define SDIM_QOE_JTAGR		0x414 /* JTAG */

/* Transmitter Control/Status Register (TCSR) bit locations */
#define SDIM_QOE_TCSR_TXMODE	0x00000003 /* Tx Mode */
#define SDIM_QOE_TCSR_TXE	0x00000010 /* Tx Enable */
#define SDIM_QOE_TCSR_TXRST	0x00000020 /* Tx Reset */
#define SDIM_QOE_TCSR_RP178	0x00004000 /* RP 178 Pattern generation bit */

/* Interrupt Control/Status Register (ICSR) bit locations */
#define SDIM_QOE_ICSR_TUIE	0x00000001 /* Transmit Underrun Interrupt Enable */
#define SDIM_QOE_ICSR_TXDIE	0x00000040 /* Transmit Data Interrupt Enable */
#define SDIM_QOE_ICSR_TU	0x00000100 /* Transmit Underrun Status */
#define SDIM_QOE_ICSR_TXD	0x00004000 /* Bit goes high when 1st byte of data is transmitted */
#define SDIM_QOE_ICSR_TUIS	0x00010000 /* Transmitter Underrun Interrupt Status */
#define SDIM_QOE_ICSR_TXDIS	0x00400000 /* Transmit Data Interrupt Status */
#define SDIM_QOE_ICSR_PMS	0x04000000 /* Packet Maturity Status */
#define SDIM_QOE_ICSR_NOSIG	0x08000000 /* No Signal */

/* Interrupt Control/Status Register bitmasks */

#define SDIM_QOE_ICSR_ISMASK	0x007f0000 /* Interrupt Status Bitmask */
#define SDIM_QOE_ICSR_IEMASK	0x0000007f /* Interrupt Enable Bitmask */

/* Control/Status Register bitmasks */
#define SDIM_QOE_CSR_PAL	0x00000004
#define SDIM_QOE_CSR_EXTCLK	0x00000040

#endif

