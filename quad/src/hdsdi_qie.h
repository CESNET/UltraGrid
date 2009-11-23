/* hdsdi_qie.h
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

#ifndef _HDSDI_QI_H
#define _HDSDI_QI_H

#define HDSDI_PCI_DEVICE_ID_LINSYS			0x00B6
#define HDSDI_NAME							"HD-SDI Q/i"

/* HD-SDI configuration */
#define HDSDI_QI_RCSR(c)					((c)*0x100+0x004) /* Receiver Control and Status Register */
#define HDSDI_QI_RDMATLR(c)					((c)*0x100+0x008) /* Receiver FIFO Control */
#define HDSDI_QI_FSR(c)						((c)*0x100+0x00c) /* FIFO Status */
#define HDSDI_QI_ICSR(c)					((c)*0x100+0x010) /* Interrupt Control and Status */
#define HDSDI_QI_CFAT(c)					((c)*0x100+0x028) /* Current Frame Arrival Timestamp */

#define HDSDI_QI_FPGAID						0x400 /* FPGA ID */
#define HDSDI_QI_CSR						0x404 /* Channel CSR */
#define HDSDI_QI_UIDR_HI					0x40c /* Unique ID, High Dword */
#define HDSDI_QI_UIDR_LO					0x410 /* Unique ID, Low Dword */
#define HDSDI_QI_ASMIR 						0x414 /* ASMI */
#define HDSDI_QI_JTAG						0x414 /* JTAG Control */
#define HDSDI_QI_CNT27						0x408 /* 27 MHz Counter */

/* HD-SDI Receiver Control and Status */
#define HDSDI_QI_RCSR_RXE					0x00000010 /* Receiver Enable */
#define HDSDI_QI_RCSR_RST					0x00000020 /* Receiver Reset */
#define HDSDI_QI_RCSR_STD					0xF0000000 /* Video Standard */

/* SMPTE Standard Formats */
#define HDSDI_QI_STD_260M_1035i				0x00000000 /* SMPTE 260M 1035i 30 Hz */
#define HDSDI_QI_STD_295M_1080i				0x10000000 /* SMPTE 295M 1080i 25 Hz */
#define HDSDI_QI_STD_274M_1080i				0x20000000 /* SMPTE 274M 1080i or 1080sF 30 Hz*/
#define HDSDI_QI_STD_274M_1080i_25HZ		0x30000000 /* SMPTE 274M 1080i or 1080sF 25 Hz */
#define HDSDI_QI_STD_274M_1080p				0x40000000 /* SMPTE 274M 1080p 30 Hz*/
#define HDSDI_QI_STD_274M_1080p_25HZ		0x50000000 /* SMPTE 274M 1080p 25 Hz */
#define HDSDI_QI_STD_274M_1080p_24HZ		0x60000000 /* SMPTE 274M 1080p 24 Hz */
#define HDSDI_QI_STD_296M_720p				0x70000000 /* SMPTE 296M 720p 60 Hz */
#define HDSDI_QI_STD_274M_1080sf			0x80000000 /* SMPTE 274M 1080sF 24 Hz */
#define HDSDI_QI_STD_296M_720p_50HZ			0x90000000 /* SMPTE 296M 720p 50 Hz */

/* HD-SDI RCSR Modes */
#define HDSDI_QI_RCSR_MODE_RAW				0x00 /* WYSIWYG Transparent Mode (Raw mode) */
#define HDSDI_QI_RCSR_MODE_SYNC				0x01 /* Synchronization Mode */
#define HDSDI_QI_RCSR_MODE_DEINTERLACE			0x02 /* De-interlacing mode */

/* HD-SDI Receive FIFO Control */
#define HDSDI_QI_RDMATL						0x080 /* Receiver DMA Trigger Level */

/* Interrupt Control and Status */
#define HDSDI_QI_ICSR_ROIE					0x00000001 /* Receive Overrun Interrupt Enable */
#define HDSDI_QI_ICSR_CDIE					0x00000008 /* Carrier Detect Interrupt Enable */
#define HDSDI_QI_ICSR_RXDIE					0x00000010 /* Receive Data Interrupt Enable */
#define HDSDI_QI_ICSR_RO					0x00000100 /* Receive Overrun Status */
#define HDSDI_QI_ICSR_RXPASSING				0x00000200 /* Receive Sync Status */
#define HDSDI_QI_ICSR_CD					0x00000800 /* Carrier Detect Status */
#define HDSDI_QI_ICSR_RXD					0x00001000 /* Receive Data */
#define HDSDI_QI_ICSR_ROIS					0x00010000 /* Receive Overrun Interrupt Status */
#define HDSDI_QI_ICSR_CDIS					0x00080000 /* Receive Carrier Detect Interrupt Status */
#define HDSDI_QI_ICSR_RXDIS					0x00100000 /* Receive Datat Interrupt Status */
#define HDSDI_QI_ICSR_LKD					0x01000000 /* Autodetect Locked Status */

#endif

