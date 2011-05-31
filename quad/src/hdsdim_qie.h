/* hdsdim_qie.h
 *
 * Header file for the Linear Systems Ltd. QuadPort H/i card.
 *
 * Copyright (C) 2009-2010 Linear Systems Ltd.
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

#ifndef _HDSDIM_QIE_H
#define _HDSDIM_QIE_H

#include <linux/pci.h> /* pci_dev */
#include <linux/init.h> /* __devinit */

#define HDSDIM_PCI_DEVICE_ID_LINSYS_HDSDIQIE	0x00B6
#define HDSDIM_NAME_QIE		"QuadPort H/i"

/* HD-SDI configuration */
#define HDSDIM_QIE_RCSR(c)	((c)*0x100+0x004) /* Receiver Control and Status Register */
#define HDSDIM_QIE_RDMATLR(c)	((c)*0x100+0x008) /* Receiver FIFO Control */
#define HDSDIM_QIE_FSR(c)	((c)*0x100+0x00c) /* FIFO Status */
#define HDSDIM_QIE_ICSR(c)	((c)*0x100+0x010) /* Interrupt Control and Status */

/* Video channel registers */
#define HDSDIM_QIE_YCER(c)	((c)*0x100+0x020) /* Y Channel CRC Error Register */
#define HDSDIM_QIE_CCER(c)	((c)*0x100+0x024) /* C Channel CRC Error Register */
#define HDSDIM_QIE_CFAT(c)	((c)*0x100+0x028) /* Current Frame Arrival Timestamp */

#define HDSDIM_QIE_VB1R(c)	((c)*0x100+0x074) /* Vertical Blanking 1 Register */
#define HDSDIM_QIE_VB2R(c)	((c)*0x100+0x078) /* Vertical Blanking 1 Register */

/* Audio channel registers */
#define HDSDIM_QIE_AG0ERR(c)	((c)*0x100+0x02c) /* Audio Group 0 Error Register */
#define HDSDIM_QIE_AG1ERR(c)	((c)*0x100+0x030) /* Audio Group 1 Error Register */
#define HDSDIM_QIE_AG2ERR(c)	((c)*0x100+0x034) /* Audio Group 2 Error Register */
#define HDSDIM_QIE_AG3ERR(c)	((c)*0x100+0x038) /* Audio Group 3 Error Register */

#define HDSDIM_QIE_AG0CR(c)	((c)*0x100+0x040) /* Audio Group 0 Control Register */
#define HDSDIM_QIE_AG1CR(c)	((c)*0x100+0x04c) /* Audio Group 1 Control Register */
#define HDSDIM_QIE_AG2CR(c)	((c)*0x100+0x058) /* Audio Group 2 Control Register */
#define HDSDIM_QIE_AG3CR(c)	((c)*0x100+0x064) /* Audio Group 3 Control Register */

#define HDSDIM_QIE_AG0DRA(c)	((c)*0x100+0x044) /* Audio Group 0 Delay Register A */
#define HDSDIM_QIE_AG1DRA(c)	((c)*0x100+0x050) /* Audio Group 1 Delay Register A */
#define HDSDIM_QIE_AG2DRA(c)	((c)*0x100+0x05c) /* Audio Group 2 Delay Register A */
#define HDSDIM_QIE_AG3DRA(c)	((c)*0x100+0x068) /* Audio Group 3 Delay Register A */

#define HDSDIM_QIE_AG0DRB(c)	((c)*0x100+0x048) /* Audio Group 0 Delay Register B */
#define HDSDIM_QIE_AG1DRB(c)	((c)*0x100+0x054) /* Audio Group 1 Delay Register B */
#define HDSDIM_QIE_AG2DRB(c)	((c)*0x100+0x060) /* Audio Group 2 Delay Register B */
#define HDSDIM_QIE_AG3DRB(c)	((c)*0x100+0x06c) /* Audio Group 3 Delay Register B */

/* HD-SDI configuration */
#define HDSDIM_QIE_FPGAID	0x800 /* FPGA ID */
#define HDSDIM_QIE_CSR		0x804 /* Channel CSR */
#define HDSDIM_QIE_CNT27	0x808 /* 27 MHz Counter */
#define HDSDIM_QIE_UIDR_HI	0x80c /* Unique ID, High Dword */
#define HDSDIM_QIE_UIDR_LO	0x810 /* Unique ID, Low Dword */
#define HDSDIM_QIE_JTAGR	0x814 /* JTAG */

/* HD-SDI Receiver Control and Status */
#define HDSDIM_QIE_RCSR_RXE		0x00000010 /* Receiver Enable, same location for both audio and video */
#define HDSDIM_QIE_RCSR_RST		0x00000020 /* Receiver Reset, same location for both audio and video */
#define HDSDIM_QIE_RCSR_STD_LOCKED	0x01000000 /* Video Standard Locked */
#define HDSDIM_QIE_RCSR_STD		0xf0000000 /* Video Standard */

/* SMPTE Standard Formats */
#define HDSDIM_QIE_STD_260M_1035i	0x00000000 /* SMPTE 260M 1035i 60 Hz */
#define HDSDIM_QIE_STD_295M_1080i	0x10000000 /* SMPTE 295M 1080i 50 Hz */
#define HDSDIM_QIE_STD_274M_1080i_60HZ	0x20000000 /* SMPTE 274M 1080i 60 Hz or 1080sF 30 Hz*/
#define HDSDIM_QIE_STD_274M_1080i_50HZ	0x30000000 /* SMPTE 274M 1080i 50 Hz or 1080sF 25 Hz */
#define HDSDIM_QIE_STD_274M_1080p_30HZ	0x40000000 /* SMPTE 274M 1080p 30 Hz*/
#define HDSDIM_QIE_STD_274M_1080p_25HZ	0x50000000 /* SMPTE 274M 1080p 25 Hz */
#define HDSDIM_QIE_STD_274M_1080p_24HZ	0x60000000 /* SMPTE 274M 1080p 24 Hz */
#define HDSDIM_QIE_STD_296M_720p_60HZ	0x70000000 /* SMPTE 296M 720p 60 Hz */
#define HDSDIM_QIE_STD_274M_1080sf_24HZ	0x80000000 /* SMPTE 274M 1080sF 24 Hz */
#define HDSDIM_QIE_STD_296M_720p_50HZ	0x90000000 /* SMPTE 296M 720p 50 Hz */
#define HDSDIM_QIE_STD_296M_720p_30HZ	0xa0000000 /* SMPTE 296M 720p 30 Hz */
#define HDSDIM_QIE_STD_296M_720p_25HZ	0xb0000000 /* SMPTE 296M 720p 25 Hz */
#define HDSDIM_QIE_STD_296M_720p_24HZ	0xc0000000 /* SMPTE 296M 720p 24 Hz */

/* HD-SDI RCSR Modes */
#define HDSDIM_QIE_RCSR_MODE_RAW		0x00 /* WYSIWYG Transparent Mode (Raw mode) */
#define HDSDIM_QIE_RCSR_MODE_V210_SYNC		0x01 /* Synchronization Mode */
#define HDSDIM_QIE_RCSR_MODE_V210_DEINTERLACE	0x02 /* De-interlacing mode */
#define HDSDIM_QIE_RCSR_MODE_UYVY_DEINTERLACE	0x03 /* 8-bit De-interlacing mode */

#define HDSDIM_QIE_RCSR_AUDSAMP_SZ_24BIT	(0x00 << 26) /* 24-bit */
#define HDSDIM_QIE_RCSR_AUDSAMP_SZ_16BIT	(0x02 << 26) /* 16-bit */
#define HDSDIM_QIE_RCSR_AUDSAMP_SZ_32BIT	(0x03 << 26) /* 36-bit */

#define HDSDIM_QIE_RCSR_AUDCH_EN_2	(0x00 << 18) /* 2 channel */
#define HDSDIM_QIE_RCSR_AUDCH_EN_4	(0x01 << 18) /* 4 channel */
#define HDSDIM_QIE_RCSR_AUDCH_EN_6	(0x02 << 18) /* 6 channel */
#define HDSDIM_QIE_RCSR_AUDCH_EN_8	(0x03 << 18) /* 8 channel */

/* HD-SDI Receive FIFO Control */
#define HDSDIM_QIE_RDMATL		0x080 /* Receiver DMA Trigger Level */

/* Interrupt Control and Status */
#define HDSDIM_QIE_ICSR_ROIE		0x00000001 /* Receive Overrun Interrupt Enable */
#define HDSDIM_QIE_ICSR_CDIE		0x00000008 /* Carrier Detect Interrupt Enable */
#define HDSDIM_QIE_ICSR_RXDIE		0x00000010 /* Receive Data Interrupt Enable */
#define HDSDIM_QIE_ICSR_RO		0x00000100 /* Receive Overrun Status */
#define HDSDIM_QIE_ICSR_RXPASSING	0x00000200 /* Receive Sync Status */
#define HDSDIM_QIE_ICSR_CD		0x00000800 /* Carrier Detect Status */
#define HDSDIM_QIE_ICSR_RXD		0x00001000 /* Receive Data */
#define HDSDIM_QIE_ICSR_ROIS		0x00010000 /* Receive Overrun Interrupt Status */
#define HDSDIM_QIE_ICSR_CDIS		0x00080000 /* Receive Carrier Detect Interrupt Status */
#define HDSDIM_QIE_ICSR_RXDIS		0x00100000 /* Receive Data Interrupt Status */
#define HDSDIM_QIE_ICSR_LKD		0x01000000 /* Autodetect Locked Status */

/* External function prototypes */

int hdsdim_qie_pci_probe (struct pci_dev *pdev) __devinit;
void hdsdim_qie_pci_remove (struct pci_dev *pdev);

#endif

