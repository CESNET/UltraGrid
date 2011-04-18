/* eeprom.c
 *
 * Functions related to the PCI 9056 serial EEPROM.
 *
 * Copyright (C) 2005, 2009 Linear Systems Ltd.
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

#include <linux/sched.h> /* schedule_timeout () */
#include <linux/delay.h> /* udelay () */

#include <asm/io.h> /* readl () */

#include "plx9080.h"
#include "eeprom.h"

/* Static function prototypes */
static void ee_clk (void __iomem *addr);
static unsigned int ee_instruction (void __iomem *addr,
	unsigned char opcode,
	unsigned short int word);

/**
 * ee_clk - pulse the serial EEPROM clock
 * @addr: address of the serial EEPROM control register
 **/
static void
ee_clk (void __iomem *addr)
{
	unsigned int reg = readl (addr);

	udelay (2);
	writel (reg | PLX_CNTRL_EESK, addr);
	readl (addr); /* Dummy read to flush PCI posted writes */
	udelay (2);
	writel (reg, addr);
	return;
}

/**
 * ee_instruction - send an opcode and address to the EEPROM
 * @addr: address of the serial EEPROM control register
 * @opcode: opcode
 * @word: serial EEPROM word address
 *
 * Returns the contents of the CNTRL register with EECS set
 * and the other EEPROM bits clear.
 **/
static unsigned int
ee_instruction (void __iomem *addr,
	unsigned char opcode,
	unsigned short int word)
{
	unsigned int reg = readl (addr) &
		~(PLX_CNTRL_EESK | PLX_CNTRL_EECS |
		PLX_CNTRL_EEW | PLX_CNTRL_EEDHIZ);
	int i;

	writel (reg, addr);
	readl (addr); /* Dummy read to flush PCI posted writes */
	udelay (1);

	/* Enable EEPROM */
	reg |= PLX_CNTRL_EECS;
	writel (reg, addr);
	ee_clk (addr);

	/* Write Start bit */
	writel (reg | PLX_CNTRL_EEW, addr);
	ee_clk (addr);

	/* Write opcode */
	writel (reg | ((opcode & 0x02) ? PLX_CNTRL_EEW : 0), addr);
	ee_clk (addr);
	writel (reg | ((opcode & 0x01) ? PLX_CNTRL_EEW : 0), addr);
	ee_clk (addr);

	/* Write word address */
	for (i = 0; i < 8; i++) {
		writel (reg | ((word & (0x80 >> i)) ? PLX_CNTRL_EEW : 0), addr);
		ee_clk (addr);
	}

	return reg;
}

/**
 * ee_ewds - erase/write disable
 * @addr: address of the serial EEPROM control register
 **/
void
ee_ewds (void __iomem *addr)
{
	unsigned int reg = ee_instruction (addr, 0x00, 0x3f);

	/* Disable EEPROM */
	reg &= ~PLX_CNTRL_EECS;
	writel (reg, addr);
	ee_clk (addr);

	return;
}

/**
 * ee_ewen - erase/write enable
 * @addr: address of the serial EEPROM control register
 **/
void
ee_ewen (void __iomem *addr)
{
	unsigned int reg = ee_instruction (addr, 0x00, 0xff);

	/* Disable EEPROM */
	reg &= ~PLX_CNTRL_EECS;
	writel (reg, addr);
	ee_clk (addr);

	return;
}

/**
 * ee_read - read
 * @addr: address of the serial EEPROM control register
 * @word: serial EEPROM word address
 **/
unsigned int
ee_read (void __iomem *addr, unsigned short int word)
{
	unsigned int reg = ee_instruction (addr, 0x02, word);
	unsigned int range = 0;
	int i;

	/* Tristate the data pin */
	reg |= PLX_CNTRL_EEDHIZ;
	writel (reg, addr);
	ee_clk (addr);

	/* Read 32 bits */
	for (i = 0; i < 32; i++) {
		range <<= 1;
		range |= ((readl (addr) & PLX_CNTRL_EER) ? 1 : 0);
		ee_clk (addr);
	}

	/* Disable EEPROM */
	reg &= ~(PLX_CNTRL_EECS | PLX_CNTRL_EEDHIZ);
	writel (reg, addr);
	ee_clk (addr);

	return range;
}

/**
 * ee_write - write
 * @addr: address of the serial EEPROM control register
 * @val: value to write
 * @word: serial EEPROM word address
 **/
void
ee_write (void __iomem *addr, unsigned short int val, unsigned short int word)
{
	unsigned int reg = ee_instruction (addr, 0x01, word);
	int i;

	/* Write 16 bits */
	for (i = 0; i < 16; i++) {
		writel (reg | ((val & (0x8000 >> i)) ? PLX_CNTRL_EEW : 0), addr);
		ee_clk (addr);
	}

	/* Tristate the data pin and initiate a programming cycle */
	reg |= PLX_CNTRL_EEDHIZ;
	reg &= ~PLX_CNTRL_EECS;
	writel (reg, addr);
	ee_clk (addr);

	/* Wait for the Ready signal */
	reg |= PLX_CNTRL_EECS;
	writel (reg, addr);
	readl (addr); /* Dummy read to flush PCI posted writes */
	do {
		set_current_state (TASK_UNINTERRUPTIBLE);
		schedule_timeout (1);
	} while (!(readl (addr) & PLX_CNTRL_EER));

	/* Acknowledge the Ready signal */
	reg &= ~PLX_CNTRL_EEDHIZ;
	writel (reg | PLX_CNTRL_EEW, addr);
	ee_clk (addr);

	/* Disable EEPROM */
	reg &= ~PLX_CNTRL_EECS;
	writel (reg, addr);
	ee_clk (addr);

	return;
}

