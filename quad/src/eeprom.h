/* eeprom.h
 *
 * Header file for eeprom.c.
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

#ifndef _EEPROM_H
#define _EEPROM_H

/* External function prototypes */

void ee_ewds (void __iomem *addr);
void ee_ewen (void __iomem *addr);
unsigned int ee_read (void __iomem *addr, unsigned short int word);
void ee_write (void __iomem *addr, unsigned short int val, unsigned short int word);

#endif

