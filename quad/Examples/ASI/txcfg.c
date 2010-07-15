/* txcfg.c
 *
 * Demonstrate DVB ASI transmitter ioctls.
 *
 * Copyright (C) 2000-2010 Linear Systems Ltd. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice,
 *      this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of Linear Systems Ltd. nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LINEAR SYSTEMS LTD. "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL LINEAR SYSTEMS LTD. OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Linear Systems can be contacted at <http://www.linsys.ca/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

#include "asi.h"
#include "master.h"
#include "../pci_ids.h"
#include "../util.h"

#define TMP_BUFLEN 80

static const char *argv0;
static const char progname[] = "txcfg";

static void
show_cap (unsigned int cap)
{
	char str[TMP_BUFLEN] = "[ ] ";

	str[1] = (cap & ASI_CAP_TX_MAKE204) ? 'X' : ' ';
	printf ("%sConversion of 188-byte packets to 204-byte packets\n", str);

	if (cap & ASI_CAP_TX_INTERLEAVING) {
		printf ("[X] Interleaved bitrate finetuning\n");
	} else {
		str[1] = (cap & ASI_CAP_TX_FINETUNING) ? 'X' : ' ';
		printf ("%sBitrate finetuning\n", str);
	}

	str[1] = (cap & ASI_CAP_TX_LARGEIB) ? 'X' : ' ';
	printf ("%sLarge interbyte stuffing\n", str);

	str[1] = (cap & ASI_CAP_TX_BYTECOUNTER) ? 'X' : ' ';
	printf ("%sTransmit byte counter\n", str);

	if (cap & ASI_CAP_TX_BYTESOR27) {
		printf ("[X] Byte counter is switchable to 27 MHz counter\n");
	} else {
		str[1] = (cap & ASI_CAP_TX_27COUNTER) ? 'X' : ' ';
		printf ("%s27 MHz counter\n", str);
	}

	str[1] = (cap & ASI_CAP_TX_SETCLKSRC) ? 'X' : ' ';
	printf ("%sExternal clock reference\n", str);

	str[1] = (cap & ASI_CAP_TX_RXCLKSRC) ? 'X' : ' ';
	printf ("%sRecovered receive clock\n", str);

	str[1] = (cap & ASI_CAP_TX_FIFOUNDERRUN) ? 'X' : ' ';
	printf ("%sOnboard transmit FIFO underrun events\n", str);

	str[1] = (cap & ASI_CAP_TX_DATA) ? 'X' : ' ';
	printf ("%sTransmit data status\n", str);

	str[1] = (cap & ASI_CAP_TX_TIMESTAMPS) ? 'X' : ' ';
	printf ("%sRemoval of packet timestamps\n", str);

	str[1] = (cap & ASI_CAP_TX_PTIMESTAMPS) ? 'X' : ' ';
	printf ("%sScheduled packet release\n", str);

	str[1] = (cap & ASI_CAP_TX_NULLPACKETS) ? 'X' : ' ';
	printf ("%sNull packet insertion\n", str);

	str[1] = (cap & ASI_CAP_TX_PCRSTAMP) ? 'X' : ' ';
	printf ("%sPCR departure timestamping\n", str);

	str[1] = (cap & ASI_CAP_TX_CHANGENEXTIP) ? 'X' : ' ';
	printf ("%sSingle-shot bitrate correction\n", str);

	str[1] = (cap & ASI_CAP_TX_EXTCLKSRC2) ? 'X' : ' ';
	printf ("%sExternal clock reference 2\n", str);

	printf ("\nPress Enter to continue: ");
	fgets (str, TMP_BUFLEN, stdin);
	return;
}

static void
get_events (int fd)
{
	unsigned int val;

	printf ("Getting the transmitter event flags.\n");
	if (ioctl (fd, ASI_IOC_TXGETEVENTS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transmitter event flags");
	} else if (val) {
		if (val & ASI_EVENT_TX_BUFFER) {
			printf ("Driver transmit buffer queue underrun "
				"detected.\n");
		}
		if (val & ASI_EVENT_TX_FIFO) {
			printf ("Onboard transmit FIFO underrun detected.\n");
		}
		if (val & ASI_EVENT_TX_DATA) {
			printf ("Transmit data change detected.\n");
		}
	} else {
		printf ("No transmitter events detected.\n");
	}
	return;
}

static void
get_buflevel (int fd)
{
	int val;

	printf ("Getting the driver transmit buffer queue length.\n");
	if (ioctl (fd, ASI_IOC_TXGETBUFLEVEL, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the driver transmit buffer queue length");
	} else {
		printf ("Driver transmit buffer queue length = %i.\n", val);
	}
	return;
}

static void
set_stuffing (int fd, unsigned int cap)
{
	struct asi_txstuffing stuffing;
	char str[TMP_BUFLEN];

	memset (&stuffing, 0, sizeof (stuffing));
	printf ("Enter the interbyte stuffing (0 - %u): ",
		(cap & ASI_CAP_TX_LARGEIB) ? 0xffff : 0x00ff);
	fgets (str, TMP_BUFLEN, stdin);
	stuffing.ib = strtol (str, NULL, 0);
	printf ("\nEnter the interpacket stuffing (0 - 16777215): ");
	fgets (str, TMP_BUFLEN, stdin);
	stuffing.ip = strtol (str, NULL, 0);
	if (cap & ASI_CAP_TX_FINETUNING) {
		printf ("\nEnter the number of packets "
			"with normal stuffing\n"
			"per finetuning cycle (0 - 255): ");
		fgets (str, TMP_BUFLEN, stdin);
		stuffing.normal_ip = strtol (str, NULL, 0);
		printf ("\nEnter the number of packets "
			"with additional stuffing\n"
			"per finetuning cycle (0 - 255): ");
		fgets (str, TMP_BUFLEN, stdin);
		stuffing.big_ip = strtol (str, NULL, 0);
		if ((cap & ASI_CAP_TX_INTERLEAVING) &&
			(stuffing.normal_ip > 0) &&
			(stuffing.big_ip > 0)) {
			printf ("\nEnter the number of packets "
				"with normal stuffing\n"
				"per interleaved finetuning cycle: ");
			fgets (str, TMP_BUFLEN, stdin);
			stuffing.il_normal = strtol (str, NULL, 0);
			printf ("\nEnter the number of packets "
				"with additional stuffing\n"
				"per interleaved finetuning cycle: ");
			fgets (str, TMP_BUFLEN, stdin);
			stuffing.il_big = strtol (str, NULL, 0);
		}
	}
	printf ("\n");
	printf ("Setting the stuffing parameters.\n");
	if (ioctl (fd, ASI_IOC_TXSETSTUFFING, &stuffing) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to set the stuffing parameters");
	} else {
		printf ("ib = %u, ip = %u",
			stuffing.ib, stuffing.ip);
		if (cap & ASI_CAP_TX_FINETUNING) {
			printf ("\nnormal_ip = %u, big_ip = %u",
				stuffing.normal_ip, stuffing.big_ip);
			if (cap & ASI_CAP_TX_INTERLEAVING) {
				printf ("\nil_normal = %u, il_big = %u",
					stuffing.il_normal, stuffing.il_big);
			}
		}
		printf ("\n");
	}
	return;
}

static void
get_bytecount (int fd)
{
	unsigned int val;

	printf ("Getting the byte counter value.\n");
	if (ioctl (fd, ASI_IOC_TXGETBYTECOUNT, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the byte counter value");
	} else {
		printf ("Byte count = %u.\n", val);
	}
	return;
}

static void
get_27count (int fd)
{
	unsigned int val;

	printf ("Getting the 27 MHz counter value.\n");
	if (ioctl (fd, ASI_IOC_TXGET27COUNT, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the 27 MHz counter value");
	} else {
		printf ("27 MHz count = %u.\n", val);
	}
	return;
}

static void
get_data (int fd)
{
	int val;

	printf ("Getting the transmit data status.\n");
	if (ioctl (fd, ASI_IOC_TXGETTXD, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transmit data status");
	} else if (val) {
		printf ("Data is being transmitted.\n");
	} else {
		printf ("Data is not being transmitted.\n");
	}
	return;
}

static void
set_pid (int fd)
{
	unsigned int val;
	char str[TMP_BUFLEN];

	printf ("Enter PID to watch: ");
	fgets (str, TMP_BUFLEN, stdin);
	val = strtoul (str, NULL, 0);
	printf ("\n");
	if (val <= 0x1fff) {
		printf ("Setting PID.\n");
		if (ioctl (fd, ASI_IOC_TXSETPID, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to set the PID to watch");
		} else {
			printf ("Watching PID %Xh.\n", val);
		}
	} else {
		printf ("Invalid PID.\n");
	}
	return;
}

static void
get_pcrstamp (int fd)
{
	struct asi_pcrstamp pcrstamp;

	printf ("Getting the last PCR departure timestamp.\n");
	if (ioctl (fd, ASI_IOC_TXGETPCRSTAMP, &pcrstamp) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the last PCR departure timestamp");
	} else {
		unsigned char *p = pcrstamp.PCR;
		long long int pcr;

		pcr = *p++;
		pcr <<= 8;
		pcr += *p++;
		pcr <<= 8;
		pcr += *p++;
		pcr <<= 8;
		pcr += *p++;
		pcr <<= 1;
		pcr += *p >> 7;
		pcr *= 300;
		pcr += (*p++ & 0x01) << 8;
		pcr += *p;
		printf ("Last PCR = %llXh.\n", pcr);
		printf ("Last PCR departure timestamp = %llXh.\n",
			pcrstamp.count);
	}
	return;
}

static void
changenextip (int fd)
{
	int val;
	char str[TMP_BUFLEN];

	printf ("Enter the change in interpacket stuffing: ");
	fgets (str, TMP_BUFLEN, stdin);
	val = strtol (str, NULL, 0);
	printf ("\n");
	switch (val) {
	case -1:
		printf ("Deleting one interpacket stuffing byte.\n");
		if (ioctl (fd, ASI_IOC_TXCHANGENEXTIP, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to delete interpacket stuffing");
		}
		break;
	case 0:
		break;
	case 1:
		printf ("Inserting one interpacket stuffing byte.\n");
		if (ioctl (fd, ASI_IOC_TXCHANGENEXTIP, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to insert interpacket stuffing");
		}
		break;
	default:
		printf ("Invalid change.\n");
		break;
	}
	return;
}

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/asi/asitx%i/%s";
	int opt, fd, choice = 0;
	unsigned int id, version, cap;
	struct util_info *info;
	struct stat buf;
	int num;
	unsigned long int count27, transport;
	const char *version_name;
	char name[TMP_BUFLEN], str[TMP_BUFLEN], *endptr;

	argv0 = argv[0];
	while ((opt = getopt (argc, argv, "hV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv0);
			printf ("Interactively issue DVB ASI "
				"transmitter ioctls to DEVICE_FILE.\n\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -V\toutput version information "
				"and exit\n\n");
			printf ("Report bugs to <support@linsys.ca>.\n");
			return 0;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2000-2010 "
				"Linear Systems Ltd.\n"
				"This is free software; "
				"see the source for copying conditions.  "
				"There is NO\n"
				"warranty; not even for MERCHANTABILITY "
				"or FITNESS FOR A PARTICULAR PURPOSE.\n");
			return 0;
		case '?':
			goto USAGE;
		}
	}

	/* Check the number of arguments */
	if ((argc - optind) < 1) {
		fprintf (stderr, "%s: missing argument\n", argv0);
		goto USAGE;
	} else if ((argc - optind) > 1) {
		fprintf (stderr, "%s: extra argument\n", argv0);
		goto USAGE;
	}

	/* Get the sysfs info */
	memset (&buf, 0, sizeof (buf));
	if (stat (argv[optind], &buf) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the file status");
		return -1;
	}
	if (!S_ISCHR (buf.st_mode)) {
		fprintf (stderr, "%s: not a character device\n", argv0);
		return -1;
	}
	if (buf.st_rdev & 0x0080) {
		fprintf (stderr, "%s: not a transmitter\n", argv0);
		return -1;
	}
	num = buf.st_rdev & 0x007f;
	snprintf (name, sizeof (name), fmt, num, "dev");
	memset (str, 0, sizeof (str));
	if (util_read (name, str, sizeof (str)) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device number");
		return -1;
	}
	if (strtoul (str, &endptr, 0) != (buf.st_rdev >> 8)) {
		fprintf (stderr, "%s: not an ASI device\n", argv0);
		return -1;
	}
	if (*endptr != ':') {
		fprintf (stderr, "%s: error reading %s\n", argv0, name);
		return -1;
	}

	/* Open the file */
	if ((fd = open (argv[optind], O_WRONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to open file for writing");
		return -1;
	}

	/* Get the transport type */
	snprintf (name, sizeof (name), fmt, num, "transport");
	if (util_strtoul (name, &transport) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transport type");
		close (fd);
		return -1;
	}

	/* Get the device ID */
	if (ioctl (fd, ASI_IOC_GETID, &id) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device ID");
		close (fd);
		return -1;
	}

	/* Get the firmware version */
	if (ioctl (fd, ASI_IOC_GETVERSION, &version) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the device firmware version");
		close (fd);
		return -1;
	}

	if (((info = getinfo (id)) == NULL) ||
		!(info->flags & UTIL_ASITX)) {
		fprintf (stderr, "%s: invalid device ID\n", argv0);
		close (fd);
		return -1;
	}
	switch (id) {
	case PCI_DEVICE_ID_LINSYS_DVBTX:
		if (version < 0x0400) {
			version_name = " (DVB Master I Send)";
		} else {
			version_name = " (DVB Master II Send)";
		}
		break;
	case PCI_DEVICE_ID_LINSYS_DVBFD:
		switch (version >> 12) {
		case 0:
			version_name = " (PID Filter)";
			break;
		case 1:
			version_name = " (PID Monitor)";
			break;
		default:
			version_name = " (Unknown)";
			break;
		}
		break;
	default:
		version_name = "";
		break;
	}

	/* Get the transmitter capabilities */
	if (ioctl (fd, ASI_IOC_TXGETCAP, &cap) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the transmitter capabilities");
		close (fd);
		return -1;
	}

	/* Get the 27 MHz counter status */
	if (cap & ASI_CAP_TX_BYTESOR27) {
		snprintf (name, sizeof (name), fmt, num, "count27");
		if (util_strtoul (name, &count27) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get "
				"the 27 MHz counter status");
			close (fd);
			return -1;
		}
	} else if (cap & ASI_CAP_TX_27COUNTER) {
		count27 = 1;
	} else {
		count27 = 0;
	}

	while (choice != 12) {
		printf ("\n\t%s, firmware version %u.%u (0x%04X)%s\n",
			info->name, version >> 8, version & 0x00ff,
			version, version_name);
		printf ("\ton %s\n\n", argv[optind]);
		printf ("\t 1. Show the transmitter capabilities\n");
		printf ("\t 2. Get the event flags\n");
		printf ("\t 3. Get the driver transmit buffer "
			"queue length\n");
		if (transport == ASI_CTL_TRANSPORT_DVB_ASI) {
			printf ("\t 4. Set the stuffing parameters\n");
		} else {
			printf ("\t 4.\n");
		}
		if (((cap & ASI_CAP_TX_BYTECOUNTER) &&
			!(cap & ASI_CAP_TX_BYTESOR27)) ||
			((cap & ASI_CAP_TX_BYTESOR27) && !count27)) {
			printf ("\t 5. Get the byte counter value\n");
		} else {
			printf ("\t 5.\n");
		}
		if (((cap & ASI_CAP_TX_27COUNTER) &&
			!(cap & ASI_CAP_TX_BYTESOR27)) ||
			((cap & ASI_CAP_TX_BYTESOR27) && count27)) {
			printf ("\t 6. Get the 27 MHz counter value\n");
		} else {
			printf ("\t 6.\n");
		}
		if (cap & ASI_CAP_TX_DATA) {
			printf ("\t 7. Get the transmit data status\n");
		} else {
			printf ("\t 7.\n");
		}
		if (cap & ASI_CAP_TX_PCRSTAMP) {
			printf ("\t 8. Set a PID to watch\n");
			printf ("\t 9. Get the last PCR departure timestamp\n");
		} else {
			printf ("\t 8.\n");
			printf ("\t 9.\n");
		}
		if (cap & ASI_CAP_TX_CHANGENEXTIP) {
			printf ("\t10. Make single-shot bitrate correction\n");
		} else {
			printf ("\t10.\n");
		}
		printf ("\t11. Fsync\n");
		printf ("\t12. Quit\n");
		printf ("\nEnter choice: ");
		fgets (str, TMP_BUFLEN, stdin);
		choice = strtol (str, NULL, 0);
		printf ("\n");
		switch (choice) {
		case 1:
			show_cap (cap);
			break;
		case 2:
			get_events (fd);
			break;
		case 3:
			get_buflevel (fd);
			break;
		case 4:
			if (transport == ASI_CTL_TRANSPORT_DVB_ASI) {
				set_stuffing (fd, cap);
			}
			break;
		case 5:
			if (((cap & ASI_CAP_TX_BYTECOUNTER) &&
				!(cap & ASI_CAP_TX_BYTESOR27)) ||
				((cap & ASI_CAP_TX_BYTESOR27) && !count27)) {
				get_bytecount (fd);
			}
			break;
		case 6:
			if (((cap & ASI_CAP_TX_27COUNTER) &&
				!(cap & ASI_CAP_TX_BYTESOR27)) ||
				((cap & ASI_CAP_TX_BYTESOR27) && count27)) {
				get_27count (fd);
			}
			break;
		case 7:
			if (cap & ASI_CAP_TX_DATA) {
				get_data (fd);
			}
			break;
		case 8:
			if (cap & ASI_CAP_TX_PCRSTAMP) {
				set_pid (fd);
			}
			break;
		case 9:
			if (cap & ASI_CAP_TX_PCRSTAMP) {
				get_pcrstamp (fd);
			}
			break;
		case 10:
			if (cap & ASI_CAP_TX_CHANGENEXTIP) {
				changenextip (fd);
			}
			break;
		case 11:
			printf ("Fsyncing.\n");
			if (fsync (fd) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to fsync");
			} else {
				printf ("Fsync successful.\n");
			}
			break;
		default:
			break;
		}
	}
	close (fd);
	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv0);
	return -1;
}

