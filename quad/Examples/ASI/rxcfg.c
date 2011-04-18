/* rxcfg.c
 * 
 * Demonstrate DVB ASI receiver ioctls.
 *
 * Copyright (C) 2000-2008 Linear Systems Ltd. All rights reserved.
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
static const char progname[] = "rxcfg";

static void
show_cap (unsigned int cap)
{
	char str[TMP_BUFLEN] = "[ ] ";

	str[1] = (cap & ASI_CAP_RX_SYNC) ? 'X' : ' ';
	printf ("%sPacket synchronization\n", str);

	str[1] = (cap & ASI_CAP_RX_MAKE188) ? 'X' : ' ';
	printf ("%sConversion of 204-byte packets to 188-byte packets\n", str);

	str[1] = (cap & ASI_CAP_RX_BYTECOUNTER) ? 'X' : ' ';
	printf ("%sReceive byte counter\n", str);

	if (cap & ASI_CAP_RX_BYTESOR27) {
		printf ("[X] Byte counter is switchable to 27 MHz counter\n");
	} else {
		str[1] = (cap & ASI_CAP_RX_27COUNTER) ? 'X' : ' ';
		printf ("%s27 MHz counter\n", str);
	}

	str[1] = (cap & ASI_CAP_RX_INVSYNC) ? 'X' : ' ';
	printf ("%sSynchronization on inverted packet synchronization bytes\n",
		str);

	str[1] = (cap & ASI_CAP_RX_CD) ? 'X' : ' ';
	printf ("%sCarrier detect\n", str);

	str[1] = (cap & ASI_CAP_RX_DSYNC) ? 'X' : ' ';
	printf ("%sSynchronization after two packet synchronization bytes\n",
		str);

	str[1] = (cap & ASI_CAP_RX_DATA) ? 'X' : ' ';
	printf ("%sReceive data status\n", str);

	if (cap & ASI_CAP_RX_NULLPACKETS) {
		printf ("[X] PID filtering with null packet replacement\n");
	} else {
		str[1] = (cap & ASI_CAP_RX_PIDFILTER) ? 'X' : ' ';
		printf ("%sPID filtering\n", str);
	}

	if (cap & ASI_CAP_RX_4PIDCOUNTER) {
		printf ("[X] Four PID counters\n");
	} else {
		str[1] = (cap & ASI_CAP_RX_PIDCOUNTER) ? 'X' : ' ';
		printf ("%sPID counter\n", str);
	}

	str[1] = (cap & ASI_CAP_RX_FORCEDMA) ? 'X' : ' ';
	printf ("%sForced DMA transfer\n", str);

	str[1] = (cap & ASI_CAP_RX_TIMESTAMPS) ? 'X' : ' ';
	printf ("%sPacket timestamping\n", str);

	str[1] = (cap & ASI_CAP_RX_PTIMESTAMPS) ? 'X' : ' ';
	printf ("%sPrepended packet timestamps\n", str);

	str[1] = (cap & ASI_CAP_RX_REDUNDANT) ? 'X' : ' ';
	printf ("%sRedundant input\n", str);

	printf ("\nPress Enter to continue: ");
	fgets (str, TMP_BUFLEN, stdin);
	return;
}

static void
get_events (int fd)
{
	unsigned int val;
	char str[TMP_BUFLEN];

	printf ("Getting the receiver event flags.\n");
	if (ioctl (fd, ASI_IOC_RXGETEVENTS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver event flags");
	} else if (val) {
		if (val & ASI_EVENT_RX_BUFFER) {
			printf ("Driver receive buffer queue "
				"overrun detected.\n");
		}
		if (val & ASI_EVENT_RX_FIFO) {
			printf ("Onboard receive FIFO overrun detected.\n");
		}
		if (val & ASI_EVENT_RX_CARRIER) {
			printf ("Carrier status change detected.\n");
		}
		if (val & ASI_EVENT_RX_AOS) {
			printf ("Acquisition of packet "
				"synchronization detected.\n");
		}
		if (val & ASI_EVENT_RX_LOS) {
			printf ("Loss of packet "
				"synchronization detected.\n");
		}
		if (val & ASI_EVENT_RX_DATA) {
			printf ("Receive data change detected.\n");
		}
	} else {
		printf ("No receiver events detected.\n");
	}
	printf ("\nPress Enter to continue: ");
	fgets (str, TMP_BUFLEN, stdin);
	return;
}

static void
get_buflevel (int fd)
{
	unsigned int val;

	printf ("Getting the driver receive buffer queue length.\n");
	if (ioctl (fd, ASI_IOC_RXGETBUFLEVEL, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get "
			"the driver receive buffer queue length");
	} else {
		printf ("Driver receive buffer queue length = %u.\n", val);
	}
	return;
}

static void
get_status (int fd)
{
	int val;

	printf ("Getting the receiver status.\n");
	if (ioctl (fd, ASI_IOC_RXGETSTATUS, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver status");
	} else {
		printf ("Receiver is ");
		switch (val) {
		case 0:
			printf ("blocking data.\n");
			break;
		case 1:
			printf ("passing data in raw mode.\n");
			break;
		default:
			printf ("passing %i-byte packets.\n", val);
			break;
		}
	}
	return;
}

static void
get_bytecount (int fd)
{
	unsigned int val;

	printf ("Getting the byte counter value.\n");
	if (ioctl (fd, ASI_IOC_RXGETBYTECOUNT, &val) < 0) {
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
	if (ioctl (fd, ASI_IOC_RXGET27COUNT, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the 27 MHz counter value");
	} else {
		printf ("27 MHz count = %u.\n", val);
	}
	return;
}

static void
set_invsync (int fd, unsigned int cap)
{
	int val;
	char str[TMP_BUFLEN];

	printf ("\t 1. Synchronize on 47h\n");
	if (cap & ASI_CAP_RX_INVSYNC) {
		printf ("\t 2. Synchronize on 47h or B8h\n");
	}
	printf ("\nEnter choice: ");
	fgets (str, TMP_BUFLEN, stdin);
	printf ("\n");
	switch (strtol (str, NULL, 0)) {
	case 1:
		printf ("Setting the packet synchronization byte.\n");
		val = 0;
		if (ioctl (fd, ASI_IOC_RXSETINVSYNC, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to disable synchronization on "
				"inverted packet synchronization bytes");
		} else {
			printf ("Synchronizing on 47h.\n");
		}
		break;
	case 2:
		if (cap & ASI_CAP_RX_INVSYNC) {
			printf ("Setting the packet synchronization byte.\n");
			val = 1;
			if (ioctl (fd, ASI_IOC_RXSETINVSYNC, &val) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to enable synchronization on "
					"inverted packet synchronization "
					"bytes");
			} else {
				printf ("Synchronizing on 47h or B8h.\n");
			}
		}
		break;
	default:
		break;
	}
	return;
}

static void
get_carrier (int fd)
{
	int val;

	printf ("Getting the carrier status.\n");
	if (ioctl (fd, ASI_IOC_RXGETCARRIER, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the carrier status");
	} else if (val) {
		printf ("Carrier detected.\n");
	} else {
		printf ("No carrier.\n");
	}
	return;
}

static void
set_dsync (int fd, unsigned int cap)
{
	int val;
	char str[TMP_BUFLEN];

	printf ("\t 1. Disable double packet synchronization\n");
	if (cap & ASI_CAP_RX_DSYNC) {
		printf ("\t 2. Enable "
			"double packet synchronization\n");
	}
	printf ("\nEnter choice: ");
	fgets (str, TMP_BUFLEN, stdin);
	printf ("\n");
	switch (strtol (str, NULL, 0)) {
	case 1:
		printf ("Setting the double packet synchronization mode.\n");
		val = 0;
		if (ioctl (fd, ASI_IOC_RXSETDSYNC, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to disable "
				"double packet synchronization");
		} else {
			printf ("Synchronizing on one packet.\n");
		}
		break;
	case 2:
		if (cap & ASI_CAP_RX_DSYNC) {
			printf ("Setting the double packet "
				"synchronization mode.\n");
			val = 1;
			if (ioctl (fd, ASI_IOC_RXSETDSYNC, &val) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to enable "
					"double packet synchronization");
			} else {
				printf ("Synchronizing on two packets.\n");
			}
		}
		break;
	default:
		break;
	}
	return;
}

static void
get_data (int fd)
{
	unsigned int val;

	printf ("Getting the receive data status.\n");
	if (ioctl (fd, ASI_IOC_RXGETRXD, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receive data status");
	} else if (val) {
		printf ("Data is being received.\n");
	} else {
		printf ("No data.\n");
	}
	return;
}

static void
set_filter (int fd)
{
	unsigned int i, val, mask, pflut[256], pidcount = 0;
	char str[TMP_BUFLEN], *endptr;

	printf ("\t 1. Discard some PIDs and pass all others\n");
	printf ("\t 2. Pass some PIDs and discard all others\n");
	printf ("\nEnter choice: ");
	fgets (str, TMP_BUFLEN, stdin);
	printf ("\n");
	switch (strtol (str, NULL, 0)) {
	case 1:
		for (i = 0; i < 256; i++) {
			pflut[i] = 0xffffffff;
		}
		do {
			printf ("Enter the PID to discard "
				"or press Enter to continue: ");
			fgets (str, TMP_BUFLEN, stdin);
			val = strtoul (str, &endptr, 0);
			printf ("\n");
			if (!strcmp (str, "\n")) {
				break;
			}
			if (*endptr != '\n' || val > 0x1fff) {
				printf ("Invalid PID.\n");
				return;
			}
			mask = 0x00000001 << (val & 0x1f);
			pflut[val >> 5] = pflut[val >> 5] & ~mask;
		} while (1);
		printf ("Setting the PID filter.\n");
		if (ioctl (fd, ASI_IOC_RXSETPF, pflut) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to set the PID filter");
		} else {
			printf ("Discarding the following PIDs (");
			for (i = 0; i < 8192; i++) {
				if ((pflut[i >> 5] & (1 << (i & 0x1f))) == 0) {
					if (pidcount > 0) {
						printf (", ");
					}
					printf ("%i", i);
					pidcount++;
				}
			}
			printf (") and passing all others.\n");
		}
		break;
	case 2:
		for (i = 0; i < 256; i++) {
			pflut[i] = 0x00000000;
		}
		do {
			printf ("Enter the PID to pass "
				"or press Enter to continue: ");
			fgets (str, TMP_BUFLEN, stdin);
			val = strtoul (str, &endptr, 0);
			printf ("\n");
			if (!strcmp (str, "\n")) {
				break;
			}
			if (*endptr != '\n' || val > 0x1fff) {
				printf ("Invalid PID.\n");
				return;
			}
			mask = 0x00000001 << (val & 0x1f);
			pflut[val >> 5] = pflut[val >> 5] | mask;
		} while (1);
		printf ("Setting the PID filter.\n");
		if (ioctl (fd, ASI_IOC_RXSETPF, pflut) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to set the PID filter");
		} else {
			printf ("Passing the following PIDs (");
			for (i = 0; i < 8192; i++) {
				if ((pflut[i >> 5] & (1 << (i & 0x1f))) != 0) {
					if (pidcount > 0) {
						printf (", ");
					}
					printf ("%i", i);
					pidcount++;
				}
			}
			printf (") and discarding all others.\n");
		}
		break;
	default:
		break;
	}
	return;
}

static void
set_pid (int fd, unsigned int cap)
{
	int choice, val;
	unsigned int cmd;
	char str[TMP_BUFLEN];

	printf ("\t 1. Set PID counter 0\n");
	if (cap & ASI_CAP_RX_4PIDCOUNTER) {
		printf ("\t 2. Set PID counter 1\n");
		printf ("\t 3. Set PID counter 2\n");
		printf ("\t 4. Set PID counter 3\n");
	}
	printf ("\nEnter choice: ");
	fgets (str, TMP_BUFLEN, stdin);
	choice = strtol (str, NULL, 0);
	printf ("\n");
	switch (choice) {
	case 1:
		cmd = ASI_IOC_RXSETPID0;
		break;
	case 2:
		if (!(cap & ASI_CAP_RX_4PIDCOUNTER)) {
			return;
		}
		cmd = ASI_IOC_RXSETPID1;
		break;
	case 3:
		if (!(cap & ASI_CAP_RX_4PIDCOUNTER)) {
			return;
		}
		cmd = ASI_IOC_RXSETPID2;
		break;
	case 4:
		if (!(cap & ASI_CAP_RX_4PIDCOUNTER)) {
			return;
		}
		cmd = ASI_IOC_RXSETPID3;
		break;
	default:
		return;
	}
	printf ("Enter the PID to count: ");
	fgets (str, TMP_BUFLEN, stdin);
	val = strtol (str, NULL, 0);
	printf ("\n");
	if (val <= 0x1fff) {
		printf ("Setting PID counter %i.\n", choice - 1);
		if (ioctl (fd, cmd, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to set the PID counter");
		} else {
			printf ("Counting PID %4Xh.\n", val);
		}
	} else {
		printf ("Invalid PID.\n");
	}
	return;
}

static void
get_pidcount (int fd, unsigned int cap)
{
	int choice;
	unsigned int val, cmd;
	char str[TMP_BUFLEN];

	printf ("\t 1. Get PID count 0\n");
	if (cap & ASI_CAP_RX_4PIDCOUNTER) {
		printf ("\t 2. Get PID count 1\n");
		printf ("\t 3. Get PID count 2\n");
		printf ("\t 4. Get PID count 3\n");
	}
	printf ("\nEnter choice: ");
	fgets (str, TMP_BUFLEN, stdin);
	choice = strtol (str, NULL, 0);
	printf ("\n");
	switch (choice) {
	case 1:
		cmd = ASI_IOC_RXGETPID0COUNT;
		break;
	case 2:
		if (!(cap & ASI_CAP_RX_4PIDCOUNTER)) {
			return;
		}
		cmd = ASI_IOC_RXGETPID1COUNT;
		break;
	case 3:
		if (!(cap & ASI_CAP_RX_4PIDCOUNTER)) {
			return;
		}
		cmd = ASI_IOC_RXGETPID2COUNT;
		break;
	case 4:
		if (!(cap & ASI_CAP_RX_4PIDCOUNTER)) {
			return;
		}
		cmd = ASI_IOC_RXGETPID3COUNT;
		break;
	default:
		return;
	}
	printf ("Getting PID count %i.\n", choice - 1);
	if (ioctl (fd, cmd, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the PID count");
	} else {
		printf ("PID count = %u.\n", val);
	}
	return;
}

static void
get_status2 (int fd)
{
	int val;

	printf ("Getting the redundant input status.\n");
	if (ioctl (fd, ASI_IOC_RXGETSTATUS2, &val) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the redundant input status");
	} else {
		printf ("Redundant input is ");
		if (val) {
			printf ("passing data.\n");
		} else {
			printf ("blocking data.\n");
		}
	}
	return;
}

static void
set_input (int fd, unsigned int cap)
{
	int val;
	char str[TMP_BUFLEN];

	printf ("\t 1. Receive on Port A\n");
	if (cap & ASI_CAP_RX_REDUNDANT) {
		printf ("\t 2. Receive on Port B\n");
	}
	printf ("\nEnter choice: ");
	fgets (str, TMP_BUFLEN, stdin);
	printf ("\n");
	switch (strtol (str, NULL, 0)) {
	case 1:
		printf ("Setting the input port.\n");
		val = 0;
		if (ioctl (fd, ASI_IOC_RXSETINPUT, &val) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to select Port A");
		} else {
			printf ("Receiving on Port A.\n");
		}
		break;
	case 2:
		if (cap & ASI_CAP_RX_REDUNDANT) {
			printf ("Setting the input port.\n");
			val = 1;
			if (ioctl (fd, ASI_IOC_RXSETINPUT, &val) < 0) {
				fprintf (stderr, "%s: ", argv0);
				perror ("unable to select Port B");
			} else {
				printf ("Receiving on Port B.\n");
			}
		}
		break;
	default:
		break;
	}
	return;
}

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/asi/asirx%i/%s";
	int opt, fd, choice = 0;
	unsigned int id, version, cap;
	struct util_info *info;
	struct stat buf;
	int num;
	unsigned long int mode, count27;
	const char *version_name;
	char name[TMP_BUFLEN], str[TMP_BUFLEN], *endptr;

	argv0 = argv[0];
	while ((opt = getopt (argc, argv, "hV")) != -1) {
		switch (opt) {
		case 'h':
			printf ("Usage: %s [OPTION]... DEVICE_FILE\n",
				argv0);
			printf ("Interactively issue DVB ASI "
				"receiver ioctls to DEVICE_FILE.\n\n");
			printf ("  -h\tdisplay this help and exit\n");
			printf ("  -V\toutput version information "
				"and exit\n\n");
			printf ("Report bugs to <support@linsys.ca>.\n");
			return 0;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2000-2006 "
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
	if (!(buf.st_rdev & 0x0080)) {
		fprintf (stderr, "%s: not a receiver\n", argv0);
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
	if ((fd = open (argv[optind], O_RDONLY, 0)) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to open file for reading");
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
		!(info->flags & UTIL_ASIRX)) {
		fprintf (stderr, "%s: invalid device ID\n", argv0);
		close (fd);
		return -1;
	}
	switch (id) {
	case PCI_DEVICE_ID_LINSYS_DVBRX:
		if (version < 0x0400) {
			version_name = " (DVB Master I Receive)";
		} else {
			version_name = " (DVB Master II Receive)";
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

	/* Get the receiver capabilities */
	if (ioctl (fd, ASI_IOC_RXGETCAP, &cap) < 0) {
		fprintf (stderr, "%s: ", argv0);
		perror ("unable to get the receiver capabilities");
		close (fd);
		return -1;
	}

	/* Get the operating mode */
	if (cap & ASI_CAP_RX_SYNC) {
		snprintf (name, sizeof (name), fmt, num, "mode");
		if (util_strtoul (name, &mode) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get "
				"the receiver operating mode");
			close (fd);
			return -1;
		}
	} else {
		mode = ASI_CTL_RX_MODE_RAW;
	}

	/* Get the 27 MHz counter status */
	count27 = 0;
	if (cap & ASI_CAP_RX_BYTESOR27) {
		snprintf (name, sizeof (name), fmt, num, "count27");
		if (util_strtoul (name, &count27) < 0) {
			fprintf (stderr, "%s: ", argv0);
			perror ("unable to get "
				"the 27 MHz counter status");
			close (fd);
			return -1;
		}
	}

	while (choice != 17) {
		printf ("\n\t%s, firmware version %u.%u (0x%04X)%s\n",
			info->name, version >> 8, version & 0x00ff,
			version, version_name);
		printf ("\ton %s\n\n", argv[optind]);
		printf ("\t 1. Show the receiver capabilities\n");
		printf ("\t 2. Get the receiver event flags\n");
		printf ("\t 3. Get the driver receive buffer "
			"queue length\n");
		printf ("\t 4. Get the receiver status\n");
		if (((cap & ASI_CAP_RX_BYTECOUNTER) &&
			!(cap & ASI_CAP_RX_BYTESOR27)) ||
			((cap & ASI_CAP_RX_BYTESOR27) && !count27)) {
			printf ("\t 5. Get the byte counter value\n");
		} else {
			printf ("\t 5.\n");
		}
		if (((cap & ASI_CAP_RX_27COUNTER) &&
			!(cap & ASI_CAP_RX_BYTESOR27)) ||
			((cap & ASI_CAP_RX_BYTESOR27) && count27)) {
			printf ("\t 6. Get the 27 MHz counter value\n");
		} else {
			printf ("\t 6.\n");
		}
		if (cap & ASI_CAP_RX_SYNC) {
			printf ("\t 7. Set the "
				"packet synchronization byte\n");
		} else {
			printf ("\t 7.\n");
		}
		if (cap & ASI_CAP_RX_CD) {
			printf ("\t 8. Get the carrier status\n");
		} else {
			printf ("\t 8.\n");
		}
		if (cap & ASI_CAP_RX_SYNC) {
			printf ("\t 9. Set the double packet "
				"synchronization mode\n");
		} else {
			printf ("\t 9.\n");
		}
		if (cap & ASI_CAP_RX_DATA) {
			printf ("\t10. Get the receive data status\n");
		} else {
			printf ("\t10.\n");
		}
		if (cap & ASI_CAP_RX_PIDFILTER) {
			printf ("\t11. Set PID filter\n");
		} else {
			printf ("\t11.\n");
		}
		if (cap & ASI_CAP_RX_PIDCOUNTER) {
			printf ("\t12. Set a PID to count\n");
			printf ("\t13. Get a PID count\n");
		} else {
			printf ("\t12.\n");
			printf ("\t13.\n");
		}
		if (cap & ASI_CAP_RX_REDUNDANT) {
			printf ("\t14. Get redundant input status\n");
		} else {
			printf ("\t14.\n");
		}
		if (cap & ASI_CAP_RX_REDUNDANT) {
			printf ("\t15. Set redundant input port\n");
		} else {
			printf ("\t15.\n");
		}
		printf ("\t16. Fsync\n");
		printf ("\t17. Quit\n");
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
			get_status (fd);
			break;
		case 5:
			if (((cap & ASI_CAP_RX_BYTECOUNTER) &&
				!(cap & ASI_CAP_RX_BYTESOR27)) ||
				((cap & ASI_CAP_RX_BYTESOR27) && !count27)) {
				get_bytecount (fd);
			}
			break;
		case 6:
			if (((cap & ASI_CAP_RX_27COUNTER) &&
				!(cap & ASI_CAP_RX_BYTESOR27)) ||
				((cap & ASI_CAP_RX_BYTESOR27) && count27)) {
				get_27count (fd);
			}
			break;
		case 7:
			if (cap & ASI_CAP_RX_SYNC) {
				set_invsync (fd, cap);
			}
			break;
		case 8:
			if (cap & ASI_CAP_RX_CD) {
				get_carrier (fd);
			}
			break;
		case 9:
			if (cap & ASI_CAP_RX_SYNC) {
				set_dsync (fd, cap);
			}
			break;
		case 10:
			if (cap & ASI_CAP_RX_DATA) {
				get_data (fd);
			}
			break;
		case 11:
			if (cap & ASI_CAP_RX_PIDFILTER) {
				set_filter (fd);
			}
			break;
		case 12:
			if (cap & ASI_CAP_RX_PIDCOUNTER) {
				set_pid (fd, cap);
			}
			break;
		case 13:
			if (cap & ASI_CAP_RX_PIDCOUNTER) {
				get_pidcount (fd, cap);
			}
			break;
		case 14:
			if (cap & ASI_CAP_RX_REDUNDANT) {
				get_status2 (fd);
			}
			break;
		case 15:
			set_input (fd, cap);
			break;
		case 16:
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

