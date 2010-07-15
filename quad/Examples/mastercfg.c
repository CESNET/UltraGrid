/* mastercfg.c
 *
 * Master configuration program.
 *
 * Copyright (C) 2004-2010 Linear Systems Ltd. All rights reserved.
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

#include "master.h"
#include "util.h"

#define MAXLEN 256
#define BYPASS_FLAG	0x00000001
#define GPO_FLAG	0x00000002
#define WATCHDOG_FLAG	0x00000004
#define BLACKBURST_FLAG	0x00000008

static const char progname[] = "mastercfg";

int
main (int argc, char **argv)
{
	const char fmt[] = "/sys/class/%s/%s/%s";
	int opt;
	unsigned int write_flags;
	unsigned long int id, fw_version;
	unsigned long int bypass_mode, bypass_status, blackburst, gpi, gpo, watchdog;
	unsigned long long int uid;
	char *endptr;
	char name[MAXLEN], data[MAXLEN];

	/* Parse the command line */
	write_flags = 0;
	bypass_mode = 0;
	blackburst = 0;
	gpo = 0;
	watchdog = 0;
	while ((opt = getopt (argc, argv, "b:g:ho:t:V")) != -1) {
		switch (opt) {
		case 'b':
			write_flags |= BYPASS_FLAG;
			bypass_mode = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid bypass mode: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'g':
			write_flags |= BLACKBURST_FLAG;
			blackburst = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid black burst type: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'h':
			printf ("Usage: %s [OPTION]... DRIVER BOARD_ID\n",
				argv[0]);
			printf ("Configure a Master board.\n\n");
			printf ("  -b BYPASS\tset the bypass mode\n");
			printf ("  -g BLACKBURST\tset the black burst type\n");
			printf ("  -h\t\tdisplay this help and exit\n");
			printf ("  -o OUTPUT\tset the general purpose output "
				"value to OUTPUT\n");
			printf ("  -t TIMEOUT\tset the watchdog timeout "
				"to TIMEOUT milliseconds\n");
			printf ("  -V\t\toutput version information "
				"and exit\n");
			printf ("\nIf no options are specified, "
				"the current configuration is displayed.\n");
			printf ("\nBYPASS may be:\n");
			printf ("\t0 (bypass)\n");
			printf ("\t1 (normal)\n");
			printf ("\t2 (enable and reset the watchdog timer)\n");
			printf ("\nBLACKBURST may be:\n");
			printf ("\t0 (NTSC)\n");
			printf ("\t1 (PAL)\n");
			printf ("\nReport bugs to <support@linsys.ca>.\n");
			return 0;
		case 'o':
			write_flags |= GPO_FLAG;
			gpo = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid general purpose "
					"output value: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 't':
			write_flags |= WATCHDOG_FLAG;
			watchdog = strtoul (optarg, &endptr, 0);
			if (*endptr != '\0') {
				fprintf (stderr,
					"%s: invalid watchdog timeout: %s\n",
					argv[0], optarg);
				return -1;
			}
			break;
		case 'V':
			printf ("%s from master-%s (%s)\n", progname,
				MASTER_DRIVER_VERSION,
				MASTER_DRIVER_DATE);
			printf ("\nCopyright (C) 2004-2010 "
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
	if ((argc - optind) < 2) {
		fprintf (stderr, "%s: missing arguments\n", argv[0]);
		goto USAGE;
	} else if ((argc - optind) > 2) {
		fprintf (stderr, "%s: extra operand\n", argv[0]);
		goto USAGE;
	}

	printf (fmt, argv[optind], argv[optind + 1], ":\n");
	if (write_flags) {
		if (write_flags & BYPASS_FLAG) {
			snprintf (name, sizeof (name),
				fmt, argv[optind], argv[optind + 1],
				"bypass_mode");
			snprintf (data, sizeof (data), "%lu\n", bypass_mode);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the bypass mode");
				return -1;
			}
			switch (bypass_mode) {
			case MASTER_CTL_BYPASS_ENABLE:
				printf ("\tEnabling bypass.\n");
				break;
			case MASTER_CTL_BYPASS_DISABLE:
				printf ("\tDisabling bypass.\n");
				break;
			case MASTER_CTL_BYPASS_WATCHDOG:
				printf ("\tGiving bypass control to "
					"the watchdog timer.\n");
				break;
			default:
				printf ("\tSet bypass mode = %lu.\n",
					bypass_mode);
				break;
			}
		}
		if (write_flags & BLACKBURST_FLAG) {
			snprintf (name, sizeof (name),
				fmt, argv[optind], argv[optind + 1],
				"blackburst_type");
			snprintf (data, sizeof (data), "%lu\n", blackburst);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the black burst type");
				return -1;
			}
			switch (blackburst) {
			case MASTER_CTL_BLACKBURST_NTSC:
				printf ("\tBlack burst type set to NTSC.\n");
				break;
			case MASTER_CTL_BLACKBURST_PAL:
				printf ("\tBlack burst type set to PAL.\n");
				break;
			default:
				printf ("\tSet black burst type = %lu.\n",
					blackburst);
				break;
			}
		}
		if (write_flags & GPO_FLAG) {
			snprintf (name, sizeof (name),
				fmt, argv[optind], argv[optind + 1],
				"gpo");
			snprintf (data, sizeof (data), "%lu\n", gpo);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the general purpose output value");
				return -1;
			}
			printf ("\tSet general purpose output = %lu.\n",
				gpo);
		}
		if (write_flags & WATCHDOG_FLAG) {
			snprintf (name, sizeof (name),
				fmt, argv[optind], argv[optind + 1],
				"watchdog");
			snprintf (data, sizeof (data), "%lu\n", watchdog);
			if (util_write (name, data, sizeof (data)) < 0) {
				fprintf (stderr, "%s: ", argv[0]);
				perror ("unable to set "
					"the watchdog timeout");
				return -1;
			}
			printf ("\tSet watchdog timeout = %lu ms.\n", watchdog);
		}
	} else {
		struct util_info *info;

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"device/device");
		if (util_strtoul (name, &id) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the firmware version");
			return -1;
		}
		if ((info = getinfo (id)) == NULL) {
			printf ("\tUnknown device\n");
		} else {
			printf ("\t%s\n",
				info->name);
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"fw_version");
		if (util_strtoul (name, &fw_version) < 0) {
			fprintf (stderr, "%s: ", argv[0]);
			perror ("unable to get the firmware version");
			return -1;
		}
		printf ("\tFirmware version: %lu.%lu (0x%04lX)\n",
			(fw_version >> 8), (fw_version & 0x00ff), fw_version);

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"bypass_mode");
		if (util_strtoul (name, &bypass_mode) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tBypass mode: %lu ", bypass_mode);
			switch (bypass_mode) {
			case MASTER_CTL_BYPASS_ENABLE:
				printf ("(bypass enabled)\n");
				break;
			case MASTER_CTL_BYPASS_DISABLE:
				printf ("(bypass disabled)\n");
				break;
			case MASTER_CTL_BYPASS_WATCHDOG:
				printf ("(bypass controlled by "
					"watchdog timer)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"bypass_status");
		if (util_strtoul (name, &bypass_status) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tBypass %sabled\n",
				bypass_status ? "dis" : "en");
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"blackburst_type");
		if (util_strtoul (name, &blackburst) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tBlack burst type: %lu ", blackburst);
			switch (blackburst) {
			case MASTER_CTL_BLACKBURST_NTSC:
				printf ("(NTSC)\n");
				break;
			case MASTER_CTL_BLACKBURST_PAL:
				printf ("(PAL)\n");
				break;
			default:
				printf ("(unknown)\n");
				break;
			}
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"gpi");
		if (util_strtoul (name, &gpi) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tGeneral purpose inputs: %lu (0x%02lX)\n",
				gpi, gpi);
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"gpo");
		if (util_strtoul (name, &gpo) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tGeneral purpose output: %lu (0x%02lX)\n",
				gpo, gpo);
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"uid");
		if (util_strtoull (name, &uid) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tUnique ID: 0x%016llX\n", uid);
		}

		snprintf (name, sizeof (name),
			fmt, argv[optind], argv[optind + 1],
			"watchdog");
		if (util_strtoul (name, &watchdog) > 0) {
			/* Don't complain on an error,
			 * since this parameter may not exist. */
			printf ("\tWatchdog timeout: %lu ms\n", watchdog);
		}
	}

	return 0;

USAGE:
	fprintf (stderr, "Try '%s -h' for more information.\n", argv[0]);
	return -1;
}

