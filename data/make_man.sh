#!/bin/sh -eu

# Requires asciidoc (a2x) and working UltraGrid (uv, hd-rum-transcode) in PATH
# TODO: add sections EXAMPLES, DESCRIPTION etc.
# use '-d' parameter to keep source AsciiDoc files (for debugging)

bugreport_and_resources() {
	cat <<-'EOF'
	== ENVIRONMENT VARIABLES ==
	*`ULTRAGRID_VERBOSE`*::
	  If is `ULTRAGRID_VERBOSE` environment variable is set, default UltraGrid
	  log level is `verbose`. Command-line option always overrides this value.

	== REPORTING BUGS ==
	Report bugs to *ultragrid-dev@cesnet.cz* or use project *GitHub* to describe issues.

	== RESOURCES ==
	* GitHub: *<https://github.com/CESNET/UltraGrid>*
	* Wiki (on-line documentation): *<https://github.com/CESNET/UltraGrid/wiki>*
	* Main web site: *<http://www.ultragrid.cz>*

	EOF
}

print_gen() {
	test -n "$DRY_RUN" || echo Generated $(echo $1 | sed 's/\.txt$//')
}

uv_man() {
	local ASCIIDOC=uv.1.txt

	cat <<-'EOF' > $ASCIIDOC
	= UV(1) =
	:doctype: manpage

	== NAME ==
	uv - UltraGrid command line interface

	== SYNOPSIS ==
	*uv* ['OPTIONS'] 'ADDRESS'

	== OPTIONS ==
	EOF

	uv --fullhelp | sed '0,/Options:/d' >> $ASCIIDOC

	bugreport_and_resources >> $ASCIIDOC
	cat <<-'EOF' >> $ASCIIDOC
	== SEE ALSO ==
	hd-rum-transcode(1)

	EOF

	test -n "$DRY_RUN" || a2x -f manpage $VERBOSE $ASCIIDOC -D .

	test -n "$KEEP_FILES" || rm $ASCIIDOC
	print_gen $ASCIIDOC
}

outdent_output() {
	sed 's/^\t//'
}

escape_apostrophe() {
	# a text "blabla 'ddd' bleble" should be replaced by
	# "blabla \'ddd' bleble" (second apostrophe not escaped,
	# otherwise AsciiDoc prints the backslash)
	sed 's/[^[:alpha:]]\(['\'']\)/\\\1/g'
}

hd_rum_transcode_man() {
	local ASCIIDOC=hd-rum-transcode.1.txt

	cat <<-'EOF' > $ASCIIDOC
	= HD-RUM-TRANSCODE(1) =
	:doctype: manpage

	== NAME ==
	hd-rum-transcode - transcoding reflector for UltraGrid

	== SYNOPSIS ==
	*hd-rum-transcode* '[global_opts] buffer_size port [host1_options] host1 [[host2_options] host2] ...'

	== OPTIONS ==
	EOF

	hd-rum-transcode -h | sed '0,/where/{/where/!d};/Please/,$d' >> $ASCIIDOC

	printf '\n== NOTES ==\n' >> $ASCIIDOC
	hd-rum-transcode -h | outdent_output | escape_apostrophe | sed '0,/Please/{/Please/!d}' >> $ASCIIDOC

	# below heredoc contains contiunation of NOTES section
	cat <<-'EOF' >> $ASCIIDOC

	Transcoder can be used as a reflector also for audio (and must be used
	if audio is deployed). It doesn't have, however, any transcoding facilities
	for audio. Only port remapping can be used for audio.

	== EXAMPLES ==
	=== Basic example ===
	*hd-rum-transcode* 8M 5004 receiver1 receiver2 -c JPEG receiver3 # video
	*hd-rum-transcode* 8M 5006 receiver1 receiver2 receiver3         # audio - no transcoding

	=== Reflector running on UltraGrid sender machine ===
	*uv* -t testcard -s testcard '-P 6004:5004:6006:5006' localhost
	*hd-rum-transcode* 8M 5004 receiver1 receiver2 -c JPEG receiver3 # video
	*hd-rum-transcode* 8M 5006 receiver1 receiver2 receiver3         # audio

	EOF

	bugreport_and_resources >> $ASCIIDOC

	cat <<-'EOF' >> $ASCIIDOC
	== SEE ALSO ==
	hd-rum(1) uv(1)

	EOF

	test -n "$DRY_RUN" || a2x -f manpage $VERBOSE $ASCIIDOC -D .
	test -n "$KEEP_FILES" || rm $ASCIIDOC
	print_gen $ASCIIDOC
}

usage() {
	printf "Usage:\n\t$0 [-d|--debug|-h|--help|-k|--keep|-n] [uv|hd-rum-trancode|all]\nwhere\n"
	printf "\t-d|--debug - print debug info\n"
	printf "\t-k|--keep - keep generated AsciiDoc sources\n"
	printf "\t-n - do not generate manpages (can be useful with -k)\n"
	printf "\n"
	printf "\tname of man page - specify 'all' generate all manual pages\n"
}

DRY_RUN=
KEEP_FILES=
VERBOSE=

TEMP=$(getopt -o 'dhkn' --long 'debug,help,keep' -n "$0" -- "$@")

if [ $? -ne 0 ]; then
        echo 'Terminating...' >&2
        exit 1
fi

# Note the quotes around "$TEMP": they are essential!
eval set -- "$TEMP"
unset TEMP

while true; do
	case "$1" in
		'-d'|'--debug')
			VERBOSE="-v"
			set -x
			;;
		'-h'|'--help')
			usage
			exit 0
			;;
		'-k'|'--keep')
			KEEP_FILES=1
			;;
		'-n')
			DRY_RUN=1
			;;
		'--')
			shift
			break
			;;
		*)
			echo 'Internal error!' >&2
			exit 1
			;;
	esac
	shift
done

if [ $# -eq 0 ]; then
	usage
	exit 1
fi

while [ $# -gt 0 ]; do
	case "$1" in
		all)
			set -- uv hd-rum-transcode
			continue
			;;
		uv)
			uv_man
			;;
		hd-rum-transcode)
			hd_rum_transcode_man
			;;
	esac
	shift
done

if [ -t 1 ]; then
	printf '\e[1mNote: \e[mDo not forget to check the generated manpages!\n'
fi

# vim: set noexpandtab tw=0:
