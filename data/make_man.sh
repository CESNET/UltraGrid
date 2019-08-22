#!/bin/sh -exu

# Requires asciidoc (a2x) and working UltraGrid (uv, hd-rum-transcode) in PATH
# TODO: add sections EXAMPLES, DESCRIPTION etc.
# use '-d' parameter to keep source AsciiDoc files (for debugging)

bugreport_and_resources() {
	cat <<-'EOF'
	== REPORTING BUGS ==
	Report bugs to *ultragrid-dev@cesnet.cz* or use project *GitHub* to describe issues.

	== RESOURCES ==
	* GitHub: *<https://github.com/CESNET/UltraGrid>*
	* Wiki (on-line documentation): *<https://github.com/CESNET/UltraGrid/wiki>*
	* Main web site: *<http://www.ultragrid.cz>*
	EOF
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

	cat <<-'EOF' >> $ASCIIDOC
	== REPORTING BUGS ==
	Report bugs to *ultragrid-dev@cesnet.cz* or use project *GitHub* to describe issues.

	== RESOURCES ==
	* GitHub: *<https://github.com/CESNET/UltraGrid>*
	* Wiki (on-line documentation): *<https://github.com/CESNET/UltraGrid/wiki>*
	* Main web site: *<http://www.ultragrid.cz>*
	EOF

	a2x -f manpage -v $ASCIIDOC -D .

	test -n "$KEEP_FILES" || rm $ASCIIDOC
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

	=== Reflector running on local machine ===
	*uv* -t testcard -s testcard '-P 6004:5004:6006:5006' localhost
	*hd-rum-transcode* 8M 5004 receiver1 receiver2 -c JPEG receiver3 # video
	*hd-rum-transcode* 8M 5006 receiver1 receiver2 receiver3         # audio

	EOF

	bugreport_and_resources >> $ASCIIDOC

	a2x -f manpage -v $ASCIIDOC -D .

	test -n "$KEEP_FILES" || rm $ASCIIDOC
}

if [ "${1-}" = "-d" -o "${1-}" = "--debug" ]; then
	KEEP_FILES=1
else
	KEEP_FILES=
fi

uv_man
hd_rum_transcode_man

# vim: set noexpandtab tw=0:
