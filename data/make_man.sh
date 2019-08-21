#!/bin/sh -exu

# Requires asciidoc (a2x) and working UltraGrid (uv) in PATH
# TODO: add sections EXAMPLES, DESCRIPTION etc.

ASCIIDOC=uv.1.txt

cat <<EOF > $ASCIIDOC
= UV(1) =
:doctype: manpage

== NAME ==
uv - UltraGrid command line interface

== SYNOPSIS ==
*uv* ['OPTIONS'] 'ADDRESS'

== Options ==
EOF

uv --fullhelp | sed '0,/Options:/d' >> $ASCIIDOC

cat <<EOF >> $ASCIIDOC
== REPORTING BUGS ==
Please report bugs to 'ultragrid-dev@cesnet.cz'.

== RESOURCES ==
GitHub: <https://github.com/CESNET/UltraGrid>

Wiki (on-line documentation): <https://github.com/CESNET/UltraGrid/wiki>

Main web site: <http://www.ultragrid.cz>
EOF

a2x -f manpage -v $ASCIIDOC -D .

rm $ASCIIDOC

