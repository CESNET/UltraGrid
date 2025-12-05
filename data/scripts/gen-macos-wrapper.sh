#!/bin/sh -eu

cat << 'EOF'
#!/bin/sh -eu

EOF

# req_macos is evaluated during the build (EOF not in '')
cat << EOF
req_macos=$(sw_vers -productVersion | cut -d. -f1)

EOF

# following macos var will be evaluated on run-time
cat << 'EOF'
macos=$(sw_vers -productVersion | cut -d. -f1)

MSG="Please use an alternative build for macOS older than ${req_macos:?}, available at:
https://github.com/CESNET/UltraGrid/releases/download/continuous/UltraGrid-nightly-alt.dmg"

if [ "${macos:?}" -lt "${req_macos:?}" ]; then
        BASENAME=$(basename "$0")
        if [ "$BASENAME" = uv-qt ]; then
                osascript -e "tell application \"SystemUIServer\"
display dialog \"$MSG\"
end"
        else
                echo "$MSG" >&2
        fi
        exit 1
fi

exec "$0-real" "$@"
EOF
