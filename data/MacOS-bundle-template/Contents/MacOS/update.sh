#!/bin/sh -eu

readonly appname=uv-qt.app
dir=$(dirname "$0")
bundle_path=$(cd "$dir"/../..; pwd)
readonly bundle_path dir

if ! expr "x$0" : x/ >/dev/null; then
        echo "Cannot update when not running with an absolute path!" 1>&2
        exit 1
fi

if expr "x$0" : x/Volumes > /dev/null; then
        echo "Refusing update of mounted image!" 1>&2
        exit 1
fi

version=$("$dir"/uv -v | sed 's/.*(\([^ ]*\).*/\1/;q')

get_release_json_val() {
        osascript -l "JavaScript" <<EOF
                var app = Application.currentApplication();
                app.includeStandardAdditions = true;

                function parseJSON(path) {
                    const data = app.read(path);
                    return JSON.parse(data);
                }
                function run() {
                    const releases = parseJSON("$1");
                    return releases.filter(item => item.prerelease == false)[0].$2;
                }
EOF
}

if [ "$version" = master ]; then
        url="https://github.com/CESNET/UltraGrid/releases/download/continuous/UltraGrid-continuous.dmg"
else
        json=$(mktemp)
        curl https://api.github.com/repos/CESNET/UltraGrid/releases -o "$json"
        current_ver=$(get_release_json_val "$json" tag_name)
        if [ "tags/$current_ver" = "$version" ]; then
                echo "Version $current_ver is current."
                exit 0
        fi
        url=$(get_release_json_val "$json" 'assets.filter(item => item.name.match(".*dmg"))[0].browser_download_url')
        rm "$json"
fi

bundle_name=$(basename "$bundle_path")
if [ "$bundle_name" != $appname ]; then
        echo "Application name is not ending with $appname" 1>&2
        exit 1
fi

updater_dir=$(mktemp -d /tmp/ug-updater.XXXXXXXX)
cd "$updater_dir"

echo "Downloading current version"
readonly dmg=UltraGrid-updated.dmg
curl -L "$url" -o $dmg

mkdir mnt
hdiutil mount -mountpoint mnt $dmg

rm -rf "$bundle_path"
echo "Removed old version, copying new files to $bundle_path..."
cp -a "mnt/$appname" "$bundle_path"

umount mnt

printf "Returning to "
cd -
echo "Removing temporary data"
rm -rf "$updater_dir"

