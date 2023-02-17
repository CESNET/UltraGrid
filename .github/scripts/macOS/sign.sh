#!/bin/sh -eux
##
## Signs given application bundle
##
## Usage:
##  sign.sh <app_bundle_directory>
##
## Environment variables:
## - **apple_key_p12_b64**  - base64-encoded $KEY_FILE (using password $KEY_FILE_PASS)
## - **notarytool_credentials** - developer credentials to be used with notarytool (in format user:password:team_id)

APP=${1?Appname must be passed as a first argument}

if [ -z "$apple_key_p12_b64" ] || [ -z "$notarytool_credentials" ]; then
        echo "Could not find key to sign the application" 2>&1
        if [ "$GITHUB_REPOSITORY" = "CESNET/UltraGrid" ] && ! expr "$GITHUB_REF" : refs/pull >/dev/null; then
                exit 1
        else
                exit 0
        fi
fi

# Import keys
# Inspired by https://www.update.rocks/blog/osx-signing-with-travis/
KEY_CHAIN=build.keychain
KEY_CHAIN_PASS=build
KEY_FILE=/tmp/signing_key.p12
KEY_FILE_PASS=dummy
echo "$apple_key_p12_b64" | base64 -d > $KEY_FILE
security create-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN
security default-keychain -s $KEY_CHAIN
security unlock-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN
security import "$KEY_FILE" -A -P "$KEY_FILE_PASS"
security set-key-partition-list -S apple-tool:,apple: -s -k $KEY_CHAIN_PASS $KEY_CHAIN

# Sign the application
# Libs need to be signed explicitly for some reason
for f in $(find "$APP/Contents/libs" -type f) $APP; do
        codesign --force --deep -s CESNET --options runtime --entitlements data/entitlements.mac.plist -v "$f"
done
#codesign --force --deep -s CESNET --options runtime -v $APP/Contents/MacOS/uv-qt

# Zip and send for notarization
ZIP_FILE=uv-qt.zip
ditto -c -k --keepParent "$APP" $ZIP_FILE
set +x
DEVELOPER_USERNAME=$(echo "$notarytool_credentials" | cut -d: -f1)
DEVELOPER_PASSWORD=$(echo "$notarytool_credentials" | cut -d: -f2)
DEVELOPER_TEAMID=$(echo "$notarytool_credentials" | cut -d: -f3)
xcrun notarytool submit $ZIP_FILE --apple-id "$DEVELOPER_USERNAME" --team-id "$DEVELOPER_TEAMID" --password "$DEVELOPER_PASSWORD" --wait
set -x
# If everything is ok, staple the app
xcrun stapler staple "$APP"

