#!/bin/sh -eux
## Usage:
##  sign.sh <app_bundle_directory>
##
## Environment variables:
## - **apple_key_p12_b64**  - base64-encoded $KEY_FILE (using password $KEY_FILE_PASS)
## - **altool_credentials** - developer credentials to be used with altool (in format user:password)

APP=${1?Appname must be passed as a first argument}

if [ -z "$apple_key_p12_b64" -o -z "$altool_credentials" ]; then
        echo "Could not find key to sign the application" 2>&1
        if [ "$GITHUB_WORKFLOW" = nightly ]; then
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
for f in `find $APP/Contents/libs -type f` $APP; do
        codesign --force --deep -s CESNET --options runtime --entitlements data/entitlements.mac.plist -v $f
done
#codesign --force --deep -s CESNET --options runtime -v $APP/Contents/MacOS/uv-qt

# Zip and send for notarization
ZIP_FILE=uv-qt.zip
UPLOAD_INFO_PLIST=/tmp/uplinfo.plist
REQUEST_INFO_PLIST=/tmp/reqinfo.plist
ditto -c -k --keepParent $APP $ZIP_FILE
DEVELOPER_USERNAME=$(echo "$altool_credentials" | cut -d: -f1)
DEVELOPER_PASSWORD=$(echo "$altool_credentials" | cut -d: -f2)
xcrun altool --notarize-app --primary-bundle-id cz.cesnet.ultragrid.uv-qt --username $DEVELOPER_USERNAME --password "$DEVELOPER_PASSWORD" --file $ZIP_FILE --output-format xml | tee $UPLOAD_INFO_PLIST

# Wait for notarization status
# Waiting inspired by https://nativeconnect.app/blog/mac-app-notarization-from-the-command-line/
SLEPT=0
TIMEOUT=7200
while true; do
        /usr/bin/xcrun altool --notarization-info `/usr/libexec/PlistBuddy -c "Print :notarization-upload:RequestUUID" $UPLOAD_INFO_PLIST` -u $DEVELOPER_USERNAME -p "$DEVELOPER_PASSWORD" --output-format xml | tee $REQUEST_INFO_PLIST
        STATUS=`/usr/libexec/PlistBuddy -c "Print :notarization-info:Status" $REQUEST_INFO_PLIST`
        if [ "$STATUS" != "in progress" -o $SLEPT -ge $TIMEOUT ]; then
                break
        fi
        sleep 60
        SLEPT=$(($SLEPT + 60))
done
if [ $STATUS != success ]; then
        UUID=`/usr/libexec/PlistBuddy -c "Print :notarization-info:RequestUUID" $REQUEST_INFO_PLIST`
        xcrun altool --notarization-info $UUID -u $DEVELOPER_USERNAME -p "$DEVELOPER_PASSWORD"
        echo "Could not notarize" 2>&1
        exit 1
fi

# If everything is ok, staple the app
xcrun stapler staple $APP

