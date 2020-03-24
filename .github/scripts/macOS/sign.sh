#!/bin/sh -eux
## Usage:
##  sign.sh <app_bundle_directory>
##
## Environment variables:
## - **apple_key_p12_b64** - tar.bz2 with $KEY_FILE (with empty password) and $CERT_FILE
## - **$altool_pass**      - developer (see $DEVELOPER_USERNAME) app password

APP=${1?Appname must be passed as a first argument}
DEVELOPER_USERNAME=martin.pulec@cesnet.cz

if [ -z "$apple_key_p12_b64" -o -z "$altool_pass" ]; then
        echo "Could not find key to sign the application" 2>&1
        exit 1
fi

# Import keys
# Inspired by https://www.update.rocks/blog/osx-signing-with-travis/
KEY_CHAIN=build.keychain
KEY_CHAIN_PASS=build
KEY_FILE='CESNET, z. s. p. o..p12'
KEY_FILE_PASS=''
CERT_FILE='developerID_application.cer'
echo "$apple_key_p12_b64" | base64 -d > /tmp/cert.tar.bz2
tar -C /tmp -xJf /tmp/cert.tar.bz2
security create-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN
security default-keychain -s $KEY_CHAIN
security unlock-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN
security import "/tmp/$CERT_FILE"
security import "/tmp/$KEY_FILE" -A -P "$KEY_FILE_PASS"
security set-key-partition-list -S apple-tool:,apple: -s -k $KEY_CHAIN_PASS $KEY_CHAIN

# Sign appllication
# these need to be signed explicitly
for f in `find $APP/Contents/libs -type f`; do
        codesign --force --deep -s CESNET --options runtime -v $f
done
codesign --force --deep -s CESNET --options runtime -v $APP
#codesign --force --deep -s CESNET --options runtime -v $APP/Contents/MacOS/uv-qt

# Zip and send for notarization
ZIP_FILE=uv-qt.zip
UPLOAD_INFO_PLIST=/tmp/uplinfo.plist
REQUEST_INFO_PLIST=/tmp/reqinfo.plist
ditto -c -k --keepParent $APP $ZIP_FILE
xcrun altool --notarize-app --primary-bundle-id cz.cesnet.ultragrid.uv-qt --username $DEVELOPER_USERNAME --password "$altool_pass" --file $ZIP_FILE --output-format xml > $UPLOAD_INFO_PLIST

# Wait for notarization status
# Waiting inspired by https://nativeconnect.app/blog/mac-app-notarization-from-the-command-line/
SLEPT=0
TIMEOUT=7200
while true; do
        /usr/bin/xcrun altool --notarization-info `/usr/libexec/PlistBuddy -c "Print :notarization-upload:RequestUUID" $UPLOAD_INFO_PLIST` -u $DEVELOPER_USERNAME -p $altool_pass --output-format xml > $REQUEST_INFO_PLIST
        STATUS=`/usr/libexec/PlistBuddy -c "Print :notarization-info:Status" $REQUEST_INFO_PLIST`
        if [ $STATUS != "in progress" -o $SLEPT -ge $TIMEOUT ]; then
                break
        fi
        sleep 60
        SLEPT=$(($SLEPT + 60))
done
if [ $STATUS != success ]; then
        UUID=`/usr/libexec/PlistBuddy -c "Print :notarization-info:RequestUUID" $REQUEST_INFO_PLIST`
        xcrun altool --notarization-info $UUID -u $DEVELOPER_USERNAME -p $altool_pass
        echo "Could not notarize" 2>&1
        exit 1
fi

# If everything is ok, staple the app
xcrun stapler staple $APP

