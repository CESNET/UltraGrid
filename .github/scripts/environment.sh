# shellcheck shell=sh
#
## Exports common environment variables to next steps via $GITHU_ENV variable.
## Note that exporting the variable is not necessarily needed but it ensures that
## the vars are visible also later in current step (the script needs to be sourced,
## not run).
##
## Environment variables:
## - **apple_key_p12_b64** - [mac only] base64-encoded $KEY_FILE (using
##                           password $KEY_FILE_PASS)

if expr "$GITHUB_REF" : 'refs/tags/' >/dev/null; then
  TAG=${GITHUB_REF#refs/tags/}
  VERSION=${TAG#v}
  CHANNEL=release
else
  VERSION=continuous
  TAG=$VERSION
fi

# include platform on mac in version string
if [ "$(uname -s)" = Darwin ]; then
        VERSION="$VERSION-$(uname -m)"
fi

if [ -z "${CHANNEL-}" ]; then
        CHANNEL=$VERSION
fi

export CHANNEL TAG VERSION

printf '%b' "CHANNEL=$CHANNEL\nTAG=$TAG\nVERSION=$VERSION\n" >> "$GITHUB_ENV"

## @note `uname -m` is x86_64 for Linux ARM builds, because this script is
##       not called from the build chroot, so utilize GITHUB_WORKFLOW
is_arm() {
        [ "$(uname -m)" = arm64 ] || [ "$GITHUB_WORKFLOW" = 'ARM builds' ]
}

export FEATURES="\
 --enable-option-checking=fatal\
 --with-live555=/usr/local\
 --enable-aja\
 --enable-blank\
 --enable-caca\
 --enable-decklink\
 --enable-file\
 --enable-gl\
 --enable-gl-display\
 --enable-holepunch\
 --enable-jack\
 --enable-jack-transport\
 --enable-libavcodec\
 --enable-natpmp\
 --enable-ndi\
 --enable-openssl\
 --enable-pcp\
 --enable-portaudio\
 --enable-qt\
 --enable-resize\
 --enable-rtdxt\
 --enable-rtsp\
 --enable-rtsp-server\
 --enable-scale\
 --enable-screen\
 --enable-sdl=2\
 --enable-sdl_mixer\
 --enable-sdp-http\
 --enable-soxr\
 --enable-speexdsp\
 --enable-swmix\
 --enable-libswscale\
 --enable-testcard-extras=all\
 --enable-text\
 --enable-video-mixer\
 --enable-vulkan\
 --enable-ximea\
 --enable-zfec\
 --disable-drm_disp\
"
CUDA_FEATURES="--enable-cuda_dxt --enable-gpujpeg --enable-ldgm-gpu --enable-uyvy"
case "$RUNNER_OS" in
        Linux)
                FEATURES="$FEATURES --enable-plugins --enable-alsa \
--enable-pipewire-audio --enable-v4l2 --enable-lavc-hw-accel-vaapi"
                if is_arm; then
                        FEATURES="$FEATURES --disable-qt"
                else
                        FEATURES="$FEATURES $CUDA_FEATURES \
--enable-lavc-hw-accel-vdpau"
                fi
                ;;
        macOS)
                FEATURES="$FEATURES --enable-avfoundation --enable-coreaudio --enable-syphon"
                ;;
        Windows)
                FEATURES="$FEATURES $CUDA_FEATURES --enable-dshow --enable-spout --enable-wasapi"
                ;;
        *)
                echo "Unexpected runner OS: ${RUNNER_OS:-(undefined)}" >&2
                return 1
                ;;
esac

if ! is_arm; then
        FEATURES="$FEATURES --enable-cineform"
fi

printf '%b' "FEATURES=$FEATURES\n" >> "$GITHUB_ENV"
# populate /etc/environment-defined var to global env
# shellcheck disable=SC2154 # defined by runner in /etc/environment
printf '%b' "ImageOS=$ImageOS\n" >> "$GITHUB_ENV"

if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" != arm64 ]; then
        export UG_ARCH=-msse4.2
        printf "UG_ARCH=%s\n" $UG_ARCH >> "$GITHUB_ENV"
fi

import_signing_key() {
        if [ "$(uname -s)" != Darwin ] || [ -z "$apple_key_p12_b64" ]; then
                return 0
        fi
        # Inspired by https://www.update.rocks/blog/osx-signing-with-travis/
        KEY_CHAIN=build.keychain
        KEY_CHAIN_PASS=build
        KEY_FILE=/tmp/signing_key.p12
        KEY_FILE_PASS=dummy
        echo "$apple_key_p12_b64" | base64 -d > $KEY_FILE
        security create-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN || true
        security default-keychain -s $KEY_CHAIN
        security unlock-keychain -p $KEY_CHAIN_PASS $KEY_CHAIN
        security import "$KEY_FILE" -A -P "$KEY_FILE_PASS"
        security set-key-partition-list -S apple-tool:,apple: -s -k $KEY_CHAIN_PASS $KEY_CHAIN
        printf '%b' "KEY_CHAIN_PASS=$KEY_CHAIN_PASS\nKEY_CHAIN=$KEY_CHAIN\n" \
                >> "$GITHUB_ENV"
}
import_signing_key

printf '%b' 'DELTA_MAC_ARCHIVE=videomaster-macos-dev.tar.gz\n' >> "$GITHUB_ENV"

git config --global user.name "UltraGrid Builder"
git config --global user.email "ultragrid@example.org"
