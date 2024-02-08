# shellcheck shell=sh
#
# Exports common environment variables to next steps via $GITHU_ENV variable.
# Note that exporting the variable is not necessarily needed but it ensures that
# the vars are visible also later in current step (the script needs to be sourced,
# not run).

if expr "$GITHUB_REF" : 'refs/tags/' >/dev/null; then
  TAG=${GITHUB_REF#refs/tags/}
  VERSION=${TAG#v}
  CHANNEL=release
else
  VERSION=continuous
  TAG=$VERSION
fi

if [ -z "${CHANNEL-}" ]; then
        CHANNEL=$VERSION
fi

export CHANNEL TAG VERSION

printf '%b' "CHANNEL=$CHANNEL\nTAG=$TAG\nVERSION=$VERSION\n" >> "$GITHUB_ENV"

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
"
CUDA_FEATURES="--enable-cuda_dxt --enable-gpujpeg --enable-ldgm-gpu --enable-uyvy"
case "$RUNNER_OS" in
        Linux)
                FEATURES="$FEATURES $CUDA_FEATURES --enable-plugins\
 --enable-alsa --enable-lavc-hw-accel-vaapi --enable-lavc-hw-accel-vdpau\
 --enable-pipewire-audio --enable-v4l2"
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

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
        FEATURES="$FEATURES --enable-cineform"
fi

printf '%b' "FEATURES=$FEATURES\n" >> "$GITHUB_ENV"
