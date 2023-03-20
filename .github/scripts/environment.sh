#!/bin/sh

if expr "$GITHUB_REF" : 'refs/tags/'; then
  TAG=${GITHUB_REF#refs/tags/}
  VERSION=${TAG#v}
  CHANNEL=release
else
  VERSION=continuous
  TAG=$VERSION
fi

if [ -z ${CHANNEL-""} ]; then
        CHANNEL=$VERSION
fi

export CHANNEL TAG VERSION

printf '%b' "CHANNEL=$CHANNEL\nTAG=$TAG\nVERSION=$VERSION\n" >> "$GITHUB_ENV"

export FEATURES="--enable-option-checking=fatal --with-live555=/usr/local --enable-aja --enable-blank --enable-caca --enable-cineform --enable-decklink --enable-file --enable-gl --enable-gl-display --enable-holepunch --enable-jack --enable-jack-transport --enable-libavcodec --enable-natpmp --enable-ndi --enable-openssl --enable-pcp --enable-portaudio --enable-qt --enable-resize --enable-rtdxt --enable-rtsp --enable-rtsp-server --enable-scale --enable-sdl=2 --enable-sdl_mixer --enable-sdp-http --enable-soxr --enable-speexdsp --enable-swmix --enable-libswscale --enable-testcard-extras=all --enable-text --enable-video-mixer --enable-vulkan --enable-ximea --enable-zfec"
CUDA_FEATURES="--enable-cuda_dxt --enable-gpujpeg --enable-ldgm-gpu --enable-uyvy"
case "$RUNNER_OS" in
        Linux)
                FEATURES="$FEATURES $CUDA_FEATURES --enable-plugins --enable-alsa --enable-lavc-hw-accel-vaapi --enable-lavc-hw-accel-vdpau --enable-v4l2 --enable-screen=x11"
                ;;
        macOS)
                FEATURES="$FEATURES --enable-avfoundation --enable-coreaudio --enable-screen --enable-syphon"
                ;;
        Windows)
                FEATURES="$FEATURES $CUDA_FEATURES --enable-dshow --enable-screen --enable-spout --enable-wasapi"
                ;;
        *)
                echo "Unexpected runner OS: $RUNNER_OS" >&2
                exit 1
                ;;
esac
printf '%b' "FEATURES=$FEATURES\n" >> "$GITHUB_ENV"
