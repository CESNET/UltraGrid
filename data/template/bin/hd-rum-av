#!/bin/sh -eu
##
## Wrapper for (transcoding) reflector runnint both audio
## and video instance simoultaneously. Base parameters are
## those used for video transcoding reflector, audio parameters
## are deduced from that (the audio reflector is mostly just a
## dummy packet reflector).
##

s_ld_library_path=${LD_LIBRARY_PATH-}
if [ "${APPIMAGE-}" ]; then
  LD_LIBRARY_PATH=
fi

exe=$(command -v "$0")
dir=$(dirname "$exe")
if ! command -v uname >/dev/null || [ "$(uname -o)" = Msys ]; then
  exeext=.exe
fi
reflector=$dir/hd-rum-transcode${exeext-}
uv=$dir/uv${exeext-}
basename=$(basename "$0")

bold=$(tput bold || true)
green=$(tput setaf 2 || true)
yellow=$(tput setaf 3 || true)
magenta=$(tput setaf 5 || true)
reset=$(tput sgr0 || true)

video_term_prefix="${magenta}[V]"
audio_term_prefix="${yellow}[A]"

adjust_port() {
  if expr "$1" : ".*:" >/dev/null; then
    rx=$(echo "$1" | cut -d: -f 1)
    tx=$(echo "$1" | cut -d: -f 2)
    printf "%s" $((rx + 2)):$((tx + 2))
  else
    printf "%s" $(($1 + 2))
  fi
}

run_reflector() {
  medium=$1
  line_prefix=$2
  shift 2
  printf "%s %s %s\n" "${bold}Running" "$medium" "reflector: ${green}$reflector $*${reset}"
  LD_LIBRARY_PATH=$s_ld_library_path "$reflector" "$@" |
    sed "s/^/$line_prefix$reset /" &
}

run_audio_conference() {
  port=$1
  echo "${bold}Running audio conference: ${green}$uv -P $port \
-r mixer $*${reset}"
  LD_LIBRARY_PATH=$s_ld_library_path "$uv" -P "$port" -r mixer |
    sed "s/^/$audio_term_prefix$reset /" &
}


# loops over $@ and adjusts (mailny ports) or dismisses unused video options
run_audio() {
  # translate long opts to short opts
  # shellcheck disable=SC2034
  for n in $(seq $#); do
    val=$1
    case "$val" in
    --help) val=-h;;
    --server) val=-S;;
    --verbose) val=-V;;
    --version) val=-v;;
    --param) val=-O;;
    --blend) val=-B;;
    --control-port) val=-n;;
    --conference) val=-r;;
    --conference-compression) val=-R;;
    --capture-filter) val=-F;;
    esac
    shift
    set -- "$@" "$val"
  done

  args=
  while getopts S:vVO:B:n:r:R:F: name; do
    case "$name" in
      O|S) args="$args -$name $OPTARG";;
      V) args="$args -$name";;
      v) return;;
      B|F|R|n) ;; # video opts
      r) conference=1;;
      *)
        echo "Unsupported global option!" >&2
        exit 1
        ;;
    esac
  done
  if [ $OPTIND -ge $# ]; then
      echo "Missing bufsize/port!"
      exit 1
  fi
  eval bufsize="\${$OPTIND}"
  eval port="\${$((OPTIND+1))}"
  args="$args ${bufsize?} $((${port?} + 2))"
  OPTIND=$((OPTIND+2))

  if [ "${conference-}" ]; then
    run_audio_conference "$port"
    return
  fi

  while [ $OPTIND -le $# ]; do
    if getopts 46P:c:f:m:l: name; then
      case "$name" in
        c|f|m|l) ;; # video opts
        4|6) args="$args -$name";;
        P) args="$args -P $(adjust_port "$OPTARG")";;
        *)
          echo "Unsupported host option!" >&2
          exit 1
          ;;
      esac
    else
      eval "host=\${$OPTIND}"
      args="$args ${host?}"
      OPTIND=$((OPTIND + 1))
    fi
  done
  # TODO - if args with spaces required, use sth like `eval arg$((argc+=1))=b`
  # shellcheck disable=SC2086 # see the above TODO
  run_reflector audio "$audio_term_prefix" $args
}

atexit() {
  echo "Exit $basename"
}

sigaction() {
  trap '' TERM # inhibit following signal to ourselves
  if ps -o cmd >/dev/null 2>&1; then
    pgid=$(ps -o pgid= -p $$ | tr -d ' ')
  else # MSW dowsn't have "ps -o"
    pgid=$$
  fi
  if [ $$ -eq "$pgid" ]; then
    kill -- -$$
  else
    echo "pgid $pgid not pid of the script ($$), not sending kill" \
      "(ok if in firejail)" >&2
  fi
  trap - INT TERM
  wait
}

usage() {
  printf "Reflector wrapper to run concurrenty both audio and video\n\n"
  printf "%b" "Options set is basically the same as for \
${bold}hd-rum-transcode$reset,\naudio will be adjusted automatically.\n"
  printf "\n"
  printf "Both server and conference mode is supported.\n\n"
  echo Usage:
  printf "%b\n" "\t${bold}$0${reset} <reflector arg>"
  printf "%b\n\n" "\t${bold}$0${reset} -H"

  echo Examples:
  printf "%b\n" "\t${bold}$0 8M 5004 curtis -c lavc:enc=libx264 ormal -P 5008 \
100::1${reset}"
  printf "%b\n" "\t\twill send audio and video unchanged to \
${bold}curtis$reset and ${bold}100::1$reset and transcode video for \
${bold}ormal$reset"
  printf "%b\n" "\t$bold$0 -S 5004 8M 6004$reset"
  printf "%b\n" "\t\tuse ${bold}server mode$reset receiving AV on 6004/6006; \
clients connect to default ports 5004/5006"
  printf "%b\n" "\t$bold$0 -r 1920:1080:30 5004$reset"
  printf "%b\n" "\t\tconference mode"
  printf "\n"

  printf "%b\n" "Use $bold-H$reset for reflector (hd-rum-transcode) help."
}

if [ $# = 0 ] || [ "$1" = -h ] || [ "$1" = --help ]; then
  usage
  exit 0
fi

if [ ! -x "$reflector" ]; then
  echo "$reflector not found or not executable!" >&2
  exit 2
fi

if [ "$1" = -H ]; then
  exec "$reflector" -h
fi

trap atexit EXIT
trap sigaction INT TERM

if tty >/dev/null && [ "${ULTRAGRID_COLOR_OUT-unset}" = unset ]; then
  # enforce color output even though output piped
  export ULTRAGRID_COLOR_OUT=1
fi

echo "$basename PID: $$"
run_audio "$@"
run_reflector video "$video_term_prefix" "$@"

wait
