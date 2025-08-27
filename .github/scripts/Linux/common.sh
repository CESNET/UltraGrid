# $2 - pattern to exclude; separate packates with '\|' (BRE alternation)
get_build_deps_excl() {
        apt-cache showsrc "$1" | sed -n "/^Build-Depends:/\
{s/Build-Depends://;p;q}" | tr ',' '\n' | cut -f 2 -d\  | grep -v "$2"
}

