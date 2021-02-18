## This file tries to find cl.exe by using vswhere if nvcc was found in $PATH. This
## is a prerequisity in MSW. Does nothing if cl.exe already in $PATH or given explicitly
## by --with-cuda-host-compiler (obviously also when not in MSW or there is not CUDA).
##
## @note
## This file works with parses arguments passed autogen.sh if the path for cl.exe
## was not given explicitly. Thus it must be sourced by autogen.sh, not executed.

if [ -z "$ORIGDIR" ]; then
        echo "Source this file from autogen.sh, not call!"
        exit 1
fi

# find MSVC if CUDA is present but no cl in PATH, don't override --with-cuda-host-compiler if explicit
cuda_host_compiler_arg_present() {
        while expr $# \> 0 >/dev/null; do
                if expr "x$1" : x--with-cuda-host-compiler >/dev/null; then
                        echo yes
                fi
                shift
        done
        echo no
}
is_win() {
        SYS=$(uname -s)
        if expr $SYS : "MSYS" >/dev/null; then
                echo yes
        fi
        echo no
}
if [ "$(is_win)" = "yes" -a "$(cuda_host_compiler_arg_present $@)" = no ]; then
        CUDA_PRESENT=$(command -v nvcc >/dev/null && echo yes || echo no)
        CL_PRESENT=$(command -v cl >/dev/null && echo yes || echo no)
        if [ $CUDA_PRESENT = yes -a $CL_PRESENT = no ]; then
                VSWHERE="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
                INSTALL_DIR=$("$VSWHERE" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
                VERSION_FILE="$INSTALL_DIR/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"
                if [ -f "$VERSION_FILE" ]; then
                        VERSION=$(cat "$VERSION_FILE")
                        PATH=$PATH:$(cygpath "$INSTALL_DIR/VC/Tools/MSVC/$VERSION/bin/HostX64/x64")
                        PATH_TO_CL=$(command -v cl.exe)
                        set -- "$@" "--with-cuda-host-compiler=$PATH_TO_CL"
                fi
        fi
fi

