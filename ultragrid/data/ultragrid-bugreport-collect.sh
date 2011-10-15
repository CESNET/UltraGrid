#!/bin/sh

PATH=/sbin:/bin:/usr/sbin:/usr/bin:/opt/local/bin:/opt/local/sbin:$PATH

LOG=ultragrid-bugreport.txt
export LC_ALL=C

test "`id -nu`" = root || echo "You SHOULD run this script as a root!!" >&2

(
exec > "$LOG" || error "cannot create $LOG file"
exec 2>&1
exec < /dev/null

set -x

if [ -f /etc/issue ]; then
        cat /etc/issue 
fi

if [ -x /usr/bin/sw_vers ]; then
        /usr/bin/sw_vers
fi

uname -a
sysctl -a

ifconfig

if [ -x /sbin/ethtool ]; then
        for n in `ls /sys/class/net/`; do
                ethtool $n
                ethtool -S $n
        done
fi


if [ -r /etc/sysctl.conf ]; then
    cat /etc/sysctl.conf
fi
netstat -ni
netstat -s
netstat -rn
if [ "`uname -s`" = "Darwin"]; then
        netstat -ln -f inet
        netstat -ln -f inet6
else
        netstat -ln -A inet
        netstat -ln -A inet6
fi
if [ -x /sbin/iptables ]; then
        iptables -L
fi

if [ -f /Library/Preferences/com.apple.alf ]; then
        defaults get /Library/Preferences/com.apple.alf globalstate
fi

ps -ef

glxinfo | grep OpenGL

if [ -x /usr/bin/lspci ]; then
        /usr/bin/lspci -vvvnn
fi
dmesg
)

gzip -f $LOG
echo >&2

echo Report $LOG.gz generated
chmod 666 $LOG.gz

