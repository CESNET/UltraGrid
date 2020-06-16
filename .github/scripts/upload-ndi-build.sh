#!/bin/sh -eu

ASSET=${1?Asset name to upload must be given}
REMOTE=${REMOTE:-xpulec@frakira.fi.muni.cz}
RDIR=${RDIR:-'$HOME/Public/ug-ndi-builds'}
SSH_HOST_KEY_PUB_URL=${SSH_HOST_KEY_PUB:-https://frakira.fi.muni.cz/~xpulec/ssh_known_hosts_github}
if [ -z "$SSH_KEY" ]; then
        echo "SSH private key required!" >&2
fi

mkdir -p ~/.ssh
curl -S $SSH_HOST_KEY_PUB_URL -o ~/.ssh/known_hosts
echo "$SSH_KEY" > ~/.ssh/id_rsa
chmod -R go-rwx ~/.ssh
scp $ASSET $REMOTE:$RDIR
