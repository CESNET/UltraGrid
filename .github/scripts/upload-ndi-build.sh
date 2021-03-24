#!/bin/sh -eu

ASSET=${1?Asset name to upload must be given}
NDI_REMOTE_SSH=${NDI_REMOTE_SSH:-'xpulec@frakira.fi.muni.cz:$HOME/Public/ug-ndi-builds'}
NDI_REMOTE_SSH_HOST_KEY_URL=${NDI_REMOTE_SSH_KEY_PUB_URL:-https://frakira.fi.muni.cz/~xpulec/ssh_known_hosts_github}
if [ -z "$SSH_KEY" ]; then
        echo "SSH private key required!" >&2
fi

mkdir -p ~/.ssh
curl -S $NDI_REMOTE_SSH_HOST_KEY_URL -o ~/.ssh/known_hosts
echo "$SSH_KEY" > ~/.ssh/id_rsa
chmod -R go-rwx ~/.ssh
scp $ASSET $NDI_REMOTE_SSH
