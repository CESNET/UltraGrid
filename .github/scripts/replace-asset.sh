#!/bin/sh -eux

DIR=$(dirname $0)

$DIR/delete-asset.sh "$@"
$DIR/upload-asset.sh "$@"

