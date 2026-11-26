#!/bin/sh
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd) || exit 1
sh "$script_dir/run.sh" "$@"
result=$?
if [ "$result" -ne 0 ]; then
    printf '%s\n' 'Launch failed. Press Return to close.'
    read -r ignored
fi
exit "$result"
