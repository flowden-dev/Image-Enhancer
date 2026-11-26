#!/bin/sh
# launch from any working directory, including paths containing spaces
set -eu
script_dir=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$script_dir"
for interpreter in python3.13 python3.14 python3.12 python3.11 python3; do
    if command -v "$interpreter" >/dev/null 2>&1 && "$interpreter" -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info < (3, 15) and sys.maxsize > 2**32))' >/dev/null 2>&1; then
        exec "$interpreter" "$script_dir/bootstrap.py" "$@"
    fi
done
printf '%s\n' 'Install 64-bit Python 3.11–3.14 with Tkinter (3.13 recommended).'
exit 1
