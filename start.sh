#!/bin/bash

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <entrypoint> <domain_id> <output_dir> <config> [config ...] [-- <extra_args>]"
    exit 1
fi

entrypoint=$1
domain_id=$2
output_dir=$3
shift 3

config_files=()
extra_args=()
found_separator=0
for arg in "$@"; do
    if [[ "$arg" == "--" && $found_separator -eq 0 ]]; then
        found_separator=1
        continue
    fi
    if [ $found_separator -eq 0 ]; then
        config_files+=($arg)
    else
        extra_args+=($arg)
    fi
done

id_to_alpha=({A..Z})
domain_alpha=${id_to_alpha[$domain_id]}

FREE_PORT=$(python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# if has extra args, show them
if [ ${#extra_args[@]} -gt 0 ]; then
    print_extra="with: ${extra_args[@]}"
fi

echo "Starting $entrypoint with domain $domain_alpha on port $FREE_PORT $print_extra"

PYTHONPATH=. accelerate launch --main_process_port $FREE_PORT $entrypoint \
    --config configs/exp/shared.yaml ${config_files[@]} \
    --seed 0 \
    --domain $domain_id \
    --output $output_dir/$domain_alpha \
    ${extra_args[@]}
