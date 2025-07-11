#!/bin/bash

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <entrypoint> <domain_ids> <devices> <output_dir> <config> [config ...]"
    exit 1
fi

SESSION="pldg"

tmux new-session -d -s $SESSION || {
    echo "Existing tmux session found. Please kill it first."
    tmux attach -t $SESSION
    exit 1
}

entrypoint=$1
domain_id_str=$2
device_str=$3
output_dir=$4
shift 4
config_files=$@

if [[ "$device_str" =~ /([0-9]+)$ ]]; then
    num_proc="${BASH_REMATCH[1]}"
    # Remove the trailing "/k" from device_str
    device_str="${device_str%/*}"
else
    num_proc=1
fi

# split device by comma
devices=(${device_str//,/ })
domains=(${domain_id_str//,/ })

device_idx=0
for i in "${!domains[@]}"; do
    device=""
    for ((j = 0; j < num_proc; j++)); do
        d=$(($device_idx % ${#devices[@]}))
        device="$device,${devices[$d]}"
        device_idx=$((device_idx + 1))
    done
    device=${device:1}
    domain_id=${domains[$i]}
    cmd="bash start.sh $entrypoint $domain_id $output_dir $config_files"
    env="CUDA_VISIBLE_DEVICES=$device"
    echo "[Domain $domain_id] $env $cmd"
    tmux new-window -t $SESSION:$((i + 1)) -n "Domain$domain_id" "$env $cmd; bash"
done
