#!/bin/bash

if [ "$#" -lt 5 ]; then
	echo "Usage: $0 <entrypoint> <domain_ids> <devices> <output_dir> <config> [config ...]"
	exit 1
fi

SESSION="seppldg"

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

# split device by comma
devices=(${device_str//,/ })
domains=(${domain_id_str//,/ })

# read n_domains from config files
n_domains=$(cat $config_files | grep "n_domains" | awk '{print $2}')
n_tasks=$((n_domains - 1))

# TD: T0 T1 T2 T3 T4
# 0   1  2  3  4  5
# 1   0  2  3  4  5
# 2   0  1  3  4  5
# 3   0  1  2  4  5
# 4   0  1  2  3  5

k=0
for domain_id in "${domains[@]}"; do
	task_id1=$(($domain_id - 1))
	task_id2=$domain_id
	if [ $task_id1 -ge 0 ]; then
		j=$((k % ${#devices[@]}))
		device=${devices[$j]}
		cmd="bash start.sh $entrypoint $domain_id $output_dir $config_files -- --task-id=$task_id1"
		env="CUDA_VISIBLE_DEVICES=$device"
		echo "[Domain $domain_id Task $task_id1] $env $cmd"
		tmux new-window \
			-t $SESSION:$((k + 1)) \
			-n "Domain$domain_id-Task$task_id1" "$env $cmd; bash"
		k=$((k + 1))
	fi

	if [ $task_id2 -lt $n_tasks ]; then
		j=$((k % ${#devices[@]}))
		device=${devices[$j]}
		cmd="bash start.sh $entrypoint $domain_id $output_dir $config_files -- --task-id=$task_id2"
		env="CUDA_VISIBLE_DEVICES=$device"
		echo "[Domain $domain_id Task $task_id2] $env $cmd"
		tmux new-window \
			-t $SESSION:$((k + 1)) \
			-n "Domain$domain_id-Task$task_id2" "$env $cmd; bash"
		k=$((k + 1))
	fi

done
