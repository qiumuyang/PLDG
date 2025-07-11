#!/bin/bash

# This function evaluates a model for a specific domain and dataset split.
#
# Parameters:
# - domain_id: The ID of the domain to evaluate, used to determine the domain's alphabetic representation.
# - split: The dataset split to evaluate, e.g., "train" or "val".
# - dir: The directory path containing the model checkpoints.
# - config: Array of configuration file paths for evaluation settings.
#
# The function converts the domain_id to an alphabetic character, checks
# the number of domains specified in the configuration, and constructs a
# range for evaluation. It then checks for the existence of the checkpoint
# file and proceeds to launch the evaluation using the `accelerate` tool,
# outputting the results to a CSV file and logs.

eval() {
    domain_id=$1
    split=$2
    dir=$3
    dir=$(realpath $dir)
    type=$4
    target=$5
    config=("${@:6}")
    domain_alpha=$(echo "$domain_id" | awk '{printf "%c", $1 + 65}')

    # target can be one / all
	if [ "$target" == "all" ]; then
		n_domains=$(grep "n_domains" $config | awk '{print $2}')
		if ! [[ $n_domains =~ ^[0-9]+$ ]]; then
			echo "Error: n_domains is not an integer, got $n_domains"
			exit 1
		fi
		domain_ranges=$(seq 0 $((n_domains - 1)) | paste -sd "," -)
	elif [ "$target" == "one" ]; then
		domain_ranges=$domain_id
	else
		domain_ranges=$target
	fi

    # echo "Evaluating domain $domain_alpha ($split) on GPU $CUDA_VISIBLE_DEVICES"
    # include domain ranges
    echo "Evaluating on domain $domain_ranges ($split) on GPU $CUDA_VISIBLE_DEVICES (Target $domain_alpha)"
    ckpt="${dir}/${domain_alpha}/model/${type}/model.safetensors"
    if [ ! -f $ckpt ]; then
        echo "Checkpoint not found: $ckpt"
        return
    fi
    output="${type}_${domain_alpha}"
    accelerate launch evaluate.py -c ${config[@]} \
        --seed 0 \
        --domain $domain_ranges \
        --split $split \
        $ckpt >$output.csv 2>$output.log &
}

if [ "$#" -lt 6 ]; then
    echo "Usage: $0 <devices> <domains> <dir> latest/best <target> <config> [<config> ...]"
    exit 1
fi
IFS=',' read -r -a devices <<<"$1"
IFS=',' read -r -a domains <<<"$2"
run_dir=$3
type=$4
target=$5
configs=("${@:6}")

if [ "$type" != "latest" ] && [ "$type" != "best" ]; then
    echo "Error: invalid type, must be 'latest' or 'best'"
    exit 1
fi

i=0
for domain in "${domains[@]}"; do
    dev1=${devices[$i % ${#devices[@]}]}
    CUDA_VISIBLE_DEVICES=$dev1 eval $domain "val" \
        $run_dir $type $target ${configs[@]}
    i=$((i + 1))
done

wait
echo "All evaluations are complete!"
