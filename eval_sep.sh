#!/bin/bash

eval() {
	domain_id=$1
	split=$2
	type=$3
	target=$4
	dir=$5
	dir=$(realpath $dir)
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

	echo "Evaluating domain $domain_alpha ($split) on GPU $CUDA_VISIBLE_DEVICES"
	ckpt="${dir}/"
	if [ ! -d $ckpt ]; then
		echo "Checkpoint not found: $ckpt"
		return
	fi
	output="${type}_${domain_alpha}"
	accelerate launch evaluate_sep.py -c $config \
	    --seed 0 \
	    --domain $domain_ranges \
	    --split $split \
		--target-domain $domain_id \
		--type $type \
	    $ckpt >$output.csv 2>$output.log &
}

# <devices> <domains>
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
	dev1=${devices[i % ${#devices[@]}]}
	CUDA_VISIBLE_DEVICES=$dev1 eval $domain "val" $type $target $run_dir ${configs[@]}
	i=$((i + 1))
done

wait
echo "All evaluations are complete!"
