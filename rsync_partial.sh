#!/bin/bash

# copy model only to remote

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <run_dir> <remote> [port]"
    exit 1
fi

run_dir=$1
remote=$2
run_dir=${run_dir%/}

# check possible 3rd argument is integer
if [[ $3 =~ ^[0-9]+$ ]]; then
    port=$3
else
    port=22
fi

regex="model/.*safetensors"
# use find to extract file list
find $run_dir -type f | grep -E $regex > file_list.txt
rsync -av -e "ssh -p $port" --files-from=file_list.txt . $remote
rm file_list.txt
