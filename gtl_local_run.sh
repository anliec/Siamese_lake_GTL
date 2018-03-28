#!/bin/bash

source ~/.bashrc

export PYTHONPATH=$HOME/Python3/bin

multiple_to_take=1
host=$HOSTNAME
num=${host:6}
num=$((num % multiple_to_take))

cmd_file="cmd_${num}.txt"
shuf cmd.txt > ${cmd_file}

i=0
while read line; do
    i=$((i + 1))
    out_dir=${line##*-o}
    if [[ -d "$out_dir" ]]; then
        echo ${out_dir} already run
    else
        echo running ${out_dir}
        python3.6 -m VGGsiamese ${line}
    fi
done < ${cmd_file}

