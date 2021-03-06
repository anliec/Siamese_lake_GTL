#!/bin/bash

# script designed to run the cmd.txt file generated by siamese_network/batch_generator.py
# It may not run on other session than my one at GTL due to my particular Python environment
# In such case you probably want to change the source and export line of this script.

RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

source /home/GTL/nsix/.bashrc
export PYTHONPATH=$HOME/Python3/bin

host=$HOSTNAME
num=${host:6}

cmd_file="cmd_${num}.txt"
shuf cmd.txt > ${cmd_file}

i=0
while read line; do
    i=$((i + 1))
    out_dir=${line##*-o }
    if [[ -d "$out_dir" ]]; then
        echo -e "${RED}${out_dir} already run${NC}"
    else
        echo -e "${BLUE}running ${out_dir}, with line: ${line}${NC}"
        nice python3.6 -m VGGsiamese ${line}
        if [[ "$?" != "0" ]]; then
            echo -e "${RED}Error on python script execution, please read logs${NC}"
            exit
        fi
    fi
done < ${cmd_file}

