#!/bin/sh

working_dir=$(pwd)

cd "$(dirname "$0")"

python param_save.py

python ../gninvert/hpjob.py -p parameters -m 3layer -g SingleDiffusionGN -o hpresults1

cd $working_dir
