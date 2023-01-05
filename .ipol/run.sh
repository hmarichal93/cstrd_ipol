#!/bin/bash
set -e

# Read input parameters
input=$1
mask=$2
sigma=$3
nr=$4
th_low=$5
th_high=$6
hsize=$7
wsize=$8
BIN=${9}
HOME=${10}

# Extract center from mask


Cx=1240
Cy=1260

# Execute algorithm
python $BIN/main.py --input $input --cx $Cx --cy $Cy --output $HOME --nr $nr --th_high $th_high --th_low $th_low --hsize $hsize --wsize $wsize --sigma $sigma

