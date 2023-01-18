#!/bin/bash
set -e

# Read input parameters
input=$1
sigma=$2
nr=$3
th_low=$4
th_high=$5
hsize=$6
wsize=$7
BIN=$8
HOME=$9

# Extract center from mask
if [ ! -f inpainting_data_0.txt ]; then
  stdout=$(python $BIN/.ipol/process_mask.py --input inpainting_data_0.txt)
else
  $ANT_CENTER_DETECTOR/build/AntColonyPith --animated=false --input $input
  stdout=$(python $BIN/.ipol/process_mask.py --input $input)
  echo 'success'
fi
Cy=$(echo $stdout | awk '{print $1}')
Cx=$(echo $stdout | awk '{print $2}')


# Execute algorithm
python $BIN/main.py --input $input --cx $Cx --cy $Cy --root $BIN --output_dir ./ --nr $nr --th_high $th_high --th_low $th_low --hsize $hsize --wsize $wsize --sigma $sigma

