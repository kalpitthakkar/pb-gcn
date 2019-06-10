#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters: Enter the path for saving weights"
    exit 1
fi

out_dir=$1
mkdir -p $1

wget -c -O $1/crossview_weights.pth https://www.dropbox.com/s/dlhyo4bsyquvk55/crossview_weights.pth?dl=0
wget -c -O $1/crosssubject_weights.pth https://www.dropbox.com/s/3n2rxosp78vdtj1/crosssubject_weights.pth?dl=0
