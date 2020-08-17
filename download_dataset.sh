#!/bin/bash
#
# Script to download imagenet dataset
#
# Run this script with sudo privileges as it requires installation of additional packages
#
# This script downloads two files ->
# ILSVRC2012_img_train.tar ( about 138 GB )
# ILSVRC2012_img_val.tar ( about 6.3 GB )
#
# First install aria2, to allow faster concurrent downloading threads

sudo apt-get install -y aria2

# Create a directory for data, if not already present
mkdir -p data

# Move into the directory
cd data

# Create a folder for imagenet.
# Warning : To avoid mistakenly overwriting existing data, this script will exit if there already exist a folder for imagenet
mkdir imagenet

if [ $? -ne 0 ] ; then
	echo "Error : Imagenet directory already exists. To avoid mistakenly redownloading, you need to manually delete the imagenet folder before proceeding"
	exit 1
fi

cd imagenet

# Download the train tar file. This will take the most amount of time. It might take upto a day to download the training dataset.
aria2c -x 16 http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar

# Download the validation tar file. This might take upto an hour.
aria2c -x 16 http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar

# Return back to main folder
cd ../../

