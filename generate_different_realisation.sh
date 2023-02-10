#!/bin/sh

set -e

img_name=GravellyMudD
folder="realisations_of_$img_name"
mkdir "$folder"

for i in $(seq 1 10); do
  diffusion_displacement_map remix /home/tin/research/one-shot-synthesis/datasets/displacement_maps/image/$img_name.png -f .3 -s 500x500
  mv remix_$img_name.png "$folder"/"$img_name"-$i.png
done

