#!/bin/bash
x=0                         # start
n_samples=3                 # end
while [ $x -le $n_samples ]
do
  ../../build/exe1 ../../parms/$x.json
  x=$(( $x + 1 ))
done
