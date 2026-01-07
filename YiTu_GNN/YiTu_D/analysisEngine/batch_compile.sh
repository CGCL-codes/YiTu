#!/bin/bash
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13
do
   echo $i $[14-i]
   ./compile.sh  $i $[14-i]
done
