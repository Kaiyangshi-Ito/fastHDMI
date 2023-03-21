#! /bin/bash                                                                    

for x in ./*.sh
do
        if [ "$x" != "$0" ]
        then
            sbatch $x
        fi
done