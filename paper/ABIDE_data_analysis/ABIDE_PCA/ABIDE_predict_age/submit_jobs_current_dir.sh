#! /bin/bash                                                                    

# for x in ./*.sh
# do
#         if [ "$x" != "$0" ]
#         then
#             sbatch $x
#         fi
# done

for x in ./*epa_ISJ*.sh
do
        if [ "$x" != "$0" ]
        then
            sbatch $x
        fi
done

for x in ./*Pearson*.sh
do
        if [ "$x" != "$0" ]
        then
            sbatch $x
        fi
done

for x in ./*skMI*.sh
do
        if [ "$x" != "$0" ]
        then
            sbatch $x
        fi
done