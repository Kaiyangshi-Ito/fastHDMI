#!/bin/bash
#SBATCH --account=def-masd
#SBATCH --cpus-per-task=30
#SBATCH --mem=200G
#SBATCH --time=160:00:00
#SBATCH --job-name=bgen2bed_missing_and_not

#############################################################################
#############################################################################
################### set up scripts, run on login shell ######################
#############################################################################
#############################################################################

nvidia-smi
lscpu

# install bigsnpr
# module load gcc/9.3.0 r/4.0.2 plink/1.9b_6.21-x86_64 java/13.0.2
# echo "install.packages(\"bigsnpr\")" >> install_bigsnpr.R
# Rscript  /home/kyang/install_bigsnpr.R
# module reset
# set up UKBB softlink
# ln -s /project/rpp-aevans-ab/neurohub/ukbb/ UKBB

#############################################################################
#############################################################################
################### create IIDs to be removed for removal ###################
#############################################################################
#############################################################################
cat << EOF > /home/kyang/relatedness_removal.py
import numpy as np
import pandas as pd

### relatedness removal
relatedness = pd.read_csv("/home/kyang/UKBB/genetics/ukb45551_rel_s488264.dat",
                          sep=" ")

ID1 = np.array(relatedness["ID1"])
ID2 = np.array(relatedness["ID2"])
Kinship = np.array(relatedness["Kinship"])
related1 = ID1[Kinship >= 0.354]
related2 = ID2[Kinship >= 0.354]
num_related = np.zeros_like(related2)
for j in np.arange(len(num_related)):
    num_related[j] = (np.sum(related1 == related2[j]) +
                      np.sum(related2 == related2[j]))

related1 = related1[np.argsort(num_related)[::-1]]
related2 = related2[np.argsort(num_related)[::-1]]
num_related = num_related[np.argsort(num_related)[::-1]]
_related = []

k = 0
for j in np.arange(len(related2)):
    if num_related[k] > 0:
        _related += [related2[k]]
        num_related -= (related1 == related2[k])
        for j in related1[related2 == related2[k]]:
            num_related -= (related2 == j)
#         num_related[0:k] = 0
    k += 1

related = np.array(_related)

print(related)
### to use only white British ancestry for data analysis
outcome_file = r"/home/kyang/UKBB/tabular/current.csv"

fields_all = ["eid"]
UKBB_all = pd.read_csv(outcome_file,
                       skipinitialspace=True,
                       usecols=fields_all,
                       encoding='unicode_escape').dropna()
_UKBB_all = np.array(list(UKBB_all.loc[:, "eid"]))

# since white British is already given, the following few lines are not needed
# fields0 = ["eid", "21000-0.0"]
# ethnic0 = pd.read_csv(outcome_file,
#                       skipinitialspace=True,
#                       usecols=fields0,
#                       encoding='unicode_escape').dropna()
# outcome0 = np.array(list(ethnic0.loc[:, "21000-0.0"]))
# preserve0 = np.array(list(ethnic0.loc[:, "eid"]))[outcome0 == 1001]

# fields1 = ["eid", "21000-1.0"]
# ethnic1 = pd.read_csv(outcome_file,
#                       skipinitialspace=True,
#                       usecols=fields1,
#                       encoding='unicode_escape').dropna()
# outcome1 = np.array(list(ethnic1.loc[:, "21000-1.0"]))
# preserve1 = np.array(list(ethnic1.loc[:, "eid"]))[outcome1 == 1001]

# fields1 = ["eid", "21000-2.0"]
# ethnic1 = pd.read_csv(outcome_file,
#                       skipinitialspace=True,
#                       usecols=fields1,
#                       encoding='unicode_escape').dropna()
# outcome2 = np.array(list(ethnic2.loc[:, "21000-2.0"]))
# preserve2 = np.array(list(ethnic2.loc[:, "eid"]))[outcome2 == 1001]

# preserve = np.concatenate((preserve0, preserve1, preserve2))

fields = ["eid", "22006-0.0"]
ethnic = pd.read_csv(outcome_file,
                     skipinitialspace=True,
                     usecols=fields,
                     encoding='unicode_escape').dropna()
preserve = np.array(list(ethnic.loc[:, "eid"]))

_also_to_be_removed = np.setdiff1d(_UKBB_all, preserve, assume_unique=False)
_also_to_be_removed = _also_to_be_removed.astype(int)

related = np.concatenate((related, _also_to_be_removed))
related = np.unique(related)




with open("/home/kyang/related.csv", "w") as output:
    output.write("\n".join(map(str, related)))
EOF
module load gcc llvm rust python
source $HOME/jupyter_py3/bin/activate
python /home/kyang/relatedness_removal.py

module reset

# create file for UKBB observations removal
cat /home/kyang/UKBB/withdrawals/*.csv /home/kyang/related.csv > /home/kyang/_UKBB_removals.csv
paste /home/kyang/_UKBB_removals.csv /home/kyang/_UKBB_removals.csv > /home/kyang/UKBB_removals.csv # to create the plink family format
rm /home/kyang/_UKBB_removals.csv # remove temporary files created
rm /home/kyang/related.csv
rm /home/kyang/relatedness_removal.py

#############################################################################
#############################################################################
################## convert bgen files to bed, first time ####################
#############################################################################
#############################################################################

module load plink/2.00-10252019-avx2
cd /home/kyang/UKBB/genetics/imp

for j in {1..22}
do
plink2 \
  --bgen "ukb_imp_chr${j}_v3.bgen" 'ref-first'\
  --sample "ukb45551_imp_chr${j}_v3_s487296.sample" \
  --memory 180000 \
  --hard-call-threshold 0.3 \
  --mind 0.1 \
  --geno 0.1 \
  --maf 0.05 \
  --hwe 0.0000000001 \
  --max-alleles 2 \
  --remove /home/kyang/UKBB_removals.csv \
  --make-bed \
  --out "/home/kyang/scratch/chr${j}"
done

module reset

#############################################################################
#############################################################################
###################### merge bed, first attempt #############################
#############################################################################
#############################################################################

module load StdEnv/2018.3 plink/1.9b_5.2-x86_64
cd /home/kyang/scratch

# ================Merge all chr into one file ==================================
# The file allchrs.txt is a file with each chromosmoe file on a separatsqe line, for example :
# directory/chr2
# directory/chr3
# .
# .
# .
# directory/chr22

plink \
  --bed chr1.bed \
  --bim chr1.bim \
  --fam chr1.fam \
  --memory 180000 \
  --mind 0.1 \
  --geno 0.1 \
  --maf 0.05 \
  --hwe 0.0000000001 \
  --remove /home/kyang/UKBB_removals.csv \
  --merge-list /home/kyang/merging_bedfrombgen.txt \
  --make-bed \
  --out /home/kyang/scratch/merged

module reset

#############################################################################
#############################################################################
########### convert bgen to bed, second time, without missing snps ##########
#############################################################################
#############################################################################

module load plink/2.00-10252019-avx2
cd /home/kyang/UKBB/genetics/imp

for j in {1..22}
do
plink2 \
  --bgen "ukb_imp_chr${j}_v3.bgen" 'ref-first'\
  --sample "ukb45551_imp_chr${j}_v3_s487296.sample" \
  --memory 180000 \
  --hard-call-threshold 0.3 \
  --mind 0.1 \
  --geno 0.1 \
  --maf 0.05 \
  --hwe 0.0000000001 \
  --max-alleles 2\
  --remove /home/kyang/UKBB_removals.csv \
  --exclude /home/kyang/scratch/merged-merge.missnp \
  --make-bed \
  --out "/home/kyang/scratch/chr${j}"
done

module reset

#############################################################################
#############################################################################
############ merge bed, second time, without missing snps ###################
#############################################################################
#############################################################################
# this should give a bed file with some missing data

module load StdEnv/2018.3 plink/1.9b_5.2-x86_64
cd /home/kyang/scratch

# ================Merge all chr into one file ==================================
# The file allchrs.txt is a file with each chromosmoe file on a separatsqe line, for example :
# directory/chr2
# directory/chr3
# .
# .
# .
# directory/chr22

plink \
  --bed chr1.bed \
  --bim chr1.bim \
  --fam chr1.fam \
  --memory 180000 \
  --mind 0.1 \
  --geno 0.1 \
  --maf 0.05 \
  --hwe 0.0000000001 \
  --merge-list /home/kyang/merging_bedfrombgen.txt \
  --remove /home/kyang/UKBB_removals.csv \
  --make-bed \
  --out /home/kyang/scratch/merged_with_missing

module reset

rm /home/kyang/scratch/chr*.* # remove the individual files since merging has finished
#############################################################################
#############################################################################
############ merge bed, second time, without missing snps ###################
#############################################################################
#############################################################################
# this returns a bed file with no missing data

module load StdEnv/2018.3 plink/1.9b_5.2-x86_64
cd /home/kyang/scratch

# ================Merge all chr into one file ==================================
# The file allchrs.txt is a file with each chromosmoe file on a separatsqe line, for example :
# directory/chr2
# directory/chr3
# .
# .
# .
# directory/chr22

plink \
  --bed merged_with_missing.bed \
  --bim merged_with_missing.bim \
  --fam merged_with_missing.fam \
  --memory 180000 \
  --mind 0.1 \
  --geno 0.0 \
  --maf 0.05 \
  --hwe 0.0000000001 \
  --remove /home/kyang/UKBB_removals.csv \
  --make-bed \
  --out /home/kyang/scratch/merged

module reset
