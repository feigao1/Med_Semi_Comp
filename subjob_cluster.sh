#!/bin/bash
nsub_VALUES="500 1000 2000 4000"

seed_VALUES="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100"

for nsub in $nsub_VALUES; do
    for seed in $seed_VALUES; do
        JOB_SUFFIX="${nsub}"
        FILE_SUFFIX="n_${nsub}_file_${seed}"
        SBATCH_JOB="#SBATCH --job-name=${JOB_SUFFIX}"
        NUM="--nsub $nsub --nrep 10 --seed $seed --nfile $seed "
        CMD_START="srun -o ${FILE_SUFFIX}_out.txt"
        CMD_END="Semi_Comp_Simu $NUM "
        CMD="$CMD_START $CMD_END"
        rm -rf tmp.out
        echo "#!/bin/bash" >> tmp.out
        echo "$SBATCH_PARTITION" >> tmp.out
        echo "$SBATCH_JOB" >> tmp.out
        echo "$CMD" >> tmp.out
        chmod 700 tmp.out
        #cat tmp.out
        sbatch tmp.out
        rm -rf tmp.out
    done
done
