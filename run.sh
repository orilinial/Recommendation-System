#!/bin/bash

# Create run files
runfile_base="run_gpu"

gpu_nums=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 )
for gpunum in "${gpu_nums[@]}"
do
    echo '#!/bin/bash' > ${runfile_base}${gpunum}
    chmod 775 ${runfile_base}${gpunum}
done

real_num_of_gpus=4

gpu_idx=0
num_of_gpus=1
gpu_offset=0




#############################################
# Batchnorm effect (on the full dataset)
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_full" )
nn_arch_lst=( "fc_full_bn" )

mmnt_lst=( 0.9 )
weight_decay_lst=( 0 0.0005 )
lr_lst=( 0.001 1.0 2.0 3.0 4.0 )
dense_refeeding_iters_lst=( 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for weight_decay in "${weight_decay_lst[@]}"
                    do
                        for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                        do
                            gpus=$(( (gpu_offset + gpu_idx) ))
                            gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                            logfname="batchnorm_${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_wd${weight_decay}_derefed${dense_refeeding_iters}"
                            desc="LR=${lr}, mmnt=${mmnt}, wd=${weight_decay} ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                            echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --weight_decay ${weight_decay} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                            gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                        done
                    done
                done
            done
        done
    done
done


#############################################
# Weight decay effect (on the full dataset)
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_full" )
nn_arch_lst=( "fc_full" )

mmnt_lst=( 0.9 )
weight_decay_lst=( 0 0.0005 )
# We run with learning 2.0 for comparison with batchnorm
lr_lst=( 0.001 2.0 )
dense_refeeding_iters_lst=( 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for weight_decay in "${weight_decay_lst[@]}"
                    do
                        for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                        do
                            gpus=$(( (gpu_offset + gpu_idx) ))
                            gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                            logfname="weightdecay_${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_wd${weight_decay}_derefed${dense_refeeding_iters}"
                            desc="LR=${lr}, mmnt=${mmnt}, wd=${weight_decay} ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                            echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --weight_decay ${weight_decay} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                            gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                        done
                    done
                done
            done
        done
    done
done


#############################################
# Random input erasing effect (on the full dataset)
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_full" )
nn_arch_lst=( "fc_full" )

mmnt_lst=( 0.9 )
lr_lst=( 0.001 )
dense_refeeding_iters_lst=( 0 )
erase_input_probability_lst=( 0 0.05 0.1 0.2 0.4 0.7 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for erase_input_probability in "${erase_input_probability_lst[@]}"
                    do
                        for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                        do
                            gpus=$(( (gpu_offset + gpu_idx) ))
                            gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                            logfname="inputdrop_${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_derefed${dense_refeeding_iters}_erp${erase_input_probability}"
                            desc="LR=${lr}, mmnt=${mmnt}, ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}, erp=${erase_input_probability}"
                            echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --erase_input_probability ${erase_input_probability} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                            gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                        done
                    done
                done
            done
        done
    done
done


#############################################
# FC full, with/without refeeding
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_full" )
nn_arch_lst=( "fc_full" )

mmnt_lst=( 0.9 )
lr_lst=( 0.005 0.001 )
dense_refeeding_iters_lst=( 1 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                    do
                        gpus=$(( (gpu_offset + gpu_idx) ))
                        gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                        logfname="refeeding_${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_derefed${dense_refeeding_iters}"
                        desc="LR=${lr}, mmnt=${mmnt}, ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                        echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                        gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                    done
                done
            done
        done
    done
done


#############################################
#
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_3months" )
nn_arch_lst=( "fc3months" )

mmnt_lst=( 0.9 )
lr_lst=( 0.001 0.005 )
dense_refeeding_iters_lst=( 1 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                    do
                        gpus=$(( (gpu_offset + gpu_idx) ))
                        gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                        logfname="${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_derefed${dense_refeeding_iters}"
                        desc="LR=${lr}, mmnt=${mmnt}, ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                        echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                        gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                    done
                done
            done
        done
    done
done


#############################################
#
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_6months" )
nn_arch_lst=( "fc6months" )


mmnt_lst=( 0.9 )
lr_lst=( 0.001 0.005 )
dense_refeeding_iters_lst=( 1 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                    do
                        gpus=$(( (gpu_offset + gpu_idx) ))
                        gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                        logfname="${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_derefed${dense_refeeding_iters}"
                        desc="LR=${lr}, mmnt=${mmnt}, ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                        echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                        gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                    done
                done
            done
        done
    done
done


#############################################
#
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_1year" )
nn_arch_lst=( "fc1year" )


mmnt_lst=( 0.9 )
lr_lst=( 0.001 0.005 )
dense_refeeding_iters_lst=( 1 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                    do
                        gpus=$(( (gpu_offset + gpu_idx) ))
                        gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                        logfname="${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_derefed${dense_refeeding_iters}"
                        desc="LR=${lr}, mmnt=${mmnt}, ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                        echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                        gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                    done
                done
            done
        done
    done
done

#############################################
# Tie vs not-tied (on the full dataset)
#############################################
ver_vals=( 1 )
results_dir="res"

ds_lst=( "netflix_full" )
nn_arch_lst=( "fc_full" "fc_full_tied" )

mmnt_lst=( 0.9 )
lr_lst=( 0.001 )
dense_refeeding_iters_lst=( 0 )
epcs=100

for ver in "${ver_vals[@]}"
do
    for ds in "${ds_lst[@]}"
    do
        for nn_arch in "${nn_arch_lst[@]}"
        do
            for mmnt in "${mmnt_lst[@]}"
            do
                for lr in "${lr_lst[@]}"
                do
                    for dense_refeeding_iters in "${dense_refeeding_iters_lst[@]}"
                    do
                        gpus=$(( (gpu_offset + gpu_idx) ))
                        gpu_to_run=$(( (gpu_offset + gpu_idx) % real_num_of_gpus ))
                        logfname="tievsnot_${ds}_v${ver}_${nn_arch}_e${epcs}_mmnt${mmnt}_lr${lr}_derefed${dense_refeeding_iters}"
                        desc="LR=${lr}, mmnt=${mmnt}, ${nn_arch} on ${ds} with refeeding=${dense_refeeding_iters}"
                        echo CUDA_VISIBLE_DEVICES=${gpu_to_run} python main.py --logname ${logfname} --nn_arch ${nn_arch} --dense_refeeding_iters ${dense_refeeding_iters} --lr ${lr} --momentum ${mmnt} --dataset ${ds} --epochs ${epcs} --desc ${desc} >> ${runfile_base}${gpus}
                        gpu_idx=$(( ((gpu_idx + 1)) % num_of_gpus ))
                    done
                done
            done
        done
    done
done
