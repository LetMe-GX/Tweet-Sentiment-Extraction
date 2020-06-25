CUDA_VISIBLE_DEVICES=1 python run.py --model_type electra \
                                     --model_name_or_path google/electra-base-discriminator \
                                     --do_train \
                                     --do_eval \
                                     --per_gpu_train_batch_size 8 \
                                     --num_train_epochs 2 \
                                     --learning_rate 3e-5 \
                                     --warmup_steps 400 \
                                     --overwrite_output_dir \
                                     --output_dir ./electra_base