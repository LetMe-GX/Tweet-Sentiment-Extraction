python run.py --model_type electra \
              --model_name_or_path google/electra-large-discriminator \
              --do_train \
              --do_eval \
              --per_gpu_train_batch_size 16 \
              --num_train_epochs 3 \
              --adversarial_fgm \
              --gradient_accumulation_steps 4 \
              --learning_rate 5e-5 \
              --warmup_steps 400