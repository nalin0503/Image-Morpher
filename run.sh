srun -p rtx3090_slab -w slabgpu08 --gres=gpu:1 \
    --job-name=test --kill-on-bad-exit=1 python3 main_lcm.py \
    --image_path_0 ./assets/vangogh.jpg --image_path_1 ./assets/pearlgirl.jpg \
    --prompt_0 "An oil painting of a man" --prompt_1 "An oil painting of a woman" \
    --output_path "./results/vangogh_pearlgirl" --use_adain --use_reschedule \
    --save_inter --use_lcm
