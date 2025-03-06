# srun -p rtx3090_slab -w slabgpu08 --gres=gpu:1 \
#     --job-name=test --kill-on-bad-exit=1 python3 main.py \
#     --image_path_0 ./assets/vangogh.jpg --image_path_1 ./assets/pearlgirl.jpg \
#     --prompt_0 "An oil painting of a man" --prompt_1 "An oil painting of a woman" \
#     --output_path "./results/vangogh_pearlgirl" --use_adain --use_reschedule \
#     --save_inter --use_lcm 

# srun -p rtx3090_slab -w slabgpu08 --gres=gpu:1 \
#     --job-name=test --kill-on-bad-exit=1 python3 main.py \
#     --image_path_0 ./assets/lion.png --image_path_1 ./assets/tiger.png \
#     --prompt_0 "A photo of a lion" --prompt_1 "A photo of a tiger" \
#     --output_path "./results/lion_tiger" --use_adain --use_reschedule \
#     --save_inter --use_lcm

srun -p rtx3090_slab -w slabgpu05 --gres=gpu:1 \
    --job-name=test --kill-on-bad-exit=1 python3 main.py \
    --image_path_0 ./assets/Trump.jpg --image_path_1 ./assets/Biden.jpg \
    --prompt_0 "A photo of an American man" --prompt_1 "A photo of an American man" \
    --output_path "./results/Trump_Biden" --use_adain --use_reschedule \
    --save_inter --use_lcm

