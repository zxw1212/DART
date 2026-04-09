respacing=''
guidance=5
batch_size=1
use_predicted_joints=0
stream_host='127.0.0.1'
stream_port=8765

model_list=(
'./mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt'
)

for model in "${model_list[@]}"; do
  python -m mld.rollout_demo \
    --denoiser_checkpoint "$model" \
    --batch_size $batch_size \
    --guidance_param $guidance \
    --respacing "$respacing" \
    --use_predicted_joints $use_predicted_joints \
    --stream_smplx 1 \
    --disable_viewer 1 \
    --stream_host "$stream_host" \
    --stream_port $stream_port
done
