datalist=(
#artaxor
#dior
#fish
#clipart1k
#neu-det
#uodd
foggy
)
shot_list=(
# 1
# 5
10
)
model_list=(
"l"
#"b"
#"s"
)
for model in "${model_list[@]}"; do
  for dataset in "${datalist[@]}"; do
    for shot in "${shot_list[@]}"; do
      CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --resume --num-gpus 1 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS output/vitl/foggy3/10shot_betamix_2000/model_final.pth SOLVER.MAX_ITER 500 DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/vit${model}/${dataset}3/${shot}shot_betamix_2000_softnms/
    done
  done
done
