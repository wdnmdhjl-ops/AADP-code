datalist=(
# artaxor
# dior
# fish
clipart1k
# neu-det
# uodd
# foggy
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
      CUDA_VISIBLE_DEVICES=0,1 python tools/demo.py --eval-only --num-gpus 2 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS output/vitl/clipart1k_10shot/model_final.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR cam/vit${model}/${dataset}_${shot}shot/
    done
  done
done
