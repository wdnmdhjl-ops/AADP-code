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
      CUDA_VISIBLE_DEVICES=0 python demo/demo.py --num-gpus 1 --config-file configs/${dataset}/vit${model}_shot${shot}_${dataset}_finetune.yaml MODEL.WEIGHTS output/vitl/foggy_10shot_13_27/model_final.pth DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml OUTPUT_DIR output/vit${model}/${dataset}_${shot}shot_02/
    done
  done
done
