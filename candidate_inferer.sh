#!/bin/bash
source ~/.virtualenvs/cv/bin/activate
# change environment right way, otherwise works on default env
python --version

models="COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
COCO-Detection/retinanet_R_101_FPN_3x.yaml
COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
"

videos=../data/selected_videos/*

m="COCO-Detection/retinanet_R_101_FPN_3x.yaml"
l=0.0002
i=1000


for v in $videos
do
    echo "infering video ${v} with model:${m} with lr: ${l} & iter: ${i}.........."
    echo "..........................."
    no_yaml_m=$(cut -d '.' -f 1 <<< $m)   # remove .extenstion from string
    w="../data/ball_output/${no_yaml_m}_lr${l}_iter${i}/model_final.pth" # finetuned model weight path
    python ball_inferer.py --config-name $m --video-input $v --output ../data/results/ball_tunes_thresh0.25/ --confidence-threshold 0.25 --opts MODEL.WEIGHTS $w SOLVER.MAX_ITER $i SOLVER.BASE_LR $l
done
