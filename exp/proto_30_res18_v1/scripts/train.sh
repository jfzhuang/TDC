CONFIG_FILE=exp/proto_30_res18_v1/configs/proto_30_res18_v1.py
GPU_NUM=2

cd /home/zhuangjiafan/codes/TemProto
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}