#!/bin/bash


# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
export HCCL_CONNECT_TIMEOUT=1200
# export ASCEND_LAUNCH_BLOCKING=1
export CPU_AFFINITY_CONF=1 # 绑核
export TASK_QUEUE_ENABLE=2 # 优化下发队列
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="2"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
export PYTHONPATH=./

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(1)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
# shellcheck disable=SC2145
echo "Parsed device_ids: ${device_ids[@]}"

stage=0
stop_stage=0

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
#HOST_NODE_ADDR=192.168.0.38
HOST_NODE_ADDR=127.0.0.1
HOST_PORT=29401
# HOST_NODE_ADDR="127.0.0.1:29401"
num_nodes=1
job_id=2023

train_config=conf/ct_config_contextasr.yaml
#gxl_data_json_info_path_s2t=conf/empty.yaml
#gxl_data_json_info_path_t2s=conf/empty.yaml
#gxl_data_json_info_path_s2s=conf/data_s2s_tmp.yaml
#gxl_data_json_info_path_t2t=conf/empty.yaml
# ---------------------------------

gxl_data_json_info_path_s2t=conf/data_s2t_contextasr.yaml
gxl_data_json_info_path_t2s=conf/empty.yaml
gxl_data_json_info_path_s2s=conf/empty.yaml
gxl_data_json_info_path_t2t=conf/data_t2t.yaml

# 自然语言think的训练数据
# epoch0 是 使用kokoro数据训练的，在内容层面先用kokoro数据训练出效果，先不管音质。
# epoch1 是 用豆包tts数据训练的，非常明显得提升了音质。
# epoch2 是 在豆包tts数据训练后，发现了一个问题， 波浪号没有停顿， 重新构造了数据，继续训练epoch2. 训练到了8749stap.
# epoch3 ，是真正的第三轮， 高质量数据的第二轮，在epoch2结束的时候，发现了模型only X 的结果有问题， 正好启动epoch3并修复bug
# epoch4 ，上一轮报了share memory out错误。
# epoch5 ， 修复了share memory out错误， 并继续训练， 训练到13749stap.
# --------------------
# tag think的训练数据
# epoch0 是 使用kokoro数据训练的，在内容层面先用kokoro数据训练出效果，先不管音质。
# epoch1 是 用豆包tts数据训练的，非常明显得提升了音质。
# epoch2 是 在豆包tts数据训练后，发现了一个问题， 波浪号没有停顿， 重新构造了数据，继续训练epoch2. 训练到了8749stap. ,持续训练到现在，作为第三轮。期间有个返工，不小心用成了language think的process.py了。


# dir=$exp_path/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_12_13_with_speech_gxl_with_asr-chat
dir=/home/A02_tmpdata1/ckpt/context_asr_full/epoch0_all_task_test2_only_contextdata 
#checkpoint=/home/A02_tmpdata3/ckpt/osum_chat/epoch0_all_data/step_10624.pt
#checkpoint=/home/A02_tmpdata3/ckpt/osum_chat/epoch0_all_data/step_14374.pt
#checkpoint=/home/A02_tmpdata3/ckpt/osum_chat/epoch1_all_data/step_2816.pt
#checkpoint=/home/A02_tmpdata3/ckpt/osum_chat/epoch1_all_data/step_3442.pt
#checkpoint=/home/A02_tmpdata3/ckpt/osum_chat/epoch1_all_data/step_5633.pt
#checkpoint=/home/A02_tmpdata3/ckpt/osum_chat/epoch2_all_data/step_4999.pt
# checkpoint=/home/A02_tmpdata3/ckpt/osum_chat_new_start_0810/epoch0_s2t_t2s_t2t/step_1249.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch0_s2t_t2s_t2t_s2s_language_think/step_6249.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch0_s2t_t2s_t2t_s2s_language_think/step_22499.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch1_s2t_t2s_t2t_s2s_hq_language_think/step_24999.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch2_s2t_t2s_t2t_s2s_hq_language_think/step_8749.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch3_s2t_t2s_t2t_s2s_hq_language_think/step_27499.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch4_s2t_t2s_t2t_s2s_hq_language_think/step_23749.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch4_s2t_t2s_t2t_s2s_hq_language_think_new/step_49999.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch5_s2t_t2s_t2t_s2s_hq_language_think/step_13749.pt
#checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch5_s2t_t2s_t2t_s2s_hq_language_think_sft/step_599.pt
#checkpoint=/home/A02_tmpdata1/ckpt/osum_chat_new_start_0810/epoch6_add_emotion_raw_in_no_think_hq_language_think/step_7499.pt
checkpoint=/home/A02_tmpdata1/ckpt/osum_chat_new_start_0810/epoch6_add_emotion_raw_in_no_think_hq_language_think/step_22499.pt
#checkpoint=/home/A02_tmpdata1/ckpt/context_asr_full/epoch0_all_task_new/step_49999.pt

mkdir -p $dir
data=$dir/data
mkdir -p $data


data_type=shard_full_data
train_data_s2t=$data/tmp/tmp_master_s2t.list
train_data_t2s=$data/tmp/tmp_master_t2s.list
train_data_s2s=$data/tmp/tmp_master_s2s.list
train_data_t2t=$data/tmp/tmp_master_t2t.list
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_s2t $train_data_s2t # 只能在master执行，因为随机数是time的，如果每个节点都执行，会导致不同节点的随机数不一致
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_t2s $train_data_t2s
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_s2s $train_data_s2s
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_t2t $train_data_t2t
# train_data=/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/data/shards_train/shards_list.txt
# train_data=conf/asr_data4huawei.list
cv_data=$data/asr_cv.list
head -n 1 $train_data_s2t > $cv_data
wc -l  "$train_data_s2t"
wc -l "$train_data_t2s"
wc -l "$train_data_s2s"
wc -l "$train_data_t2t"

# exit 0


























train_engine=deepspeed # torch_ddp


tensorboard_dir=$dir/tensorboard
num_workers=1
prefetch=50
average_checkpoint=false
decode_checkpoint=$dir/final.pt
average_num=5
average_mode=step
max_step=88888888
decode_modes="attention"
decoding_chunk_size=-1
ctc_weight=0.5
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
decode_batch=10
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model+optimizer"


. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk  -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="nccl" #"nccl"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "$0: PYTORCH_CUDA_ALLOC_CONF is $PYTORCH_CUDA_ALLOC_CONF"
  # torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --node_rank=1 \
  #          --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --node_rank=0 \
          --master_addr=$HOST_NODE_ADDR --master_port=$HOST_PORT \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data_s2t $train_data_s2t \
      --train_data_t2s $train_data_t2s \
      --train_data_s2s $train_data_s2s \
      --train_data_t2t $train_data_t2t \
      --cv_data $cv_data \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --timeout 1200 \
      --use_amp \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states} \


      # --load_dir $dir \
      # --ckpt_id 'epoch_1' \ # 直接加载deepspeed目录
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    subdir=$dir/step_22499
    tag=$(basename "$subdir")
    echo "$tag"
    python3 ${dir}/zero_to_fp32.py \
      ${dir} ${dir}/${tag}.pt -t ${tag}
    rm -rf ${dir}/${tag}
    # for subdir in $(find "$dir" -maxdepth 1 -type d | grep -v "^$dir$")
    # do
    #   if [ $(find "$subdir" -mindepth 1 -type d | wc -l) -eq 0 ]; then
    #     # NOTE(xcsong): zero_to_fp32.py is automatically generated by deepspeed
    #     tag=$(basename "$subdir")
    #     echo "$tag"
    #     python3 ${dir}/zero_to_fp32.py \
    #       ${dir} ${dir}/${tag}.pt -t ${tag}
    #     rm -rf ${dir}/${tag}
    #   fi
    # done
  fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --mode ${average_mode} \
      --max_step ${max_step} \
      --val_best
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decode_checkpoint=$dir/step_499.pt
  i=0
  for testset in ${test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    device_id=${device_ids[i % ${#device_ids[@]}]}
    echo "Testing ${testset} on GPU ${device_id}"
    export CUDA_VISIBLE_DEVICES=$device_id
    python wenet/bin/recognize.py --gpu ${device_id} \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type shard_full_data \
      --test_data /home/work_nfs8/xlgeng/data/scp_test/$testset/shards_list.txt \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size ${decode_batch} \
      --blank_penalty ${blank_penalty} \
      --length_penalty ${length_penalty} \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_dir $result_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} &
    ((i++))
    if [[ $device_id -eq $((num_gpu - 1)) ]]; then
      wait
    fi
  }
  done
  wait
  for testset in ${test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        /home/work_nfs8/xlgeng/data/scp_test/$testset/text $result_dir/$mode/text > $result_dir/$mode/wer
    done
  }
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi
