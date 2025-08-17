#input_exp_dir=/home/A02_tmpdata3/code/osum_xlgeng_3B/examples/wenetspeech/whisper/exp/epoch1_all_data
num_pt=1
#python do_avg_for_exp_dir.py $input_exp_dir $num_pt



# 目标根目录
ROOT_DIR="/home/A02_tmpdata3/code/OSUM-EChat_code/osum_chat_new_start_0810"

# 检查根目录是否存在
if [ ! -d "$ROOT_DIR" ]; then
    echo "错误：目标目录 $ROOT_DIR 不存在"
    exit 1
fi

# 定义数组变量存储所有一级子目录的绝对路径
subdir_paths=()

# 遍历一级子目录并获取绝对路径
for subdir in "$ROOT_DIR"/*/; do
    # 处理路径，去除末尾斜杠
    real_path=$(realpath "$subdir" | sed 's/\/$//')
    # 添加到数组
    subdir_paths+=("$real_path")
done

# 后续使用示例：遍历数组中的路径
echo "获取到的一级子目录绝对路径如下："
for path in "${subdir_paths[@]}"; do
    echo "$path"
#    python do_avg_for_exp_dir.py $path $num_pt
    # 这里可以添加后续处理逻辑，例如：
    # cd "$path" && echo "正在处理目录：$path"
done