from gxl_ai_utils.utils import utils_file
import os
from pathlib import Path

import torch

from convert_ckpt_dir_to_pt import convert_ckpt_to_pt

# input_exp_dir = "/home/A02_tmpdata3/code/osum_xlgeng_3B/examples/wenetspeech/whisper/exp/epoch1_all_data"
# num_pt = 5  # 用于平均的参数数量
input_exp_dir, num_pt = utils_file.do_get_commandline_param(2, ['input_exp_dir', 'num_pt'])
num_pt = int(num_pt)
print(f"input_exp_dir: {input_exp_dir}, num_pt: {num_pt}")

input_exp_path = Path(input_exp_dir)

# 得到所有step*目录并转换为pt文件
step_dirs = [path for path in input_exp_path.glob("step*") if path.is_dir()]
print("\n找到的step目录：")
utils_file.print_list(step_dirs)
for dir_path in step_dirs:
    print(dir_path)
    pt_file = dir_path.with_suffix(".pt")  # 使用Path方法处理后缀，更规范

    if pt_file.exists():
        print(f"{pt_file} 已存在，跳过")
        utils_file.remove_dir(str(dir_path))
        continue

    try:
        convert_ckpt_to_pt(str(dir_path))  # 若函数要求字符串路径则转换
        print(f"{dir_path} 转换成功")
        utils_file.remove_dir(str(dir_path))  # 删除step目录
    except Exception as e:
        print(f"{dir_path} 转换失败: {str(e)}")
print('dir 转 pt 完成')
pt_files_path = [path for path in input_exp_path.glob("step*.pt") if path.is_file()]

# 收集仅由step*目录生成的pt文件
# pt_files_path = [str(step_dir.with_suffix(".pt")) for step_dir in step_dirs
#                  if step_dir.with_suffix(".pt").exists()]

print("\n找到的step相关pt文件：")
utils_file.print_list(pt_files_path)


def average_model_parameters(pt_files, input_exp_path:Path, num_pt):
    """
    从step_xxx.pt文件中提取数字，按数字排序后取最后num_pt个文件计算参数平均值，为每个参数保留原始数据类型

    Args:
        pt_files: 模型参数文件路径列表（文件名格式为step_xxx.pt）
        output_avg_path: 平均后参数的保存路径
        num_pt: 用于平均的文件数量

    Returns:
        平均后的参数字典
    """
    if not pt_files:
        raise ValueError("未找到任何有效的.pt文件")

    # 提取文件名中的step数字并与路径关联
    step_info = []
    for path in pt_files:
        filename = os.path.basename(path)
        if filename.startswith("step_") and filename.endswith(".pt"):
            try:
                step_num = int(filename[len("step_"):-len(".pt")])
                step_info.append((step_num, path))
            except ValueError:
                raise ValueError(f"文件名格式错误，无法提取step数字: {filename}")
        else:
            raise ValueError(f"文件名不符合格式要求（需为step_xxx.pt）: {filename}")

    # 按step数字升序排序并选择最后num_pt个文件
    step_info.sort(key=lambda x: x[0])
    if len(step_info) < num_pt:
        raise ValueError(f"可用的pt文件数量不足{num_pt}个，当前数量: {len(step_info)}")

    if num_pt == -1:
        num_pt = len(step_info)  # 全部文件均用于平均
    selected_steps = step_info[-num_pt:]
    selected_files = [path for (num, path) in selected_steps]
    max_step = selected_steps[-1][0]
    print(f"将使用以下{num_pt}个最新的step文件进行平均: {[num for (num, path) in selected_steps]}")

    # 加载第一个文件的参数并记录每个参数的原始类型
    first_params = torch.load(selected_files[0], map_location="cpu")
    param_dtypes = {key: first_params[key].dtype for key in first_params}  # 为每个参数记录原始类型
    # print(f"已记录每个参数的原始数据类型，示例: {next(iter(param_dtypes.items()))}")

    # 初始化平均参数（使用原始类型）
    avg_params = {key: first_params[key].clone() for key in first_params}
    num_models = len(selected_files)

    # 累加其余模型的参数
    for i in range(0, num_models):
        print(f"正在累加第{i+1}个模型参数, path: {selected_files[i]}")
        params = torch.load(selected_files[i], map_location="cpu")

        # 检查参数键是否一致
        if params.keys() != avg_params.keys():
            raise ValueError(f"文件 {selected_files[i]} 与 {selected_files[0]} 参数结构不一致")

        # 检查每个参数的类型是否与第一个文件一致
        for key in params:
            if params[key].dtype != param_dtypes[key]:
                raise ValueError(f"参数 {key} 在文件 {selected_files[i]} 中的类型与第一个文件不一致")

        # 累加参数：临时转为float32避免精度损失，再转回原始类型
        for key in avg_params:
            # original_dtype = param_dtypes[key]
            # 临时提升到float32进行累加，防止低精度下的溢出
            avg_params[key] = avg_params[key]+ params[key]

    # 计算平均值：同样临时用float32保证精度，再转回每个参数的原始类型
    for key in avg_params:
        avg_params[key] = avg_params[key] / num_models

    try:
        output_avg_path = input_exp_path / f"avg_model_{num_pt}_{max_step}.pt"
        utils_file.makedir_for_file(output_avg_path)
        torch.save(avg_params, output_avg_path)
        print(f"\n参数平均完成，已保存至: {output_avg_path}")
        print(
            f"参数类型保持原始状态，示例参数 {next(iter(avg_params.keys()))} 类型: {avg_params[next(iter(avg_params.keys()))].dtype}")
    except Exception as e:
        print(f"保存平均参数时出错: {str(e)}")
        raise  # 重新抛出异常，便于上层处理



# 使用Path处理输出路
average_model_parameters(pt_files_path, input_exp_path, num_pt)
