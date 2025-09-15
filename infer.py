from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
import torch
from gxl_ai_utils.utils import utils_file


config_path = "../conf/ct_config.yaml"
model_path = "/home/A02_tmpdata3/ckpt/osum_chat_new_start_0810/epoch0_s2t_t2s_t2t/step_1249.pt"

def load_model_and_tokenizer(checkpoint_path, config_path):
    """
    封装了加载模型和分词器的逻辑
    Args:
        checkpoint_path (str): 模型权重文件路径
        config_path (str): 模型配置文件路径
    Returns:
        model: 加载好的模型
        tokenizer: 加载好的分词器
    """
    print(f"正在从以下路径加载模型: {checkpoint_path}")
    args = GxlNode({"checkpoint": checkpoint_path})
    configs = utils_file.load_dict_from_yaml(config_path)
    model, configs = init_model(args, configs)
    model = model.to(device).to(torch.bfloat16)
    model.eval()  # 设置为评估模式
    tokenizer = init_tokenizer(configs)
    print(f"模型 {checkpoint_path} 加载完成并移动到 {device}")
    return model, tokenizer


print("加载模型和分词器...")