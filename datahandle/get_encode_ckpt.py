from gxl_ai_utils.utils import utils_file
import torch

def do_get_encode_ckpt():
    print("Getting encode ckpt...")
    osum_echat_ckpt_path = "/apdcephfs_qy3/share_976139/users/xuelonggeng/ckpt/osum_echat/models--ASLP-lab--OSUM-EChat/snapshots/d658ae8c15675b8f7ce0ffdee879f99549a1e70b/language_think_final.pt"
    output_encoder_ckpt_path = "/apdcephfs_qy3/share_976139/users/xuelonggeng/ckpt/osum_echat/only_encder_ckpt.pt"
    param_dict = torch.load(osum_echat_ckpt_path, map_location='cpu')
    encoder_dict = {}
    for key in param_dict.keys():
        if key.startswith('encoder.'):
            encoder_dict[key] = param_dict[key]
    torch.save(encoder_dict, output_encoder_ckpt_path)
    print("Encoder ckpt saved to:", output_encoder_ckpt_path)
    keys = list(param_dict.keys())
    utils_file.print_list(keys)

def check():
    output_encoder_ckpt_path = "/apdcephfs_qy3/share_976139/users/xuelonggeng/ckpt/osum_echat/only_encder_ckpt.pt"
    param_dict = torch.load(output_encoder_ckpt_path, map_location='cpu')
    keys = list(param_dict.keys())
    utils_file.print_list(keys)


if __name__ == '__main__':
    check()