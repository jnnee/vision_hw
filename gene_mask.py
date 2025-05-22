import torch
import torch.nn.functional as F
from src.utils_mask import get_mask_location
from torchvision import transforms
from preprocess.dwpose import DWposeDetector
from PIL import Image
import numpy as  np
import os
from preprocess.humanparsing.run_parsing import Parsing
def generate_mask( vton_img, category, offset_top, offset_bottom, offset_left, offset_right,output_dir):
    #with torch.inference_mode():
    with torch.no_grad():
        mask_name=vton_img.split('/')[-1].split('.')[-2]
        vton_img = Image.open(vton_img)
        model_root="/mnt/sda2/lry/fit_pose_ckpt"
        dwprocessor = DWposeDetector(model_root=model_root, device="cpu")
        parsing_model = Parsing(model_root=model_root, device='cpu')
        #vton_img_det = resize_image(vton_img)
        pose_image, keypoints, _, candidate =dwprocessor(np.array(vton_img)[:, :, ::-1])
        candidate[candidate < 0] = 0
        candidate = candidate[0]

        candidate[:, 0] *= vton_img.width
        candidate[:, 1] *= vton_img.height

        pose_image = pose_image[:, :, ::-1]  # rgb
        pose_image = Image.fromarray(pose_image)
        model_parse, _ = parsing_model(vton_img)

        mask, mask_gray = get_mask_location(category, model_parse, \
                                            candidate, model_parse.width, model_parse.height, \
                                            offset_top, offset_bottom, offset_left, offset_right)
        mask = mask.resize(vton_img.size)
        mask_gray = mask_gray.resize(vton_img.size)
        mask = mask.convert("L")
        mask_gray = mask_gray.convert("L")
        mask_gray.convert('RGB').save(os.path.join(output_dir, f"{mask_name}.png"))
        masked_vton_img = Image.composite(mask_gray, vton_img, mask)

        im = {}
        im['background'] = np.array(vton_img.convert("RGBA"))
        im['layers'] = [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:, :, np.newaxis]), axis=2)]
        im['composite'] = np.array(masked_vton_img.convert("RGBA"))

        return im, pose_image
if __name__ == '__main__':
    data_dir="/mnt/sda2/lry/zalando-hd-resized"
    # 使用原始 data_root_path，不修改 self.args
    data_root = os.path.join(data_dir, "train")  # 局部变量
    # pair_txt = os.path.join(data_root, "train_pairs.txt")
    assert os.path.exists(
        pair_txt := os.path.join(data_dir, 'train_pairs.txt')), f"File {pair_txt} does not exist."
    with open(pair_txt, 'r') as f:
        lines = f.readlines()
    # self.args.data_root_path = os.path.join(self.args.data_root_path, "train")
    # output_dir = os.path.join(self.args.output_dir, "vitonhd", 'paired' if not self.args.eval_pair else 'paired')
    data = []
    for line in lines:
        person_img, _ = line.strip().split(" ")
        # im_name, _ = line.strip().split()
        #cloth_img = person_img

        person_img=os.path.join(data_root,"image", person_img)
        print(person_img)
        output_dir = os.path.join(data_root, "gene_mask")
        generate_mask(person_img, "Upper-body", 0.05, 0, 0.05, 0.05, output_dir)
    #return data