import json
import pandas as pd
import os
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as trans
from torch.utils.data import ConcatDataset
import torch
from torchvision.utils import save_image

def crop_image(img, bbox, output_size):
    # 打开图像文件
    width, height = img.size
    target_w, target_h = output_size

    # 计算实际的边界框坐标
    x = bbox[0] * width
    y = bbox[1] * height
    w = bbox[2] * width
    h = bbox[3] * height

    # 计算新的边界框
    size_w = max(w, target_w)
    size_h = max(h, target_h)

    # 保持长宽比例一致
    if size_w/size_h > target_w/target_h:
        size_h = size_w * target_h / target_w
    else:
        size_w = size_h * target_w / target_h

    x_new = x + w/2 - size_w/2
    y_new = y + h/2 - size_h/2

    # 平移边界框
    if x_new < 0:
        x_new = 0
    if y_new < 0:
        y_new = 0
    if x_new + size_w > width:
        x_new = width - size_w
    if y_new + size_h > height:
        y_new = height - size_h

    # 裁剪图像
    cropped_img = img.crop((x_new, y_new, x_new+size_w, y_new+size_h))

    # 调整图像尺寸
    resized_img = cropped_img.resize((target_w, target_h))

    return resized_img

class TerraIncDataset(ImageFolder):
    def __init__(self, root, 
                 info_cols=['desc'], 
                 crop=False, 
                 fix=False,
                 crop_size=(224, 224)):
        self.root = root
        self.info_cols = info_cols
        self.crop = crop
        self.fix = fix
        self.crop_size = crop_size
        self.trans=trans.Compose([
            # trans.Resize(crop_size),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])
        self.empty_trans = trans.Compose([
            trans.RandomResizedCrop(crop_size),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])
        super().__init__(os.path.join(root,"imgs"))

        # 加载物种信息
        xlsx_file = os.path.join(root,"meta/species_info.xlsx")
        self.cls_idx_fix = {
            "bird":0,"bobcat":1,"cat":2,"coyote":3,"dog":4,
            "empty":5,"opossum":6,"rabbit":7,"raccoon":8,"squirrel":9}
        self.species_info = pd.read_excel(xlsx_file)

        # 加载边界框信息
        json_file = os.path.join(root,"meta/terra_inc_mega_01_converted.json")
        with open(json_file, 'r') as f:
            self.bboxes_info = json.load(f)

    def __getitem__(self, index):
        fpath, _ = self.samples[index]
        label = fpath.rsplit("/",2)[-2]
        label_idx = self.cls_idx_fix[label]
        img = self.loader(fpath)
        
        # 如果crop参数为True，根据边界框裁剪图像
        if self.crop:
            bbox_key = fpath.rsplit("imgs/",1)[-1]
            imgh_crop_info = self.bboxes_info[bbox_key]
            max_conf = imgh_crop_info["conf"]

            # mega default threshold = 0.1, 
            if max_conf >0.09:
                bbox = imgh_crop_info["bbox"]
                img= crop_image(img,bbox,self.crop_size)

                # width, height = img.size

                # bbox[0] = bbox[0] * width
                # bbox[1] = bbox[1] * height

                # bbox[2] = bbox[2] * width
                # bbox[3] = bbox[3] * height

                # # 计算裁剪区域的长和宽
                # crop_width, crop_height = bbox[2], bbox[3]

                # # 如果裁剪区域的长宽都小于设定的长宽，则扩大裁剪区域，使得裁剪区域和设定一致
                # if crop_width < self.crop_size[0] or crop_height < self.crop_size[1]:
                #     crop_width = max(crop_width, self.crop_size[0])
                #     crop_height = max(crop_height, self.crop_size[1])

                # # 如果裁剪区域的长或者宽超过了设定值，则扩大长或者宽，使得裁剪区域的比例和设定的长宽比例一致，然后再resize到设定尺寸
                # else:
                #     ratio = max(crop_width / self.crop_size[0], crop_height / self.crop_size[1])
                #     crop_width = int(crop_width / ratio)
                #     crop_height = int(crop_height / ratio)

                # # 确保裁剪区域不超过原图边界
                # left, upper, right, lower = max(0, bbox[0]), max(0, bbox[1]), min(width, bbox[0] + crop_width), min(height, bbox[1] + crop_height)
                
                # img = img.crop((left, upper, right, lower)).resize(self.crop_size)

        # 获取物种信息
        species_info = self.species_info.loc[self.species_info['y'] == label_idx, self.info_cols].to_dict('records')[0]

        label_idx = self.cls_idx_fix[label]
        if self.fix:
            imgh_crop_info = self.bboxes_info[fpath]
            max_conf = imgh_crop_info["conf"]
            if max_conf <0.1:
                label_idx = self.cls_idx_fix["empty"]


        if max_conf > 0.09:
            img = self.trans(img)
        else:
            img = self.empty_trans(img)
        
        # save_image(img,os.path.join("/data/sls_dump/datasets/terra_inc_crop",bbox_key))
        

        return img, label_idx, species_info
    


class TerraIncDataset_DM():
    def __init__(self, root, info_cols=['desc'], crop=False, fix=False, crop_size=(224, 224)):
        self.root = root
        self.info_cols = info_cols
        self.crop = crop
        self.fix = fix
        self.crop_size = crop_size
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            pass
        else:
            self.test_set = TerraIncDataset(
                root = self.root,
                info_cols=self.info_cols,
                crop=self.crop,
                fix=self.fix,
                crop_size=self.crop_size
            )


def get_terrainc_from_cfg(cfg):
    terrainc = TerraIncDataset_DM(root=cfg.dataset.root,
                               info_cols=cfg.dataset.info,
                               crop=cfg.dataset.mega_crop,
                               crop_size = cfg.dataset.img_size,
                               fix=cfg.dataset.mega_fix)
    terrainc.setup(cfg.task)
    return terrainc