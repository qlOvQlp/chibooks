import os
import json 


from torchvision.datasets import ImageFolder
from torchvision import transforms as trans
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from collections import defaultdict

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



class wilds_iwildcam(ImageFolder):
    def __init__(self, 
        root: str, 
        subset: str,
        img_size: [int,int]=[320,240],
        transform: Optional[Callable] = None, 
        use_mega_label_fix: Optional[bool] = False,
        use_mega_bbox: Optional[bool] = False,
        use_spec_filter:Optional[bool] = False
    ):
        super().__init__(os.path.join(root,subset), transform)
        self.img_size = img_size
        self.use_mega_label_fix = use_mega_label_fix
        self.use_mega_bbox = use_mega_bbox
        ## level 1:raw 2.simple 3:mid 4:detalied
        # self.cls_idx_fix, self.mega_label , self.cls_sample_amout= self.__aux_init(root,subset,threshold)

        with open(os.path.join(root,"meta/cls_to_idx.json"),"r") as f:
            cls_to_idx = json.load(f)
        self.cls_idx_fix = cls_to_idx


        ## select reasonable ani spec images
        ## expect useful ani class name list input
        if use_spec_filter:
            with open(os.path.join(root,"meta/filter_ani.json"),"r") as f:
                ani_list = json.load(f)
            self.ani_list = ani_list
            new_sample = []
            new_targets = []
            for sample in self.samples:
                if self.classes[sample[1]] in ani_list:
                    ## needed ani 
                    new_sample.append(sample)
                    new_targets.append(sample[1])

            self.samples = new_sample
            self.targets = new_targets
            
        if use_mega_label_fix:
            mega_label_path = os.path.join(root,f"meta/mega_label/mega_label_p20_from_r20.json")
            with open(mega_label_path,"r") as f:
                mega_label = json.load(f)
            self.mega_label = mega_label
        if use_mega_bbox:
            mega_bbox_path = os.path.join(root,f"meta/mega_label/mega_bbox_p20_from_r20.json")
            with open(mega_bbox_path,"r") as f:
                mega_bbox = json.load(f)
            self.mega_bbox = mega_bbox   
        

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        _,pos,seq,f_name = path.rsplit("/",3)
        if self.use_mega_label_fix:
            mega_y = self.mega_label[f_name]
        else:
            mega_y = -1


                
        # change class name to a unified index "target: int"
        cls_name = self.classes[target]
        target = self.cls_idx_fix[cls_name]

        if self.use_mega_label_fix:
            # only solve target is animal but mega found none
            if mega_y is False: 
                cls_name = "empty"
                target = self.cls_idx_fix["empty"]


        if self.use_mega_bbox:
            bbox = self.mega_bbox[f_name]
            sample = crop_image(img=sample,bbox=bbox,output_size=self.img_size)

        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, target, {"mega_fixed":mega_y, "f_name":f_name, "cls_amount":self.cls_sample_amount[target]}
        return sample, target, {"f_name":f_name,"cls_name":cls_name}


class wilds_iwildcam_dm():
    def __init__(self, 
        root: str, 
        use_mega_bbox: bool = False,
        use_mega_label_fix: bool = False,
        use_spec_filter: bool = False,
        use_val_set: bool = False,

        img_size = [224,336]
    ):
        self.root = root
        self.use_mega_bbox = use_mega_bbox
        self.use_mega_label_fix = use_mega_label_fix
        self.img_size = img_size
        self.use_spec_filter = use_spec_filter
        self.use_val_set = use_val_set

        self.train_trans = trans.Compose([
            trans.RandomResizedCrop(
                size = self.img_size,
                scale = (0.64,1)),
            trans.RandomHorizontalFlip(),
            trans.RandomGrayscale(),
            trans.ColorJitter(
                brightness=0.3,
                contrast=0.3),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
        ])
        self.val_trans = trans.Compose([
            # trans.Resize(256,trans.InterpolationMode.BICUBIC),
            # trans.CenterCrop(self.img_size),
            trans.Resize(self.img_size,trans.InterpolationMode.BICUBIC),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])
        self.test_trans = trans.Compose([
            # trans.Resize(256,trans.InterpolationMode.BICUBIC),
            # trans.CenterCrop(self.img_size),
            trans.Resize(self.img_size,trans.InterpolationMode.BICUBIC),
            trans.ToTensor(),
            trans.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
        ])
    
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_set = wilds_iwildcam(
                root=self.root,
                subset="train",
                transform=self.train_trans,
                use_mega_label_fix=self.use_mega_label_fix,
                use_mega_bbox=self.use_mega_bbox,
                use_spec_filter=self.use_spec_filter
            )
            if self.use_val_set:
                self.val_set = wilds_iwildcam(
                    root=self.root,
                    subset="val",
                    transform=self.val_trans,
                    use_mega_label_fix=self.use_mega_label_fix,
                    use_mega_bbox=self.use_mega_bbox,
                    use_spec_filter=self.use_spec_filter
                )

        else:
            self.test_set = wilds_iwildcam(
                root=self.root,
                subset="test",
                transform=self.test_trans,
                use_mega_label_fix=self.use_mega_label_fix,
                use_mega_bbox=self.use_mega_bbox,
                use_spec_filter=self.use_spec_filter
            )

def get_wilds_iwildcam_from_cfg(cfg):
    dm = wilds_iwildcam_dm(
        root=cfg.dataset.root,
        use_mega_bbox=cfg.dataset.sp.use_mega_bbox,
        use_mega_label_fix=cfg.dataset.sp.use_mega_label_fix,
        img_size=cfg.dataset.img_size,
        use_spec_filter=cfg.dataset.sp.use_spec_filter,
        use_val_set=cfg.fit.val_freq > 0
    )
    dm.setup(cfg.task)
    return dm


if __name__ == "__main__":

    from torchvision import transforms as trans
    from tqdm import tqdm

    print("dataset test ...")
    dataset = wilds_iwildcam(
        root = "/data/wilds_rebuild",
        subset = "train",
        transform=trans.ToTensor(),
        use_spec_filter=True,
        use_mega_label_fix=True,
    )
    empty_cnt = 0
    for data in tqdm(dataset):
        x,y,meta = data
        # cnt empty
        if y == 0:
            empty_cnt+=1
    
    print(f"empty:{empty_cnt}")