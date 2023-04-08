import torch
import json
from pycocotools.coco import COCO 
import os
import cv2


class CustomDataset():
  def __init__(self,annotationFile:str,root_dir:str):
    self.coco = COCO(annotationFile)
    self.root_images = root_dir
    self.x_train = self.coco.getImgIds()
  def readFile(self,annotationfile:str):
    with open(annotationfile,"r") as f:
      return json.load(f)

  def __len__(self):
    return len(self.x_train)

  def get_resized_annotations(self,image_dim:tuple,bbox:list,resize:tuple):
    """resize annotations
    Args:
        image_dim (tuple): image shape
        bbox (list): bbox coordinates [x,y,w,h]
        resize (tuple): resize dimension (x,y)

    Returns:
        list: [x,y,w,h]
    """
    width_ratio,height_ratio = [resize[idx]/image_dim[idx] for idx in range(len(resize))]
    return (bbox[0]*height_ratio,bbox[1]*width_ratio,bbox[2]*height_ratio,bbox[3]*width_ratio)

  def __getitem__(self,index):
    img_info = self.coco.loadImgs(self.x_train[index])[0]
    img_main = cv2.imread(os.path.join(self.root_images, img_info['file_name']))
    img = cv2.resize(img_main,(600,600))
    img = img/255  
    annIds = self.coco.getAnnIds(imgIds=img_info['id']) 
    anns = self.coco.loadAnns(annIds)
    num_objs = len(anns)
    boxes = torch.zeros([num_objs,4], dtype=torch.float32)
    labels = torch.zeros(num_objs, dtype=torch.int64)

    for i in range(num_objs):
        [x,y,w,h]=anns[i]["bbox"]
        [x,y,w,h] = self.get_resized_annotations(img_main.shape,[x,y,w,h],(600,600))
        boxes[i] = torch.tensor(list(map(int,(x,y,x+w,y+h))))
        labels[i] = torch.tensor([anns[i]["category_id"]])

    img = torch.as_tensor(img, dtype=torch.float32)
    img = img.permute(2,0,1)
    data = {}
    data["boxes"] =  boxes
    data["labels"] = labels
    return img, data
  def collate_fn(self,batch):
        return tuple(zip(*batch))

# CustomDataset(annotationFile="/content/aquarium_dataset/train_annotations.json",root_dir="/content/aquarium_dataset/train_resized")