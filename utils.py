import numpy as np
import cv2
  
def get_image_info(annotationDict:dict,image_id:int):
    return [[ann["category_id"],ann["bbox"]] for ann in annotationDict["annotations"] if ann["image_id"] == image_id]

def cv2_draw(image_data:list,bbox:list,text:str,color:tuple=(0,0,255),thickness:int=2,display:bool=True):
    bbox = (list(map(int,bbox)))
    cv2.rectangle(img=image_data,pt1=(bbox[:2]),pt2=(np.array(bbox[2:]).__add__(bbox[:2])),color=color,thickness=thickness)
    cv2.putText(img=image_data,text=text,org=np.array(bbox[:2]).__add__([0,-10]),color=color,fontFace=cv2.FONT_ITALIC,fontScale=2,thickness=2)
    if display:
        cv2.imshow("Frame",image_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
       return image_data

def resize_annotations(image_dim:tuple,bbox:list,resize:tuple):
    """resize annotations
    Args:
        image_dim (tuple): image shape
        bbox (list): bbox coordinates [x,y,w,h]
        resize (tuple): resize dimension (x,y)

    Returns:
        list: [x,y,w,h]
    """
    width_ratio,height_ratio = [resize[idx]/image_dim[idx] for idx in range(len(resize))]
    return [bbox[0]*height_ratio,bbox[1]*width_ratio,bbox[2]*height_ratio,bbox[3]*width_ratio]
