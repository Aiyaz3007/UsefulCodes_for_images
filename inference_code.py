import json
import cv2
from os.path import join,isdir,isfile
from random import randint
from os import makedirs,system
from tqdm import tqdm as progress_bar
from termcolor import colored
import numpy as np
from sys import platform

class VisualizeImage:
    """draw boxes from annotation file
    """
    def __init__(self,imagePath,annotationFile,outputPath):
        self.annotationFile = annotationFile
        self.imagePath = imagePath
        self.outputPath = outputPath
        self.missingImageCount = 0


    def readAnnotations(self):
        with open(self.annotationFile,"r") as f:
            annotations = json.load(f)
        
        Color = []
        category_names = []
        for x in annotations["categories"]:
            Color.append((randint(0,255),randint(0,255),randint(0,255)))
            category_names.append(x["name"])
        
        if not isdir(self.outputPath):
                makedirs(self.outputPath)
        bar = progress_bar(range(1,len(annotations["images"])),desc=colored(text="Process",color="green"),colour="green")
        annotation_ite = annotations["images"]
        for idx,image in enumerate(annotation_ite):
            count = 0

            imageName = image["file_name"]
            image_id = image["id"]
            fullPath = join(self.imagePath,imageName)
            if isfile(fullPath): #is file is present in mention dir
                image_data = cv2.imread(fullPath)
                for annotation in annotations["annotations"]:
                    annotation_image_id = annotation["image_id"]

                    if image_id == annotation_image_id:
                        count += 1
                        category_id = annotation["category_id"]
                        x,y,w,h = annotation["bbox"]
                        x,y,w,h = int(x),int(y),int(w),int(h)
                        # if len(annotation['segmentation']) == 0:
                        #     x,y,w,h = int(x)+10,int(y)+10,int(w) + 10,int(h) + 10
                        if annotation["bbox"] != []:
                            cv2.rectangle(
                                    img=image_data,
                                    pt1=(int(x),int(y)),
                                    pt2=(int(x+w),int(y+h)),
                                    color=Color[category_id-1],
                                    thickness=2)
                        if annotation["segmentation"] != []:
                            pointsPrevious = []
                            for point in np.array(annotation["segmentation"][0]).reshape(-1,2):
                                if len(pointsPrevious) > 1:
                                    pt1,pt2 = pointsPrevious
                                    cv2.line(img=image_data,pt1=(int(pt1[0]),int(pt1[1])),pt2=(int(pt2[0]),int(pt2[1])),color=Color[category_id-1],thickness=2)
                                    pointsPrevious.pop(0)
                                pointsPrevious.append(point)
                            
                                                                                    

                        cv2.putText(
                                img=image_data,
                                text=category_names[category_id-1],
                                org=(x,y-10),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=2,
                                color=Color[category_id-1],
                                thickness=2
                        )

                outputPath = join(self.outputPath,imageName)
                cv2.imwrite(
                        filename=outputPath,
                        img=image_data)
            else:
                self.missingImageCount += 1
                
            if platform!="win32":
                system("clear")
            else:
                system("cls")

            bar.update(1)
            if idx+1 >= len(annotation_ite):bar.close()
            print(colored("Image No:","yellow"),idx+1,"\n"+colored("Missing Image Count:","yellow"),self.missingImageCount, "\n"+colored("No of Annotation:","yellow"),count)


import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--annotationFilePath", help = "Annotation Path",required=True)
parser.add_argument("-d", "--datasetPath", help = "Dataset Path [images]",required=True)
parser.add_argument("-o", "--outputPath", help = "Output Path",required=True)
parser.add_argument("-c", "--count", help = "number of images need to process",required=False)


args = parser.parse_args()

obj=VisualizeImage(annotationFile=args.annotationFilePath,imagePath=args.datasetPath,outputPath=args.outputPath)
obj.readAnnotations()

