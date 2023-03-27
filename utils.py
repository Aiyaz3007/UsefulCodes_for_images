import numpy as np
import cv2


def get_xml_coorinates(filename:str):
  def readFile(filename:str):
    text = ""
    with open(filename,"r") as f:
      text = f.read()
    return text

  data = readFile(filename)
  filename = data[data.find("<filename>"):data.find("</filename>")].replace("<filename>","")
  category = data[data.find("<name>"):data.find("</name>")].replace("<name>","")

  xmin = int(data[data.find("<xmin>"):data.find("</xmin>")].replace("<xmin>",""))
  ymin = int(data[data.find("<ymin>"):data.find("</ymin>")].replace("<ymin>",""))
  xmax = int(data[data.find("<xmax>"):data.find("</xmax>")].replace("<xmax>",""))
  ymax = int(data[data.find("<ymax>"):data.find("</ymax>")].replace("<ymax>",""))
  return filename,category,(xmin,ymin,xmax,ymax)


def resize_image_annotations(filename:str,resize:tuple):
  imageToPredict = cv2.imread(filename)
  x_ = imageToPredict.shape[0]
  y_ = imageToPredict.shape[1]

  x_scale = resize[0]/x_
  y_scale = resize[1]/y_
  img = cv2.resize(imageToPredict,(416,416))
  img = np.array(img)


  x = int(np.round(128*x_scale))
  y = int(np.round(25*y_scale))
  xmax= int(np.round  (447*(x_scale)))
  ymax= int(np.round(375*y_scale))

  return (x,y,xmax,ymax)


def draw_bbox(filename:str,coordinates:tuple,category:str):
  image_data = cv2.imread(filename)
  cv2.rectangle(image_data,(coordinates[0],coordinates[1]),(coordinates[2],coordinates[3]),(0,255,0),2)
  cv2.putText(image_data,category,(coordinates[0],coordinates[1]-10),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255))
  return image_data