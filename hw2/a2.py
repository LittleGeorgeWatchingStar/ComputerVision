from PIL import Image, ImageDraw
import numpy as np
import math 
from scipy import signal 
import ncc

img1 = Image.open('/Users/zhangquan/Documents/CS425/hw2/1.jpg')
img2 = Image.open('/Users/zhangquan/Documents/CS425/hw2/judybats.jpg')

#1
def MakePyramid(image, minsize):
    pyramid = []
    im = image
    while (im.size[0] >= minsize) and (im.size[1]) >= minsize:
        pyramid.append(im);
        im = im.resize((int(im.size[0] * 0.75), int(im.size[1] * 0.75)), Image.BICUBIC)

    return pyramid
         
#2
def ShowPyramid(pyramid):
    width = 0
    height = 0

    for img in pyramid:
        width = width + img.size[0]
        height = max(height, img.size[1])

    image = Image.new("RGB", (width, height), 0xFFFFFF)

    offsetX = 0
    offsetY = 0

    for img in pyramid:
        image.paste(img, (offsetX, offsetY))
        offsetX = offsetX + img.size[0]
        offsetY = 0

    image.show()
    
#3
def FindTemplate(pyramid, template, threshold):
    goalWidth = 15
    # resize template
    ratio = goalWidth / template.size[0]
    goalHeight = int(template.size[1] * ratio)
    print(goalHeight)
    print(ratio)
    template = template.resize((goalWidth, goalHeight), Image.BICUBIC)

    pointLists = []
    for image in pyramid:
        nccResult = ncc.normxcorr2D(image, template)
        aboveThreshold = np.where(nccResult > threshold)
        # print(aboveThreshold[0])
        # print(aboveThreshold[1])
        pointLists.append(zip(aboveThreshold[1], aboveThreshold[0]))

    convert = pyramid[0].convert('RGB')

    for i in range(len(pointLists)):
        pointList = pointLists[i]
        scaleFactor = 0.75 ** i

        for pt in pointList:

            ptx = pt[0] / scaleFactor
            pty = pt[1] / scaleFactor

            adjustx = template.size[0] // (2 * scaleFactor)
            adjusty = template.size[1] // (2 * scaleFactor)

            x1 = ptx - adjustx
            y1 = pty - adjusty
            x2 = ptx + adjustx
            y2 = pty + adjusty
            draw = ImageDraw.Draw(convert)
            draw.rectangle([x1,y1,x2,y2], outline="red")
            del draw

    return convert
  
def main():
    imgArray = ['/Users/zhangquan/Documents/CS425/hw2/judybats.jpg', '/Users/zhangquan/Documents/CS425/hw2/students.jpg', '/Users/zhangquan/Documents/CS425/hw2/tree.jpg'
    , '/Users/zhangquan/Documents/CS425/hw2/family.jpg', '/Users/zhangquan/Documents/CS425/hw2/fans.jpg', '/Users/zhangquan/Documents/CS425/hw2/sports.jpg']

    for imgLoc in imgArray:
        img = Image.open(imgLoc)
        img = img.convert('L')

        pyramid = MakePyramid(img, 20)
        # ShowPyramid(pyramid)

        templateLoc = '/Users/zhangquan/Documents/CS425/hw2/template.jpg'
        template = Image.open(templateLoc)
        template = template.convert('L')

        threshold = 0.5715
        found = FindTemplate(pyramid, template, threshold)
        found.save(imgLoc)
        found.show()    
        
          
main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
