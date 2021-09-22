# import the packages we need for this assignment
from PIL import Image
import numpy as np
import math 
from scipy import signal 

imgDog1 = Image.open('/Users/zhangquan/Documents/dog.jpg')
imgDog = Image.open('/Users/zhangquan/Documents/0b_dog.bmp')
imgCat = Image.open('/Users/zhangquan/Documents/0a_cat.bmp')
img1a = Image.open('/Users/zhangquan/Documents/1a_bicycle.bmp')
img1b = Image.open('/Users/zhangquan/Documents/1b_motorcycle.bmp')
img2a = Image.open('/Users/zhangquan/Documents/2a_einstein.bmp')
img2b = Image.open('/Users/zhangquan/Documents/2b_marilyn.bmp')
img3a = Image.open('/Users/zhangquan/Documents/3a_fish.bmp')
img3b = Image.open('/Users/zhangquan/Documents/3b_submarine.bmp')
img4a = Image.open('/Users/zhangquan/Documents/4a_bird.bmp')
img4b = Image.open('/Users/zhangquan/Documents/4b_plane.bmp')

imgBW = imgDog1.convert('L')
# part 1, generate odd number filters
#1
def boxfilter(x):    
    assert (x%2==1),"must be odd"
    #result = np.zeros(x,x)
    result = np.ones((x,x),dtype="float32")
    n = 1/(x*x)
    result = np.full((x,x),n)
    return result
         
#2
def gauss1d(sigma):
    assert sigma > 0, 'gauss1d: sigma cannot be less than or equal to zero'

    sigma = float(sigma)
    # length should be 6 times sigma rounded up to the next odd integer
    length = math.ceil(sigma * 6)
    # increment if length is even
    if length % 2 == 0:
        length = length + 1;

    # construct initial 1D x values
    maxx = math.floor(length/2) 
    arange = np.arange(-maxx, maxx + 1)
    # apply the formula to each value in the array
    twoSigmaSqr = 2 * sigma * sigma
    gaussFilter = np.exp(-arange ** 2 / twoSigmaSqr)
    # normalize (ensure sum of matrix values is 1)
    gaussFilter /= np.sum(gaussFilter)
    
    return gaussFilter
    
def gauss2d(sigma):
    gauss = gauss1d(sigma)[np.newaxis]
    gaussTranspose = gauss1d(sigma)[np.newaxis].transpose()

    convolved = signal.convolve2d(gauss, gaussTranspose)
    return convolved
    
# imgBW = imgDog1.convert('L') is the greyscale of dog.jpg
# so run with gaussConvolve2d(imgBW, 3)  
def gaussConvolve2d(image_array, sigma):
    tempGauss2d = gauss2d(sigma)
    
    filtered_array = signal.convolve2d(image_array, tempGauss2d, 'same')
    filtered_image = Image.fromarray(filtered_array)
    # filtered image 
    filtered_image.show()
    # original image
    imgDog1.show()
    return filtered_array
    
#part2
#1
def gaussConvolveColor(img, sigma):
    imgColor = img.convert('RGB')
    img_array = np.asarray(imgColor)
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]
    
    redBlur = gaussConvolve2d(red, sigma)
    greenBlur = gaussConvolve2d(green, sigma)
    blueBlur = gaussConvolve2d(blue, sigma)
    
    combinedBlur = np.stack((redBlur, greenBlur, blueBlur), axis=-1)
    
    filtered_color = Image.fromarray(combinedBlur.astype('uint8'))
    filtered_color.show()
    
#2
def gaussHighFre(img, sigma):    
    imgColor = img.convert('RGB')
    img_array = np.asarray(imgColor, dtype=np.double)
    
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]
    
    redBlur = gaussConvolve2d(red, sigma)
    greenBlur = gaussConvolve2d(green, sigma)
    blueBlur = gaussConvolve2d(blue, sigma)
    
    redHigh = np.subtract(red+128, redBlur)
    greenHigh = np.subtract(green+128, greenBlur)
    blueHigh = np.subtract(blue+128, blueBlur)
    
    combinedHigh = np.stack((redHigh, greenHigh, blueHigh), axis=-1)
    filtered_color = Image.fromarray(combinedHigh.astype('uint8'))
    filtered_color.show()
    
#3
def gaussHybrid(img, img2, sigma):    
    imgColor = img.convert('RGB')
    img_array = np.asarray(imgColor, dtype=np.double)
    red = img_array[:, :, 0]
    green = img_array[:, :, 1]
    blue = img_array[:, :, 2]
    
    redBlur = gaussConvolve2d(red, sigma)
    greenBlur = gaussConvolve2d(green, sigma)
    blueBlur = gaussConvolve2d(blue, sigma)
    redHigh = np.subtract(red, redBlur)
    greenHigh = np.subtract(green, greenBlur)
    blueHigh = np.subtract(blue, blueBlur)
    
    imgColor2 = img2.convert('RGB')
    img_array2 = np.asarray(imgColor2, dtype=np.double)
    red2 = img_array2[:, :, 0]
    green2 = img_array2[:, :, 1]
    blue2 = img_array2[:, :, 2]
    
    redBlur2 = gaussConvolve2d(red2, sigma)
    greenBlur2 = gaussConvolve2d(green2, sigma)
    blueBlur2 = gaussConvolve2d(blue2, sigma)
    redHigh2 = np.subtract(red2, redBlur2)
    greenHigh2 = np.subtract(green2, greenBlur2)
    blueHigh2 = np.subtract(blue2, blueBlur2)
    
    redHybrid = np.add(redHigh2, redBlur)
    greenHybrid = np.add(greenHigh2, greenBlur)
    blueHybrid = np.add(blueHigh2, blueBlur)
    
    redHybrid2 = np.add(redHigh, redBlur2)
    greenHybrid2 = np.add(greenHigh, greenBlur2)
    blueHybrid2 = np.add(blueHigh, blueBlur2) 
    
    combinedHybrid2 = np.stack((redHybrid2, greenHybrid2, blueHybrid2), axis=-1)
    filtered_color2 = Image.fromarray(combinedHybrid2.astype('uint8'))
    filtered_color2.show() 
    filtered_color2.save('part2_4_2.png','PNG')
    
    combinedHybrid = np.stack((redHybrid, greenHybrid, blueHybrid), axis=-1)
    filtered_color = Image.fromarray(combinedHybrid.astype('uint8'))
    filtered_color.show()    
    filtered_color.save('part2_4_1.png','PNG')
    
  
    