import numpy as np
import cv2
from PIL import Image 



# Run gui.py to see the app
# Do not change any name or prameters of any function or return just make shure u rturn grayscale image.

# These images are used to test in this file.
TEST_IMAGES = {
    'eren': 'data/images/Eren.jpg',
    'mikasa': 'data/images/Mikasa.jpg',
    'boy': 'data/images/Animeboy.jpg',
    'girl': 'data/images/Animegirl.jpg',
}

# complete these Kernels depends on what model do you have 
# make sure to search about 5X5 kernel and write down it here correctly
KERNEL = {
    'box': {
        '3x3': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]).astype('float') / 9,
        '5x5': np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]).astype('float') / 25,
    },
    'avg': {
        '3x3': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]).astype('float') / 9,
         '5x5': np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]]).astype('float') / 25,
    },
    'prewitt': {
        'ver': np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]).astype('float'),
        'hor': np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]).astype('float'),
    },
    'sobel': {
        'ver': np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype('float'),
        'hor': np.array([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]).astype('float'),
    },
    'sharpen': np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]).astype('float'),
}

# You can use it in testing in this file 
# dont remove it because it's used in GUI 
# returns grayscale image
def load_grayscale(filename):
    # Read image
    image = cv2.imread(filename)
    
    # Retrun grayscale image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#################
#  Segmentation #
#################

# this function takes im (image) and thres (desired threshold) that u have entered in gui 
# don't care about how, just implement it here and test it here and it will give the correct result in gui
def apply_manual_thres(image, thres):
    # islam mohammed task
    #first method O(1)
    im_out = image.copy()
    im_out[image > thres] = 255
    im_out[image < thres] = 0

    #second method O(n)
    #  im_out = image.copy()
    # r, c = im_out.shape
    # for x in range(0, r):
    #     for y in range(0, c):
    #         if im_out[x, y] <= thres:
    #             im_out[x,y] = 0
    #         else:
    #             im_out[x,y] = 255
    return im_out

# implement otsu's method here it takes the image
# returns the modified image and the optimal threshold 
# don't change any of return values please.
def otsu(image):
    #mean  with islam 
    pixel_number = image.shape[0] * image.shape[1]
    mean_weigth = 1/pixel_number
    #histofram
    his, bins = np.histogram(image, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]: 
        #from equation 
        #weight 
        Wb = np.sum(his[:t]) * mean_weigth  
        Wf = np.sum(his[t:]) * mean_weigth  
        #
        mub = np.mean(his[:t])
        muf = np.mean(his[t:])
        
        value = Wb * Wf * (mub - muf) ** 2
        
        if value > final_value:
            final_thresh = t
            final_value = value

    final_img = image.copy()
    # print(final_thresh)
    final_img[image > final_thresh] = 255
    final_img[image < final_thresh] = 0

    return final_img, final_thresh

########################
#  Contrast Stretching #
########################
# apply histogram equalization and then return the modified image
def hist_eq(im):
    #return your code
    return im

# apply contrast stretching it takes the image
# and it returns the modified image
def contrast_stretch(im):
    def normalizeRed(intensity):
        iI      = intensity
        minI    = 86
        maxI    = 230
        minO    = 0
        maxO    = 255
        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO
    # Method to process the green band of the image
    def normalizeGreen(intensity):
        iI      = intensity
        minI    = 90
        maxI    = 225
        minO    = 0
        maxO    = 255
        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
        return iO
    # Method to process the blue band of the image
    def normalizeBlue(intensity):
        iI      = intensity
        minI    = 100
        maxI    = 210
        minO    = 0
        maxO    = 255
        iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
        return iO
    # Create an image object
    # imageObject    = Image.open("/content/20220309_161114.jpg")
    # Split the red, green and blue bands from the Image
    b,g,r=cv2.split(im)
    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand   = r.point(normalizeRed)
    normalizedGreenBand = g.point(normalizeGreen)
    normalizedBlueBand  = b.point(normalizeBlue)
    # Create a new image from the contrast stretched red, green and blue brands
    im = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))
    # plt.imshow(normalizedImage)

    return im


######################
#   Linear Filters   #
######################
# this function used in linear filter takes image and filter(kernel), and kernel size
# you can use it in box filter, average and any other linear filter
# this function returns the modified image 
# you can get the kernel with specific size from KERNEL dictionary by typing KERNEL[filter][size]
def apply_linear_filter(image, filter, size):
   
    # write ur code here
    return image

######################
# Non-Linear Filters #
######################

def max_filter(img, Ksize):
    im_out = np.zeros(img.shape)

    return im_out

def min_filter(img, Ksize):
    im_out = np.zeros(img.shape)

    return im_out

def mean_filter(img, Ksize):
        im_out = np.zeros(img.shape)
        new_im = np.zeros((img.shape[0] + 2, img.shape[1]+2))
        new_im[1:new_im.shape[0]-1 , 1:new_im.shape[1]-1] = img
        result = np.zeros(new_im.shape)
        for r in np.arange(1,new_im.shape[0]-1):
            for c in np.arange(1, new_im.shape[1] - 1):
                curr_region = new_im[r-1:r+2,c-1:c+2]
                curr_result = curr_region * KERNEL['box'][Ksize]
                score = np.sum(curr_result)
                result[r,c] = score

        final_im = result[1:result.shape[0]-1,1:result.shape[1]-1]
        return final_im


######################
#   Edge Detection   #
######################
# you can find the kernel for edge detection by typing KERNEL[filter][type_]
# this function return modified image
def apply_edge_detection(image, filter, type_):
    # write your code here
    r, c = image.shape
    im_out = image.copy()
    for x in range(3, r-2):
        for y in range(3, c-2):
            local_pixels = image[x-1:x+2, y-1:y+2]
            trans_pixel = local_pixels * KERNEL[filter][type_]
            score = (trans_pixel.sum() + 4)/8  #range 0,1 to be declear
            im_out[x,y] = score 

    return im_out

# it takes the vertical and horizontal images
# it returns the edge detection image with horizontal and vertical
def combine_both_edge_images(im_v, im_h):
    # im_out = np.zeros_like(im_h)
    # write ur code here
    # for x in range(0, im_out.shape[0]):
    #     for y in range(0, im_out.shape[1]):
    #        im_out =  (im_v[x,y]**2 + im_h[x,y]**2)**0.5
    #        im_out[x,y] = im_out

    # r, c = im_out.shape
    # for x in range(3, r-2):
    #     for y in range(3, c-2):
    #         local_pixels = im_out[x-1:x+2, y-1:y+2]
    #         # row, col = local_pixels
    #         # trans_pixel_hor = np.dot(row, col) * im_h
    #         trans_pixel_hor = im_h[x:r-2,y:c-2] * local_pixels
    #         score_hor = (trans_pixel_hor.sum() + 4)/8  # to declear value

    #         trans_pixel_ver =im_v[x:r-2,y:c-2] * local_pixels 
    #         score_ver = (trans_pixel_ver.sum() + 4)/8

    #         edge_score = (score_ver**2 + score_hor**2)**0.5
    #         im_out[x,y] = edge_score * 3  # to focus value to return high quality
    # im_ve = (im_v.sum() + 4)/8
    # im_hi = (im_h.sum() + 4)/8
    edge_score = (im_v**2 + im_h**2)**0.5  #from equation
    im_out = edge_score * 7   ## to declear output and return high quality output
    # im_out = im_out/im_out.max()
    # im_out = im_v + im_h
    return im_out

########################
# CANNY EDGE DETECTION #
########################
# its your turn to implement canny without any help. 
def canny(image):
    # write ur code here
    return image

##################
#   SHARPENING   #
##################
# a is the amount of sharpening if u implement prewitt it will add 1 if its enhanced 
# if u implement sobel u will add a to the sobel kernel.
# now you have addition and after that you have to add to KERNEL['sharpen'] that u choose to sharping filter in Kernels.
# so kernel will be kernel+addition
# and then apply filter for every pixel like any other linear filter.

def apply_sharpening_filter(image, a):
    addition = np.array([
        [0, 0, 0],
        [0, a, 0],
        [0, 0, 0]]).astype('float')

    im_output = image.copy()    
    im_output = cv2.GaussianBlur(im_output, (3, 3), 0)
    dst = cv2.Laplacian(im_output, cv2.CV_16S, ksize = 3)
    Laplacian = cv2.convertScaleAbs(dst)
    return Laplacian

######################
#  Time Transmission #
######################
# returns time transmission in seconds just type the equation and return the time
def time_transmission(img, baudrate, channels):
    time = 0
    # write ur code here
    
    return time


###################
#    Histogram    #
###################

# returns 1d array have its length = 255, for example at index 0 it has how many 0 appears
# if you implement this function means that u implement show_hist() and both_hist() 
# dont touch show_hist() and both_hist() it will work if you implement hist() correctly

def hist(grayscale):
    return [0, 1, 2, 3]

def show_hist(figure, grayscale):
    plt = figure.subplots()
    plt.clear()

    h = hist(grayscale)
    plt.bar(range(256), h)

    figure.canvas.draw()

def both_hist(figure, original, modified):
    plt = figure.subplots()
    plt.clear()
    
    h1 = hist(original)
    h2 = hist(modified)
    plt.plot(range(256), h1, color='r', label='original')
    plt.plot(range(256), h2, color='g', label='output')

    figure.canvas.draw()



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('For testing purposes')
    # for example
    print(hist(load_grayscale(TEST_IMAGES['eren'])))







