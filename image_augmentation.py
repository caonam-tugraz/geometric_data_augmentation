import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data

from PIL import Image, ImageOps
import random
import cv2
import matplotlib.pyplot as plt

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

# ****************************************************************************************

class HistogramEqualize:
    """Performs histogram equalization

    """

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, image: Image):
        if random.random() <= self.p:
            # open_cv_image = np.array(image)
            # src = open_cv_image[:, :, ::-1]

            # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # equalized = cv2.equalizeHist(src)
            equalized = ImageOps.equalize(image)
            return equalized
        else:
            return image

# ****************************************************************************************

def plot_1_channel_img(_input):
    temp = np.dstack([_input, _input, _input])
    im_pil = Image.fromarray(temp)
    plt.imshow(im_pil)
    
    
    
class ImageToSketch:
    """Performs ImageToSketch
    """

    def __init__(self, p=0.7, dim=(224, 224)):
        self.p = p
        self.dim = dim

    def __call__(self, image: Image):
        if random.random() <= self.p:
            # use numpy to convert the pil_image into a numpy array
            numpy_image = np.array(image)

            # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
            # the color is converted from RGB to BGR format
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            img_gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            img_smoothing = cv2.GaussianBlur(img_gray, (21, 21), sigmaX=0, sigmaY=0)
            sketched_img = cv2.divide(img_gray, img_smoothing, scale=255)

            # plot_1_channel_img(sketched_img)
            
            equalized = cv2.equalizeHist(sketched_img)
            # plot_1_channel_img(equalized)

            # apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            morph = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel)

            kernel = np.ones((2, 2), np.uint8)
            img_dilation = cv2.dilate(morph, kernel, iterations=1)

            plot_1_channel_img(img_dilation)

            # contours2, hierarchy2 = cv2.findContours(img_dilation, cv2.RETR_TREE,
            #                                                cv2.CHAIN_APPROX_SIMPLE)
            # image_copy2 = img.copy()
            # # cv2.drawContours(image_copy2, contours2, -1, (0, 255, 0), 2, cv2.LINE_AA)

            # # image_copy3 = img.copy()
            # for i, contour in enumerate(contours2): # loop over one contour area
            #     # draw a circle on the current contour coordinate
            #     if cv2.contourArea(contour) > 5:
            #         cv2.drawContours(image_copy2, contour, -1, (0, 255, 0), 2, cv2.LINE_AA)

            # plt.imshow(image_copy2)
            # plt.show()

            # canny = cv2.canny(sketched_img)
            temp = np.dstack([img_dilation, img_dilation, img_dilation])

            

            im_pil = Image.fromarray(temp)
            im_pil = ImageOps.invert(im_pil)



            return im_pil
        else:
            return image

#%%
def color_to_sketch(img_color):
 
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    img_inverted = 255 - img_gray

    # Apply Gaussian blur to the inverted image
    img_blur = cv2.GaussianBlur(img_inverted, (5,5), sigmaX=0, sigmaY=0)

    # Invert the blurred image
    img_blend = cv2.divide(img_gray, 255 - img_blur, scale=256)

    return img_blend

#%%
class ImageToSketch2:
    """Performs ImageToSketch2
    """

    def __init__(self, p=0.7, dim=(224, 224)):
        self.p = p
        self.dim = dim

    def __call__(self, image: Image):
        if random.random() <= self.p:
            # use numpy to convert the pil_image into a numpy array
            numpy_image = np.array(image)
    
            # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
            # the color is converted from RGB to BGR format
            opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
            # plt.imshow(cv2_to_rgb(resized_image))
            # plt.show()
            
            sketch_image = color_to_sketch(opencv_image)
    
            inverted_image = cv2.bitwise_not(sketch_image)
            
            # plt.imshow(inverted_image, cmap = 'gray')
            # plt.show()
            
    
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            sketch_image = cv2.erode(sketch_image, kernel, 1)
            
            kernel = np.ones((2, 2), np.uint8)
            img_dilation = cv2.dilate(sketch_image, kernel, iterations=1)
    
            # One more time
            sketch_image = cv2.erode(img_dilation, kernel, 1)
            
            kernel = np.ones((2, 2), np.uint8)
            img_dilation = cv2.dilate(sketch_image, kernel, iterations=1)
    
            # Invert the colors of the image
            inverted_image1 = cv2.bitwise_not(img_dilation)
            
            # plt.imshow(inverted_image, cmap = 'gray')
             # plt.show()
             
            if random.random() < 0.5:
                temp = np.dstack([inverted_image, inverted_image, inverted_image])
            else:
                temp = np.dstack([inverted_image1, inverted_image1, inverted_image1])

            im_pil = Image.fromarray(temp)

            return im_pil
        else:
            return image
 

        
 #%%
class ImageToSketch3:
     """Performs ImageToSketch3
     """

     def __init__(self, p=0.7, dim=(224, 224)):
         self.p = p
         self.dim = dim

     def __call__(self, image: Image):
         if random.random() <= self.p:
             # use numpy to convert the pil_image into a numpy array
             numpy_image = np.array(image)
     
             # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
             # the color is converted from RGB to BGR format
             opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
     
             # plt.imshow(cv2_to_rgb(resized_image))
             # plt.show()
             
             sketch_image = color_to_sketch(opencv_image)
     
             inverted_image = cv2.bitwise_not(sketch_image)
             
             # plt.imshow(inverted_image, cmap = 'gray')
             # plt.show()
             
     
             kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
             sketch_image = cv2.erode(sketch_image, kernel, 1)
             
             kernel = np.ones((2, 2), np.uint8)
             img_dilation = cv2.dilate(sketch_image, kernel, iterations=1)
     
             # One more time
             sketch_image = cv2.erode(img_dilation, kernel, 1)
             
             kernel = np.ones((2, 2), np.uint8)
             img_dilation = cv2.dilate(sketch_image, kernel, iterations=1)
     
             # Invert the colors of the image
             inverted_image1 = cv2.bitwise_not(img_dilation)
             
             # plt.imshow(inverted_image, cmap = 'gray')
              # plt.show()
              
             if random.random() > 0.2:
                 temp = np.dstack([inverted_image, inverted_image, inverted_image])
             else:
                 temp = np.dstack([inverted_image1, inverted_image1, inverted_image1])

             im_pil = Image.fromarray(temp)

             return im_pil
         else:
             return image
# ****************************************************************************************

class Tenengrad_filter:
    """Performs ImageToSketch
    """

    def __init__(self, p=0.7, ksize=3):
        self.p = p
        self.ksize = ksize

    def __call__(self, image: Image):
        if random.random() <= self.p:
            # use numpy to convert the pil_image into a numpy array
            numpy_image = np.array(image)
            gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)

            # Compute the gradients using Scharr kernel in x and y directions
            gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
            
            # response1 = (gx + gy)
            # response1 = cv2.normalize(response1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # temp = np.dstack([response1, response1, response1])
            
            # im_pil = Image.fromarray(temp)
            
            # plt.imshow(im_pil)
            
            # Compute the squared gradients and sum them up using a kernel
            k = np.ones((self.ksize, self.ksize), dtype=np.float32)
            gxx = cv2.filter2D(gx**2, -1, k)
            gyy = cv2.filter2D(gy**2, -1, k)
            
            # Compute the Tenengrad filter response as the square root of the sum of squared gradients
            response = np.sqrt(gxx + gyy)
            
            # Normalize the response to [0, 255] range
            response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            temp = np.dstack([response, response, response])
            
            im_pil = Image.fromarray(temp)
            
            # plt.imshow(im_pil)
            # im_pil = ImageOps.invert(im_pil)

            return im_pil
        else:
            return image

# ****************************************************************************************
class ImageToSketch_Tenengrad:
    """Performs ImageToSketch
    """

    def __init__(self, p=0.5, ksize = 3, dim=(224, 224)):
        self.p = p
        self.dim = dim
        self.ksize = ksize

    def __call__(self, image: Image):
        if random.random() <= self.p:
            if random.random() <= 0.5:
                # use numpy to convert the pil_image into a numpy array
                numpy_image = np.array(image)
    
                # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
                # the color is converted from RGB to BGR format
                opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
                img_gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    
                img_smoothing = cv2.GaussianBlur(img_gray, (21, 21), sigmaX=0, sigmaY=0)
                sketched_img = cv2.divide(img_gray, img_smoothing, scale=255)
    
                equalized = cv2.equalizeHist(sketched_img)
    
                # apply morphology
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                morph = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel)
    
                kernel = np.ones((2, 2), np.uint8)
                img_dilation = cv2.dilate(morph, kernel, iterations=1)
    
                plot_1_channel_img(img_dilation)
    
                temp = np.dstack([img_dilation, img_dilation, img_dilation])
    
                im_pil = Image.fromarray(temp)
                im_pil = ImageOps.invert(im_pil)
    
            else:
                # use numpy to convert the pil_image into a numpy array
                numpy_image = np.array(image)
                gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    
                # Compute the gradients using Scharr kernel in x and y directions
                gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
                gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)        
                
                # Compute the squared gradients and sum them up using a kernel
                k = np.ones((self.ksize, self.ksize), dtype=np.float32)
                gxx = cv2.filter2D(gx**2, -1, k)
                gyy = cv2.filter2D(gy**2, -1, k)
                
                # Compute the Tenengrad filter response as the square root of the sum of squared gradients
                response = np.sqrt(gxx + gyy)
                
                # Normalize the response to [0, 255] range
                response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                temp = np.dstack([response, response, response])
                im_pil = Image.fromarray(temp)
    
            return im_pil
        else:
            return image
   
# ****************************************************************************************

class Normalize_histogram:
    """Performs ImageToSketch
    """

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, image: Image):
        if random.random() <= self.p:
            # use numpy to convert the pil_image into a numpy array
            image_array = np.array(image)
            # Convert image array to BGR color space (if necessary)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
            # Convert image array to the Lab color space
            lab_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2Lab)
        
            # Split the Lab image into L, a, and b channels
            l_channel, a_channel, b_channel = cv2.split(lab_image)
        
            # Apply histogram equalization to the L channel
            l_channel_eq = cv2.equalizeHist(l_channel)
        
            # Merge the equalized L channel with the original a and b channels
            lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        
            # Convert the equalized Lab image back to the BGR color space
            equalized_image_array = cv2.cvtColor(lab_image_eq, cv2.COLOR_Lab2BGR)
        
            # Convert image array back to PIL image
            equalized_image = Image.fromarray(equalized_image_array)
        
            return equalized_image
        else:
            return image

# ****************************************************************************************     
class ImageToSketch1:
    """Performs ImageToSketch
    """

    def __init__(self, p=0.7, dim=(224, 224)):
        self.p = p
        self.dim = dim

    def __call__(self, image: Image):
        if random.random() <= self.p:
            # use numpy to convert the pil_image into a numpy array
            numpy_image = np.array(image)

            # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
            # the color is converted from RGB to BGR format
            # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)

            gray = cv2.resize(gray, self.dim, interpolation=cv2.INTER_AREA)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            cl1 = clahe.apply(gray)
            
            # Apply histogram equalization
            equ = cv2.equalizeHist(cl1)
            
            result = np.where(equ >= cl1, equ, cl1)
            
            temp = np.dstack([result, result, result])

            im_pil = Image.fromarray(temp)

            return im_pil
        else:
            return image

# ****************************************************************************************
class ImgInvert:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, image: np.array):
        if random.random() <= self.p:
            open_cv_image = np.array(image)
            src = open_cv_image[:, :, ::-1]
            # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            src = cv2.bitwise_not(src)

            src = Image.fromarray(src)

            return src

        else:
            return image

# ****************************************************************************************

class ImgCustomRotate:
    def __init__(self, p=0.8):
        self.p = p
        self.degrees = [0, 5, 10, 15, 85, 90, 95,
                        175, 180, 185, 265, 270, 275, 350, 355]
        

    def __call__(self, image: np.array):
        if random.random() <= self.p:
            # open_cv_image = np.array(image)
            # src = open_cv_image[:, :, ::-1]
            # # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # src = cv2.bitwise_not(src)
            # src = Image.fromarray(src)

            degree = self.degrees[random.randint(0, len(self.degrees)-1)]
            # print(degree)
            src = transforms.functional.rotate(image, angle=degree)

            return src

        else:
            return image

# ****************************************************************************************

class ImgCustomRotate1:
    def __init__(self, p=0.8):
        self.p = p
        self.degrees = [0, 90,
                        180, 270]

    
    def __call__(self, image: np.array):
        if random.random() <= self.p:
            # open_cv_image = np.array(image)
            # src = open_cv_image[:, :, ::-1]
            # # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # src = cv2.bitwise_not(src)
            # src = Image.fromarray(src)

            degree = self.degrees[random.randint(0, len(self.degrees)-1)]
            # print(degree)
            src = transforms.functional.rotate(image, angle=degree)

            return src

        else:
            return image
        
# ****************************************************************************************
class ImgShift:
    def __init__(self, p=0.8, translate=(0.15, 0.15)):
        self.p = p
        self.translate = translate

    def __call__(self, image: np.array):
        # totensor = transforms.ToTensor()
        # img_tensor = totensor(image)

        if random.random() <= self.p:
            affine_transfomer = transforms.RandomAffine(
                degrees=0, translate=self.translate)
            affine_imgs = affine_transfomer(image)

            return affine_imgs

        else:
            return image


# ****************************************************************************************
class ImgSharpen:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, image: np.array):
        if random.random() <= self.p:
            open_cv_image = np.array(image)
            src = open_cv_image[:, :, ::-1]
            # gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            gaussian_blur = cv2.GaussianBlur(src, (9, 9), 10)
            sharpened = cv2.addWeighted(src, 1.5, gaussian_blur, -0.5, 0)

            # sharpened = cv2.addWeighted(src, 3.5, gaussian_blur, -2.5, 0)
            # kernel = np.array([[0, -1, 0],
            #                    [-1, 5,-1],
            #                    [0, -1, 0]])
            # sharpened = cv2.filter2D(src=src, ddepth=-1, kernel=kernel)

            # src = cv2.bitwise_not(src)
            sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

            src = Image.fromarray(sharpened)

            return src

        else:
            return image
