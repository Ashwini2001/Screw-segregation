######################
## Essential libraries
######################
import cv2
import numpy as np
import os
import math
import csv




########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Images'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))
ref = 500


def find_biggest_contour(image):
    image = image.copy()
    contours,hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour),contour) for contour in contours]
    biggest_contour = max(contour_sizes,key=lambda x:x[0])[1]
    mask = np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,contours,-1,255,-1)
    return biggest_contour,mask

def process(ip_image):
    # Convert to RGB
    image = cv2.cvtColor(ip_image,cv2.COLOR_BGR2RGB)

    # Blur the image to remove unwanted edges
    image_blur = cv2.GaussianBlur(image,(7,7),0)

    image_blur_hsv = cv2.cvtColor(image_blur,cv2.COLOR_RGB2HSV)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([-1,-1,-1])
    upper_red = np.array([255 ,255,85])

    # Threshold the HSV image to get only blue colors
    mask1 = cv2.inRange(image_blur_hsv, lower_red, upper_red)

    lower_red = np.array([40,30,40])
    upper_red = np.array([255 ,225,80])

    mask2 = cv2.inRange(image_blur_hsv, lower_red, upper_red)

    mask = mask1+mask2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    mask_closed = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    mask_clean = cv2.morphologyEx(mask_closed,cv2.MORPH_OPEN,kernel)
    big_screw_contour,mask = find_biggest_contour(mask_clean)

    rect = cv2.minAreaRect(big_screw_contour)
    width = rect[1][0]
    height = rect[1][1]
    print(width)
    print(height)
    if(height > ref):
        typeofscrew = "long"
    if(height <= ref):
        typeofscrew = "short"
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rectangled = cv2.drawContours(image,[box],0,(0,0,255),2)
    cv2.imshow("window", rectangled)
    cv2.waitKey(0);
    return  typeofscrew


####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
####################################################################
def main():
    i = 1
    line = []
    ## Reading 1 image at a time from the Images folder
    for image_name in os.listdir(images_folder_path):
        ## verifying name of image
        print(image_name)
        ## reading in image 
        ip_image = cv2.imread(images_folder_path+"/"+image_name)
        ## verifying image has content
        print(ip_image.shape)
        ## passing read in image to process function
        A = process(ip_image)
        ## saving the output in  a list variable
        line.append([str(i), image_name , str(A)])
        ## incrementing counter variable
        i+=1
    ## verifying all data
    print(line)
    ## writing to sortedlist.csv in Generated folder without spaces
    with open(generated_folder_path+"/"+'sortedlist.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(line)
    ## closing csv file    
    writeFile.close()



    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()
