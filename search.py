import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob

#variables
fileName = input('Enter the file name: ')
img1 = cv2.imread('people/' + fileName, 0) #haystack
img2s = glob.glob('people/templates/*') #needles
orb = cv2.ORB_create()

bestKp1 = None
bestImg2 = None
bestKp2 = None
bestMatches = None

for img in img2s:
    
    img2 = cv2.imread(img, 0)

    #get keypairs and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    #get BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    #get the matches and sort by accuracy (distance)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    second = "None" if bestMatches == None else str(bestMatches[0].distance)
    print(str(matches[0].distance) + " vs " + second)

    if bestMatches == None or matches[0].distance <= bestMatches[0].distance:
        bestKp1 = kp1
        bestImg2 = img2
        bestKp2 = kp2
        bestMatches = matches


#draw first 10 matches of the most accurate
img3 = cv2.drawMatches(img1, bestKp1, bestImg2, bestKp2, bestMatches[:10], None, -1, flags=2)
plt.imshow(img3)
plt.show()


