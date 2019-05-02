import numpy as np
import scipy.io as sio

import scipy.spatial as sp
#from matplotlib.pyplot import imread

import matplotlib.pyplot as plt
import math
from scipy.misc import imread

from basicFunc import center_avg
from basicFunc import print_cent
from basicFunc import init_centroids
from basicFunc import graph


def show_image(B, k):
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    lenB=len(B)

    average = [None] *lenB
    centerPlace = -1

    buffer = []
    A = imread(path)
    A_norm = A.astype(float) / 255.
    lenA = len(A_norm)
    ACopy = [[0 for x in range(lenA)] for y in range(lenA)]

    for iteration in range(10):
        counter = 0
        sum = 0
        for line in range(lenA):
            for col in range(lenA):
                mini = -1
                for center in range(lenB):
                    #find the distance
                    d = sp.distance.euclidean(A_norm[line][col], B[center])
                    if (mini == -1):
                        mini = d
                        centerPlace = center
                    elif(d < mini):
                            centerPlace = center
                            mini = d
                #end of centers
                ACopy[line][col] = centerPlace
                counter += 1
                sum += mini
            #end of columns
        #end of lines
        buffer.append(sum/counter)

        print("iter", iteration, end="")
        print(":", end = " ")
        for center in range(lenB):
            print_cent((B[center]))
            end=""
            #print(end="")
            if (center != lenB - 1):
                print(", ", end="")
            B[center] = center_avg(ACopy, A_norm, center, B[center])
        # end of recalculating the center
        print()
    # end of iterations

    print("iter 10:", end=" ")
    for center in range(lenB):
        print_cent((B[center]))
        end = ""
        # print(end="")
        if (center != lenB - 1):
            print(", ", end="")

        for line in range(len(ACopy)):
            for col in range(len(ACopy[line])):
                if (ACopy[line][col] == center):
                    A_norm[line][col] = B[center]
    #graph(k, buffer)
    print()

    img_size = A_norm.shape

    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])
    # plot the image
    #plt.imshow(A_norm)
    #plt.grid(False)

    #plt.show()