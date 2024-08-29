# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:13:49 2024

@author: YUJI-ISSHIKI
"""
import numpy as np
import math
import cv2
from scipy.signal import find_peaks

def getTranslationMatrix2d(dx, dy):
    return np.matrix([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

def rotateImage(image,angle):
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
    trans_mat = np.identity(3)

    w2 = image_size[0] * 0.5
    h2 = image_size[1] * 0.5

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    tl = (np.array([-w2, h2]) * rot_mat_notranslate).A[0]
    tr = (np.array([w2, h2]) * rot_mat_notranslate).A[0]
    bl = (np.array([-w2, -h2]) * rot_mat_notranslate).A[0]
    br = (np.array([w2, -h2]) * rot_mat_notranslate).A[0]

    x_coords = [pt[0] for pt in [tl, tr, bl, br]]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in [tl, tr, bl, br]]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))
    new_image_size = (new_w, new_h)

    new_midx = new_w * 0.5
    new_midy = new_h * 0.5

    dx = int(new_midx - w2)
    dy = int(new_midy - h2)

    trans_mat = getTranslationMatrix2d(dx, dy)
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
    result = cv2.warpAffine(image, affine_mat, new_image_size, flags=cv2.INTER_LINEAR)

    return result

def rotation_matrix(x,y,ai):
    R = np.array([[np.cos(ai),-np.sin(ai)],[np.sin(ai),np.cos(ai)]])
    rlist = np.empty((2,0),int)
    for xi,yi in zip(x,y):
        u = np.array([xi,yi])
        u_rotated = np.dot(R,u)
        rlist = np.c_[rlist,u_rotated]
    xr,yr = rlist
    return  xr,yr

def symmetric_axis(angle):
    x = np.linspace(-100,100,100)
    y = math.tan(angle)*x
    return x, y

def local_minimum(a,r,l):
    mins, _ = find_peaks(r*-1)   
    dmin = 50
    if len(mins)>1:
        g = []
        mins_new = []

        for m in mins:
            if len(g)==0:
                g.append(m)
            if 0<m-g[-1]<dmin:
                g.append(m)
            if m-g[-1]>=dmin:
                mins_new.append(g[np.argmin(r[g])])
                g = [m]
        if len(g)>0:
            mins_new.append(g[np.argmin(r[g])])
        mins = mins_new
    return mins

def find_symmetry_axis(x,y,savename,num):
    x = x-np.average(x)
    y = y-np.average(y)
    a = np.linspace(0,math.pi,180)
    r = []
    
    for ai in a:
        xr,yr = rotation_matrix(x,y,ai)
        r_all = 0
        for i in range(len(xr)):
            r_all += min((xr+xr[i])**2+(yr-yr[i])**2)
        r.append(r_all)
    r = np.array(r)/len(x)
    a_min = a[np.argmin(r)]
    return a_min
