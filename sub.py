# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:13:49 2024

@author: YUJI-ISSHIKI
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mahotas
import scipy.optimize
from scipy.spatial import distance
from scipy import optimize as _optimize
from sklearn.cluster import SpectralClustering
from skimage.filters import threshold_otsu
from skimage.draw import polygon
from skimage.measure import find_contours
import sxmReader
import rot as rm
import os
import statistics
import math

################################   FILE   #####################################

def file_generator(filename):
    if os.path.exists(filename+"/"):
        pass
    else:
        os.mkdir(filename+"/")  

def read_data(fn):
    load = sxmReader.NanonisSXM(fn)
    xx = load.retrieve_channel_data(load.channels_name[0])
    xx = (xx-np.nanmin(xx))*1e9
    xx = np.nan_to_num(xx, nan=0)
    x_range,y_range = [float(k)*1e9 for k in load.header['SCAN_RANGE'][0]]
    xyrange = [x_range,y_range]
    return xx, xyrange

############################# PLOT SOMETHING ##################################
def plot_all_templates(templates,fname):
    print("Number of molecules =", len(templates))
    cr = get_rows(len(templates))   
    extent = (-1,1,-1,1)
    fig = plt.figure(figsize=(10,10))
    for i in range(len(templates)):
        
        fig.add_subplot(cr,cr, i+1)
        plt.imshow(templates[i],cmap=cm.grey,extent=extent)
        # plt.plot(x,math.tan(angles[i])*x,ls="--",color="w")
        
        plt.tick_params(length=0,direction="in")
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
    plt.savefig(fname+"-templates.png", bbox_inches = 'tight', pad_inches = 0, dpi=300)
    plt.show()

def plot_sort_templates(templates,labels,fname):
    n = max(labels)+1
    n_im = []
    for i in range(n):
        n_im.append(labels.count(i))
    
    n_cr = []
    for i in range(n):
        n_cr.append(get_rows(n_im[i]))
    
    for i in range(n):
        fig = plt.figure(figsize=(10,10))
        k = 0
        for j in range(len(templates)):
            if labels[j]==i:
                fig.add_subplot(n_cr[i],n_cr[i],k+1)
                plt.imshow(templates[j],cmap=cm.grey)
                k += 1
            plt.tick_params(length=0,direction="in")
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
        plt.savefig(fname+"-templates-"+str(i)+".png", bbox_inches = 'tight', pad_inches = 0, dpi=300)
        plt.show()

def plot_single_templates(templates,contours,fname):
    # templates, angles = rotate_imagedata(templates,contours)
    file_generator(fname)
    extent = (-1,1,-1,1)
    for i in range(len(templates)):
        plt.figure(figsize=(10,10))
        plt.imshow(templates[i],cmap=cm.grey,extent=extent)
        plt.tick_params(length=0,direction="in")
        plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
        plt.savefig(fname+"/"+str(i)+".png", bbox_inches = 'tight', pad_inches = 0, dpi=150)
        plt.show()

def plot_unsorted(im,real_contours,fname):
    rescale=(1,1)
    plt.figure(constrained_layout=True, figsize=(6.73,6.73))
    extent = (0, im.shape[1]*rescale[1], im.shape[0]*rescale[0], 0)
    plt.imshow(im, cmap=cm.gray, extent=extent)
    plt.gca().axis('off')
    for i,c in enumerate(real_contours):
        tempx = np.multiply(c[:,1], rescale[0])
        tempy = np.multiply(c[:,0], rescale[1])
        plt.plot(tempx, tempy, color='yellow', linewidth=1)
    plt.savefig(fname+"_image_with_contours.png",bbox_inches='tight',pad_inches = 0,dpi=300)
    plt.show()

def plot_twod(im,rescale,savename):
    extent = (0, im.shape[1]*rescale[1], im.shape[0]*rescale[0], 0)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    im = ax.imshow(im, cmap=cm.grey, interpolation='nearest',extent=extent)
    ax.tick_params(length=0, labelbottom=False,labelleft=False,labelright=False,labeltop=False)  
    plt.savefig(savename,bbox_inches = 'tight',pad_inches = 0,dpi=300)
    plt.show()

############################ PROCESS FOR PLOT #################################
def get_rows(n):
    for i in range(100):
        if i**2>n:
            break
    return i

def rotate_imagedata(templates,contours):
    savename = "a"
    angles = []
    rot_templates = []
    for i,c in enumerate(contours):
        angle = rm.find_symmetry_axis(c[:,0],c[:,1],savename,i)
        angles.append(angle)
        template = rm.rotateImage(templates[i],angle*180/math.pi)
        rot_templates.append(template)
    return rot_templates, angles

def rescale_boxsize(templates,labels):
    n = max(labels)+1
    for i in range(n):
        boxsize = 1
        for j in range(len(templates)):
            if labels[j]==i:
                if boxsize<np.shape(templates[j])[0]:
                    boxsize=np.shape(templates[j])[0]
            
        for j in range(len(templates)):
            template = templates[j]
            nx,ny=np.shape(template)
            
            if labels[j]==i:
                if boxsize>nx:
                    new_template = np.zeros((boxsize, boxsize))
                    
                    mx = int((boxsize-nx)/2)
                    my = int((boxsize-ny)/2)
                    
                    for l in range(nx):
                        for k in range(ny):
                            new_template[l+mx][k+my]= template[l][k]
                    templates[j]=new_template
    return(templates)

######################## PROCESS FOR FLATTEN IMAGES ###########################
def _plane(a0,a1,b1,x0,y0):
    return lambda x,y: a0+a1*(x-x0)+b1*(y-y0)

def _planemoments(xx):
    a0 = np.abs(xx).min()
    index = (xx-a0).argmin()
    x,y = xx.shape
    x0 = float(index/x)
    y0 = float(index%y)
    a1 = 0.0
    b1 = 0.0
    return a0,a1,b1,x0,y0

def _fitplane(xx):
    params = _planemoments(xx)
    errorfunction = lambda p: np.ravel(_plane(*p)(*np.indices(xx.shape))-xx)
    p, success = _optimize.leastsq(errorfunction,params)
    return p

def _return_plane(params,xx):
    _fit_data = _plane(*params)
    return _fit_data(*np.indices(xx.shape))

def _plane_fit_2d(xx):
    return xx-_return_plane(_fitplane(xx),xx)

def _convert_binary(xx,th_binary):
    x_pixel = np.shape(xx)[0]
    xxc = xx.copy()
    for i in range(x_pixel):
        x = xxc[i]
        x[x>th_binary]=True
        x[x<=th_binary]=False
        xxc[i] = x
    return(xxc)

def linear(para,x):
    y = para[0]*x+para[1]
    return(y)

def objectiveFunctionl(para,x,y):
    e = y-linear(para,x)
    return(e)

def fitlinear(x,y):
    para_int = [1,1]
    param_output = scipy.optimize.leastsq(objectiveFunctionl,para_int,args=(x,y),ftol=0.0000001,full_output=True)
    para = param_output[0]
    return(para)

def im_remove_spike(xx):
    xxc = xx.copy()
    for i in range(len(xx)):
        d = np.diff(xx[i])
        s = np.where(abs(d)>0.008)[0]
        for j in range(len(s)-1):
            if s[j+1]-s[j]<20:
                for k in range(s[j+1]-s[j]):
                    xxc[i][s[j]+k] = None
    return xxc-np.nanmin(xxc)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolation_none_data(xx):
    for i, y in enumerate(xx):
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        xx[i] = y
    return xx

def background_segment(xx,m):
    n = np.shape(xx)[0]
    p = np.empty((3,0),int)
    k = int(n/m)
    
    for i in range(k):
        for j in range(k):
            xx_seg = xx[i*m:(i+1)*m,j*m:(j+1)*m]
            p = np.c_[p,[i,j,np.var(xx_seg)]]
    i,j = [int(p[0][np.argmin(p[-1])]),int(p[1][np.argmin(p[-1])])]
    xxs = im_remove_spike(xx)
    xxs = interpolation_none_data(xxs)
    xx_seg = xxs[i*m:(i+1)*m,j*m:(j+1)*m]

    xx = xx-_return_plane(_fitplane(xx_seg),xx)
    return xx

def check_cross_section(xx):
    for i in range(len(xx)):
        plt.plot(xx[i],"b",alpha=10/len(xx))
        plt.plot(xx.T[i],"r",alpha=10/len(xx))
    plt.show()

def filter_data(xx,xyrange,savename):
    plot_twod(xx,xyrange,savename+"-raw.png")
    check_cross_section(xx)
    xx = background_segment(xx,int(np.shape(xx)[0]/4))
    check_cross_section(xx)
    plot_twod(xx,xyrange,savename+"-filtered.png")
    return xx-np.min(xx)

def binary(xx):
    xxc = xx.copy()
    xxc[xxc>=0.15]=1
    xxc[xxc<0.15]=0
    return xxc

def delete_molecule(xxb,n):
    xxbc = xxb.copy()
    for i, xi in enumerate(xxb):
        d = abs(np.diff(xi))
        if max(d)>0:
            index = np.where(d>0)[0]
            for j in range(index[0],index[-1]+1):
                if xxbc[i][j]==n:
                    xxbc[i][j]-=1
    return xxbc

def subtract_steps(xx,b):
    xx[b==0] = xx[b==0]-np.average(xx[b==0])
    xx[b==1] = xx[b==1]-np.average(xx[b==1])
    if np.count_nonzero(b==2)>0:
        xx[b==2] = xx[b==2]-np.average(xx[b==2])
    xx[b==5] = np.average(xx[b!=5])
    return xx

def extract_boundary(b,bc):
    m = 5
    n = np.shape(b)[0]
    for i in range(n):
        d = np.diff(b[i])
        for j, di in enumerate(d):
            if di != 0:
                for k in range(m):
                    if 0<j-k:
                        bc[i][j-k]=5
                    if j+k<n:
                        bc[i][j+k]=5
    return bc

def binary_double(xx):
    xxc = xx.copy()
    xxc[xxc>=0.36]=2
    xxc[(0.36>xxc)&(xxc>=0.18)]=1
    xxc[xxc<0.18]=0
    return xxc

def delete_molecule_double(b):
    bc = b.copy()
    bc[bc>0]=1
    bc = delete_molecule(bc,1)
    b[bc==0]=0
    b = delete_molecule(b,2)
    return(b)

def get_boundary(b):
    d = [[1,0],[1,1],[1,-1],[0,1],[0,-1],[-1,0],[-1,1],[-1,-1]]
    index = np.where(b==5)[0]
    boundary = index.copy()
    for ii in index:
        for di in d:
            boundary.append(ii+di)
    return boundary
            
###########################  FLATTEN IMAGE  ###################################

def flatten_images(xx,xyrange,savename,step):
    xx = filter_data(xx,xyrange,savename)
    if step>0:
        if step==1:
            b = binary(xx)
            b = delete_molecule(b,1)
        if step==2:
            b = binary_double(xx)
            b = delete_molecule_double(b)
        bc = b.copy()
        bc = extract_boundary(b,bc)
        bc = extract_boundary(b.T,bc.T).T
        plot_twod(bc,xyrange,savename+"-boundary.png")
        xx = subtract_steps(xx,bc)
        
    check_cross_section(xx)
    plot_twod(xx,xyrange,savename+"-flat.png")
    return xx

###################### FUNCTION FOR CATEGORIZING ##############################
def normalized_x(x):
    x = x-min(x)
    x = x/max(x)
    return x

def seleting_zernike_moments(df):
    df_new = []
    df = df.T
    for i in range(len(df)):
        xi = normalized_x(df[i])
        vi = statistics.pvariance(xi)
        if vi>0.05:
            df_new.append(xi)
    df_new = np.array(df_new).T
    return(df_new)      

def rotation_matrix(x,y,ai):
    R = np.array([[np.cos(ai),-np.sin(ai)],[np.sin(ai),np.cos(ai)]])
    rlist = np.empty((2,0),int)
    for xi,yi in zip(x,y):
        u = np.array([xi,yi])
        u_rotated = np.dot(R,u)
        rlist = np.c_[rlist,u_rotated]
    xr,yr = rlist
    return  xr,yr

def smooth_data(x,y,n):
    xs = []
    ys = []
    for i in range(len(x)):
        if (i+1)*n<len(x):
            xs.append(np.average(x[i*n:(i+1)*n]))
            ys.append(np.average(y[i*n:(i+1)*n]))
    return np.array(xs), np.array(ys)

def smooth_data_all(x,y,n):
    xs = []
    ys = []
    for i in range(len(x)-n):
        xs.append(np.average(x[i:i+n]))
        ys.append(np.average(y[i:i+n]))
    return np.array(xs), np.array(ys)

def making_single_img(im,poly,otsu_output,boxsize):
    center = int(boxsize/2)
    centerx,centery = [int(np.mean(poly[0])), int(np.mean(poly[1]))]
    template = np.zeros((boxsize,boxsize))
    translate_poly = (poly[0] - centerx + center, poly[1] - centery + center)

    if max(translate_poly[0])>=boxsize or max(translate_poly[1])>=boxsize:
        x,y = poly
        centerx,centery = [int((max(x)+min(x))/2),int((max(y)+min(y))/2)]
        translate_poly = (poly[0] - centerx + center, poly[1] - centery + center)

    template[translate_poly] = im[poly] - otsu_output
    return template

def select_contours(im,contours,min_pixels,max_pixels,otsu_output):
    real_contours = []
    for c in contours:
        if min(c[:,0])>2 and max(c[:,0])<im.shape[0]-2:
            if min(c[:,1]) > 2 and max(c[:,1]) < im.shape[1]-2:
                if min_pixels<len(c)<max_pixels: #eliminate small diry
                    poly = polygon(c[:,0],c[:,1])
                    if np.average(im[poly])-otsu_output>0: #eliminate dirty which shows negative
                        real_contours.append(c)
    return real_contours

def connect_close_data(real_contours,rescale,minimum_separation):
    new_contours = []
    used_indexes = []
    angles = []
    for ii, cc in enumerate(real_contours):
        poly1 = polygon(cc[:,0], cc[:,1])
        for jj, cc2 in enumerate(real_contours):
            if jj>ii and ii not in used_indexes:
                poly2 = polygon(cc2[:,0], cc2[:,1])
                dist = distance.cdist(cc*rescale, cc2*rescale)
                answerx = np.isin(poly2[0], poly1[0])
                answery = np.isin(poly2[1], poly1[1])
                if np.all(np.logical_and(answerx, answery)):
                    used_indexes.append(jj)
                if np.amin(dist) < minimum_separation:
                    cc = np.concatenate((cc, cc2))
                    used_indexes.append(jj)
        if ii not in used_indexes:
            new_contours.append(cc)
    return new_contours, angles

def get_imagesize(new_contours):
    diagonals = []
    for c in new_contours:
        x,y = c.T
        diagonal = int(np.sqrt((np.max(x)-np.min(x))**2 + (np.max(y)-np.min(y))**2))
        diagonals.append(diagonal+1)
    return diagonals

def get_templates(im,new_contours,diagonals,otsu_output):
    templates, max_templates, contour_lengths, max_pixels = [[],[],[],[]]
    for i,c in enumerate(new_contours):
        poly = polygon(c[:,0],c[:,1])
        template = making_single_img(im,poly,otsu_output,max(diagonals))
        templates.append(template)
        max_template = making_single_img(im,poly,otsu_output,diagonals[i])
        max_templates.append(max_template)
        contour_lengths.append(len(c))
        max_pixels.append(np.amax(template) - otsu_output)
    return templates, max_templates, contour_lengths, max_pixels

def get_zernike_moment(max_templates, contour_lengths, max_pixels, zernike_radius):
    zernike_moments = []
    for template, length, pixel in zip(max_templates, contour_lengths, max_pixels):
        degree = 20
        answer = mahotas.features.zernike_moments(template, degree = degree, radius=zernike_radius)
        answer = np.append(answer, pixel)
        answer = np.append(answer, length)
        zernike_moments.append(answer)
    zernike_moments = np.asarray(zernike_moments)
    zernike_moments = seleting_zernike_moments(zernike_moments)
    return zernike_moments

#############################    COUNTING    ##################################
def get_contours(im, minimum_separation=0.2, rescale=(1,1), zernike_radius=None, block_size=35, offset=0, savename=None):
    im = im/np.amax(im)
    im = _plane_fit_2d(im)
    otsu_output = threshold_otsu(im)/4
    binary_local = _convert_binary(im,otsu_output)
    contours = find_contours(binary_local,0.5)
    
    min_radius, max_radius = [1.3,1000]
    min_pixels = 2*np.pi*min_radius*np.shape(im)[0]/rescale[0]
    max_pixels = 2*np.pi*max_radius*np.shape(im)[0]/rescale[0]

    real_contours = select_contours(im,contours,min_pixels,max_pixels,otsu_output)
    new_contours, angles = connect_close_data(real_contours,rescale,minimum_separation)
    diagonals = get_imagesize(new_contours)
    templates, max_templates, contour_lengths, max_pixels = get_templates(im,new_contours,diagonals,otsu_output)

    contour_lengths = [xx / max(contour_lengths) for xx in contour_lengths]
    max_pixels = [xx / max(max_pixels) for xx in max_pixels]
    if zernike_radius == None:
        zernike_radius = int(np.median(diagonals))
    zernike_moments = get_zernike_moment(max_templates,contour_lengths,max_pixels,zernike_radius)
    
    contours_dict = {}
    contours_dict['image'] = im
    contours_dict['rescale'] = rescale
    contours_dict['contours'] = new_contours
    contours_dict['otsu_threshold'] = otsu_output
    contours_dict['templates'] = templates
    contours_dict['max_templates'] = max_templates
    contours_dict['angles'] = angles
    contours_dict['contour_lengths'] = contour_lengths
    contours_dict['max_pixels'] = max_pixels
    contours_dict['zernike_moments'] = zernike_moments
    return contours_dict

#############################  CATEGORIZING  ##################################
def sort_contours(zernike_moments, n_clusters=None):
    af = SpectralClustering(n_clusters=n_clusters).fit(zernike_moments)
    bin_reorder = np.flipud(np.argsort(np.bincount(af.labels_)))
    new_labels = [0 for ii in range(len(af.labels_))]
    for ii, ll in enumerate(bin_reorder):
        for jj, label in enumerate(af.labels_):
            if label == ll:
                new_labels[jj] = ii
    return new_labels
###############################################################################




