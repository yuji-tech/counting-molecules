# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:13:49 2024

@author: YUJI-ISSHIKI
"""


import sub as sb
import glob

d = "imageset"
hlist = ["holder_name_1","holder_name_2","holder_name_3"]
slist = [[0,1,0],[1,1,2],[1,0,2]] #number of metal steps in images

for i,h in enumerate(hlist):
    hname = "count/"+h
    sb.file_generator(hname)

    for j,filename in enumerate(glob.glob(d+"/"+h+"/"+"*.sxm")):
        print(filename)
        fname = hname+"/"+filename[len(d)+len(h)+2:-4]
        
        im, xyrange = sb.read_data(filename)
        im = sb.flatten_images(im,xyrange,fname,slist[i][j])
        contours_dict = sb.get_contours(im, rescale=xyrange, minimum_separation=0.2, block_size=35, offset=-0.5, savename=fname)
        sorted_labels = sb.sort_contours(contours_dict['zernike_moments'], n_clusters = 5)
        sb.plot_all_templates(contours_dict['max_templates'],fname)
        sb.plot_sort_templates(contours_dict['max_templates'],sorted_labels,fname)
        sb.plot_unsorted(im,contours_dict['contours'],fname)
        sb.plot_single_templates(contours_dict['max_templates'],contours_dict['contours'],fname)
