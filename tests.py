# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:59:11 2022

@author: lfcas
"""
import pclpy
from pclpy import pcl
import numpy as np


plyreader = pclpy.pcl.io.PLYReader()
pc = pclpy.pcl.PointCloud.PointXYZRGB()
plyreader.read(r'C:\Users\lfcas\Documents\Internship\3D_Feature_Extract\pcd_1.ply',pc)