# -*- coding: utf-8 -*-
""" Point Cloud preprocessing:
    
Loads the farm point cloud and reduces it to the wanted plants:
    Whole scene outlier removal
    Removes the background
    Plant outlier removal

"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import utils 



class Mesh:
    ''' Class containing the important processes of preprocessing.
    
    Args: 
        mesh: open3d Trianglemesh
        scaling_factor: scaling factor for the real distance measurements
        n_points: number of points to sample from mesh
        
    
    '''
    
    def __init__(self, mesh, n_points= 1000, scaling_factor=1):
        self.mesh = mesh
        self.scaling_factor = scaling_factor    
        self.pcd = self.mesh.sample_points_uniformly(number_of_points= n_points)
    
    def sample(self, n_points):
        ''' Create sample pcd of mesh to facilitate preprocessing

        Parameters
        ----------
        n_points : number of points to sample

        '''
        self.pcd = self.mesh.sample_points_uniformly(number_of_points= n_points)
        
        
    def display(self):
        '''Display current state of pcd'''
        utils.display_geo([self.mesh])
        
    def display_pcd(self):
        utils.display_geo([self.pcd])
        



if __name__ == '__main__':
    
    # mesh = o3d.io.read_triangle_mesh(r'C:\Users\lfcas\Documents\Internship\3D_Feature_Extract\mesh.obj')
    # mesh.paint_uniform_color([1, 0.706, 0])
    
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points= 1000000)
    utils.display_geo([pcd])
    
    
    # display_inlier_outlier(pcd, ind)
      

 





