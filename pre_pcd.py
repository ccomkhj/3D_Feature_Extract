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



class PointCloud:
    ''' Class containing the important processes of preprocessing.
    
    Args: 
        pcd: open3d pointcloud
        scaling_factor: scaling factor for the real distance measurements
    
    '''
    
    def __init__(self, pcd, scaling_factor=1):
        self.pcd = pcd
        # self.pcd.estimate_normals()
        # self.floor =[]
        if scaling_factor == 1:
            self.x_ratio = 1
            self.y_ratio = 1
            self.z_ratio = 1
        
    def outlier_removal(self, nb_neighbors, std_ratio, verbose = False):
        '''Statistical_outlier_remval
        
        Args:
            nb_neighbors:
            std_ratio:
        '''
        
        self.pcd, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                            std_ratio=std_ratio)
        
           
        # utils.display_inlier_outlier(downpcd, ind)
        
        
    def display(self):
        '''Display current state of pcd'''
        utils.display_geo([self.pcd])
        
    def floor_allignment(self):
        ''' Allign pcd with floor and wall'''
        
        #Downsample point cloud and find floor plane
        downpcd = self.pcd.voxel_down_sample(voxel_size = 0.1)
        floor, inliers = downpcd.segment_plane(distance_threshold =  0.02, ransac_n = 3, num_iterations = 1000)
        a, b, c, d = floor
        f = downpcd.select_by_index(inliers)
        origin= f.get_center()
        #Compute floor normals translate pcd
        f.estimate_normals()
        f.translate(-origin)
        # display_pcd(f)
        self.pcd.translate(-origin)
        f.orient_normals_to_align_with_direction()
        
        #Normal Average
        v = np.average(np.asarray(f.normals), axis=0)
        
        #align z axis with normal vector of floor
        rz = np.arctan2(v[1],v[0])
        ry = np.arctan2(np.sqrt(v[1]*v[1]+v[0]*v[0]),v[2])
        R_z = np.asarray([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R_y = np.asarray([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        self.pcd.rotate(R_z,center= (0,0,0))  
        self.pcd.rotate(R_y,center= (0,0,0))
        
        #rotate 180 degrees if the points have negative z value
        avg_z = np.average(np.asarray(pcd.points), axis = 0)[2]
        
        # print(np.asarray(pcd.points))
        if avg_z>0:
            R = np.array([[-1,0,0], [0,1,0], [0,0,-1]])
            self.pcd.rotate(R,center= (0,0,0))
        
    def crop_floor(self, ratio = 0.2):
        ''' Crop detected floor from point cloud.
        
        Args:
            ratio: percentage of z axis to remove, default = 2.
        
        '''
        aabox = self.pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(aabox.get_min_bound())
        max_bound = np.asarray(aabox.get_max_bound())
        
        new_bound = (max_bound[2] - min_bound[2] ) * ratio
        min_bound[2] += new_bound
        
        aabox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        
        self.pcd = self.pcd.crop(aabox)
        
        
        
    def dbscan(self):
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                self.pcd.cluster_dbscan(eps=0.03, min_points=30, print_progress=True))
        
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(r"C:\Users\lfcas\Documents\Internship\3D_Feature_Extract\farm.ply")
    pointcloud = PointCloud(pcd)
    pointcloud.outlier_removal(nb_neighbors=20, std_ratio=1)
    
    # Alling pcd with floor and croping
    pointcloud.floor_allignment()
    pointcloud.crop_floor()
    # pointcloud.display()
    
    pointcloud.outlier_removal(nb_neighbors=20, std_ratio=1)
    #part of the 
    
    # pointcloud.dbscan()
    print(pointcloud.pcd.points)
    print(pointcloud.pcd.colors)
    
    # pointcloud.display()
    # utils.save_pcd(r'C:\Users\lfcas\Documents\Internship\3D_Feature_Extract\pcd_1.las', pointcloud.pcd)
    
    
    # display_inlier_outlier(pcd, ind)
      

 





