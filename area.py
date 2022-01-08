import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

''' 
insight: https://www.topbots.com/automate-3d-point-cloud-segmentation/
'''


path = r'C:\Users\lfcas\Documents\Internship\meshroom_reconstructions\Test_5\High_FeatureExtraction\sfm.ply'
pcd = o3d.io.read_point_cloud(path)
# o3d.visualization.draw_geometries([pcd])

''' 
RANSAC for planar shape detection in point clouds

3 parameters. These are the distance threshold (distance_threshold) from the plane to consider a point inlier or outlier,
the number of sampled points drawn (3 here, as we want a plane) to estimate each plane candidate (ransac_n) and the number of iterations (num_iterations). 
These are standard values, but beware that depending on the dataset at hand, the distance_threshold should be double-checked.
'''
plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

 	
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

color_pixels = np.asarray(pcd.colors)
loc_pixels = np.asarray(pcd.points)
input = np.hstack((color_pixels, loc_pixels))

n_colors = 11
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(input)

labels = kmeans.predict(input)


pallete = dict()
color_map = np.zeros((50*n_colors, 100,3), dtype=np.uint8)
for i in range(n_colors):
    np.random.seed(seed=i)
    pallete[i] = np.array([np.random.randint(256), np.random.randint(256), np.random.randint(256)])
    color_map[50*(i):50*(i+1)] = pallete[i]
plt.imshow(color_map)
plt.show()

segmentation = []
for label in labels:
    segmentation.append( np.vectorize(pallete.get)(label).tolist() )
    
pcd.colors = o3d.utility.Vector3dVector(np.array(segmentation)/255)
o3d.visualization.draw_geometries([pcd])

color_target = pallete[8]/255
index_target = np.where(np.all(color_pixels== color_target  ,axis=1))[0]
loc_target = loc_pixels[index_target]

filter_pcd = o3d.geometry.PointCloud()
filter_pcd.points = o3d.utility.Vector3dVector(loc_target)
filter_pcd.colors = o3d.utility.Vector3dVector(np.tile(color_target, (loc_target.shape[0],1)))
o3d.visualization.draw_geometries([filter_pcd])

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(filter_pcd, alpha=0.1)
filter_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30)) # max_nn = 30 default
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

print('Area of leaves: ', mesh.get_surface_area())

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3*avg_dist   
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    filter_pcd, o3d.utility.DoubleVector([radius, radius*2, radius*100, radius*200]))
o3d.visualization.draw_geometries([filter_pcd, rec_mesh])

bbox = filter_pcd.get_axis_aligned_bounding_box()

rec_mesh2 = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        filter_pcd, depth = 7, width = 0, scale = 1.1, linear_fit = False)[0]

rec_mesh2 = rec_mesh2.crop(bbox)

o3d.visualization.draw_geometries([filter_pcd, rec_mesh2])