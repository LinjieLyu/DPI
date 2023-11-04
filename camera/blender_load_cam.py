import bpy
import math
import random
from mathutils import Vector
import os
from math import degrees,radians
import numpy as np



cam_file = './360_v2/bonsai/c2w.npz'

camera_dict = np.load(cam_file)
print(sorted(camera_dict.files))
c2w = camera_dict['c2w']
fx=camera_dict['fx']
fy=camera_dict['fy']     
cx=camera_dict['cx']
cy=camera_dict['cy']       

print(c2w[0].shape)
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

scale=1

for c in range(len(c2w)):
 
    RT=c2w[c]
    
 
    camera_data = bpy.data.cameras.new(name='Camera.%d' % c)
    camera = bpy.data.objects.new('Camera.%d' % c, camera_data)
    bpy.context.scene.collection.objects.link(camera)
    
    scene = bpy.context.scene
    sensor_width_in_mm = K[1,1]*K[0,2] / (K[0,0]*K[1,2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0,2]*2  # principal point assumed at the center
    resolution_y_in_px = K[1,2]*2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0,0] / s_u
    # recover original resolution
    if c==0:
        scene.render.resolution_x = (int)(resolution_x_in_px / scale+ 0.5)
        scene.render.resolution_y = (int)(resolution_y_in_px / scale+ 0.5)
        scene.render.resolution_percentage = scale * 100
        
    # Set the new camera as active
    for i in range(3):
        for j in range(4):
            camera.matrix_world[i][j] = RT[i,j]
#    camera.data.lens_unit = 'FOV'
    camera.data.type = 'PERSP'
    camera.data.lens = f_in_mm 
    camera.data.lens_unit = 'MILLIMETERS'
    camera.data.sensor_width  = sensor_width_in_mm
    
