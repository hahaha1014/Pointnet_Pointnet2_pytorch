import numpy as np
import random
import time
import torch

file_path =  "/home/data6T/pxy/pointnet.pytorch/sonar_data/pts_scene/origin_data/016546.pts"
data = np.loadtxt(file_path).astype(np.float32)
intensity = data[:,3]
points = data[:,:3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#points = torch.from_numpy(points).to(device)



def voxel_filter(point_cloud, leaf_size, num_points, use_centroid = True):
    filtered_points = []
    filter_index = []
    # 作业3
    # 屏蔽开始
    point_cloud = np.array(point_cloud)
    #列最大
    uppers = np.max(point_cloud,axis=0)
    lowers = np.min(point_cloud,axis=0)
    #print("uppers:", uppers)
    #print("lowers:",lowers)
    #print("(uppers - lowers)/leaf_size\n",(uppers - lowers)/leaf_size)
    #三个维度各有多少个点
    dims = np.ceil((uppers - lowers)/leaf_size)
    #print("dims", dims)
 
    indices = (point_cloud - lowers)//leaf_size
    #print("indices", indices.shape)
 
    #线性索引
    h_indices = indices[:,0] + indices[:,1]*dims[0] + indices[:,2]*dims[0]*dims[1]
    #print("h_indices",h_indices.shape)
 
    for h_index in np.unique(h_indices):
        points = point_cloud[h_indices == h_index]
        if use_centroid:
            filtered_points.append(np.mean(points,axis = 0))
        else:
            filtered_points.append(random.choice(points))
    # 屏蔽结束
 
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    if filtered_points.shape[0] > num_points*2:
        return voxel_filter(point_cloud, leaf_size*2, num_points, use_centroid)
    if filtered_points.shape[0] < num_points:
        return voxel_filter(point_cloud, leaf_size/1.5, num_points, use_centroid)
    return random.sample(list(filtered_points), num_points)

start_time = time.time()
filtered_points = voxel_filter(points,0.3, 1024, False)
end_time = time.time()
# 计算代码执行时间
execution_time = end_time - start_time
print("*************Code execution time:", execution_time*1000, "ms")

print(len(filtered_points))
with open("tmp.pts", "w") as out:
    np.savetxt(out,filtered_points,fmt='%f')