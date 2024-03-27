import torch
import numpy as np
import time


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # 记录开始时间
    start_time = time.time()

    device = xyz.device
    #print(device)
    batchsize, ndataset, dimension = xyz.shape
    #to方法Tensors和Modules可用于容易地将对象移动到不同的设备（代替以前的cpu()或cuda()方法）
    # 如果他们已经在目标设备上则不会执行复制操作
    centroids = torch.zeros(batchsize, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batchsize, ndataset).to(device) * 1e10
    #randint(low, high, size, dtype)
    # torch.randint(3, 5, (3,))->tensor([4, 3, 4])
    farthest =  torch.randint(0, ndataset, (batchsize,), dtype=torch.long).to(device)
    #batch_indices=[0,1,...,batchsize-1]
    batch_indices = torch.arange(batchsize, dtype=torch.long).to(device)
    for i in range(npoint):
        # 更新第i个最远点
        centroids[:,i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
        # 计算点集中的所有点到这个最远点的欧式距离
        #等价于torch.sum((xyz - centroid) ** 2, 2)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        #取出每一行的最大值构成列向量，等价于torch.max(x,2)
        farthest = torch.max(distance, -1)[1]
    
    # 记录结束时间
    end_time = time.time()
    # 计算代码执行时间
    execution_time = end_time - start_time
    print("*************Code execution time:", execution_time, "seconds")

    return centroids

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    point_set = np.loadtxt("/home/data6T/pxy/pointnet.pytorch/sonar_data/pts_scene/origin_data/016546.pts").astype(np.float32)
    point_set = point_set[:,:3]
    point_arr = np.expand_dims(point_set,axis=0)
    xyz = torch.from_numpy(point_arr).to(device)
    npoints = 1024
    res = farthest_point_sample(xyz, npoint=npoints)
    #print(res)

    res_cpu = res.to("cpu").numpy()
    with open("tmp.pts", "w") as out:
        np.savetxt(out,point_set[res_cpu[0]],fmt='%f')

