import numpy as np
from sklearn.neighbors import KernelDensity
import time

file_path =  "/home/data6T/pxy/pointnet.pytorch/sonar_data/pts_scene/origin_data/016546.pts"
data = np.loadtxt(file_path).astype(np.float32)
intensity = data[:,3]
data = data[:,:3]

start_time = time.time()

# 计算每个点的高斯核密度
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)

# 设置三维空间的网格范围和分解率
x_min, x_max = data[:,0].min(), data[:,0].max()
y_min, y_max = data[:,1].min(), data[:,1].max() 
z_min, z_max = data[:,2].min(), data[:,2].max()

nx, ny, nz = 10, 10, 10

x_grid = np.linspace(x_min, x_max, nx)
y_grid = np.linspace(y_min, y_max, ny)  
z_grid = np.linspace(z_min, z_max, nz)

X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid)

# 获取每个小格体中的点
positions = np.c_[X.ravel(), Y.ravel(), Z.ravel()] 

# 计算每个小格体的核密度总和  
density = kde.score_samples(positions).reshape(nx,ny,nz)

# 根据密度进行采样
p = density/density.sum()
samples_idx = np.random.choice(np.prod(density.shape), 1024, p=p.ravel())

end_time = time.time()
# 计算代码执行时间
execution_time = end_time - start_time
print("*************Code execution time:", execution_time*1000, "ms")

samples = positions[samples_idx]
print(samples)
#with open("tmp.pts", "w") as out:
#    np.savetxt(out,samples,fmt='%f')