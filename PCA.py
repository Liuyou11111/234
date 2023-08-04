import open3d as o3d
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame

def PCA(points,correlation=False,sort=True):
    data_mean = np.mean(points, axis=0)  # 对列求取平均值
    # 归一化
    normalize_data = points - data_mean
    # print("normalize_data:",normalize_data)
    # SVD分解
    # 构造协方差矩阵
    H = np.dot(normalize_data.T, normalize_data)
    # print("H:",H)
    # SVD分解
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)  # H = U S V
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        # print("第一主成分：",eigenvectors[:,0])
        # print("第二主成分：",eigenvectors[:, 1])
    return eigenvectors, eigenvalues

def faxiangliang():
    points = np.genfromtxt("7point_cloudqz", delimiter=" ")
    points = DataFrame(points[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    points.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(points)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
    w,v=PCA(points)
    print("w:",w)
    point_cloud_vector=w[:,2]
    print("主方向：",point_cloud_vector)


    #计算法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)  #对点云建立kd树 方便搜索
    normals = []
    # print(point_cloud_o3d)  #geometry::PointCloud with 10000 points.

    x_sum=0
    y_sum=0
    z_sum=0
    n=points.shape[0]
    print(n)  # 10000
    points = np.array(points, dtype=np.float64)
    for i in range(n):
        x_sum += points[i][0]
        y_sum += points[i][1]
        z_sum += points[i][2]
    x_center = x_sum / n
    y_center = y_sum / n
    z_center = z_sum / n
    point_center_idx=0
    min_distance=99999.0
    for i in range(n):
        distance=(points[i][0]-x_center)**2+(points[i][1]-y_center)**2+(points[i][2]-z_center)**2
        if distance <min_distance:
            min_distance=distance
            point_center_idx=i
    colors=[]
    x_sum=0
    y_sum=0
    z_sum=0
    for i in range(n):
    # for i in range(10):
        # search_knn_vector_3d函数 ， 输入值[每一点，x]      返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        print(len(point_cloud_o3d.points))
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 10)  # 10 个临近点
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]  # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        print(k_nearest_point)
        # print(k_nearest_point)
        v, u = PCA(k_nearest_point)
        m = np.matmul(v[:, 2], [[0], [0], [-1]])
        # print(m)
        if m < 0:
            v[:, 2][2] = -v[:, 2][2]
            v[:, 2][1] = -v[:, 2][1]
            v[:, 2][0] = -v[:, 2][0]
        # print(v[:, 2])
        normals.append(v[:, 2])
        colors.append([1, 0, 0])
        print("vv",v[:, 2])
        x_sum=x_sum+ v[:, 2][0]
        y_sum=y_sum+ v[:, 2][1]
        z_sum=z_sum+ v[:, 2][2]
    y = pow(x_sum * x_sum + y_sum * y_sum + z_sum * z_sum,0.5)
    print("yyy:",y)
    for i in range(n):
        if i ==point_center_idx:
            normals[i]=[x_sum/y,y_sum/y,z_sum/y]
        else:
            normals[i] =[0,0,0]

    normals = np.array(normals, dtype=np.float64)
    print("法：",normals[point_center_idx])
    # for n in range(points.shape[0]):
    #     if normals[n][2]<0:
    #         normals[n][2]=-normals[n][2]
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    faxiangliang()
