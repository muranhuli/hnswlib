import h5py
import numpy as np
from sklearn.cluster import KMeans
import argparse

# 读取file_path，由外部传入
# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Plot CDF from a file containing one numeric value per line.")

# 添加参数：文件路径
parser.add_argument("file_path", help="Path to the input file containing the data")

# 解析命令行参数
args = parser.parse_args()


# 1. 加载HDF5数据
file_path = args.file_path

# 打开HDF5文件并读取/train目录下的数据
with h5py.File(file_path, 'r') as f:
    train_data = f['/train'][:]

# 检查数据形状
print("训练数据形状：", train_data.shape)

# 2. 设置聚类数目 n
n_clusters = train_data.shape[0]//13  # 你可以根据需求调整 n 的值

# 3. 使用KMeans进行聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=3000, tol=1e-5)

# 进行聚类计算
kmeans.fit(train_data)

# 获取聚类结果：每个样本所属的聚类类别
labels = kmeans.labels_

# 获取聚类中心
cluster_centers = kmeans.cluster_centers_

# 4. 将标签和质心保存到原HDF5文件中
with h5py.File(file_path, 'r+') as f:  # 重新以读写模式打开文件
    # 如果数据集已存在，删除旧的，以便写入新的数据
    if '/kmeans_labels' in f:
        del f['/kmeans_labels']
    if '/kmeans_centers' in f:
        del f['/kmeans_centers']

    # 创建新的数据集保存标签
    f.create_dataset('/kmeans_labels', data=labels)

    # 创建新的数据集保存质心
    f.create_dataset('/kmeans_centers', data=cluster_centers)

print("聚类标签和质心已保存到 HDF5 文件中。")



