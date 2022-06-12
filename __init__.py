import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

plt.show()  # 新增此行代码
import numpy as np
import glob

# 读取每个文件中的图片
imgs_path = glob.glob('birds/*/*.jpg')
# 获取标签的名称
all_labels_name = [img_p.split('\\')[1].split('.')[1] for img_p in imgs_path]
# 把标签名称进行去重
labels_names = np.unique(all_labels_name)
# print(len(labels_names))
# 包装为字典,将名称映射为序号
label_to_index = dict((name, i) for i, name in enumerate(labels_names))
# print(label_to_index)
# 反转字典
index_to_label = dict((v, k) for k, v in label_to_index.items())
# print(index_to_label)
# 吧所有标签映射为序号
all_labels = [label_to_index.get(name) for name in all_labels_name]
# print(all_labels[:3])
# 将数据随机打乱，划分为训练数据及测试数据
np.random.seed(2021)  # 设置随机因子,伪随机，方便测试
random_index = np.random.permutation(len(imgs_path))
# 把图片路径和标签进行相同次序的打乱
imgs_path = np.array(imgs_path)[random_index]
all_labels = np.array(all_labels)[random_index]
# 切片,取80%作为训练数据，20%作为测试数据
i = int(len(imgs_path) * 0.8)
# 训练数据
train_path = imgs_path[:i]
train_labels = all_labels[:i]
# 测试数据
test_path = imgs_path[i:]
test_labels = all_labels[i:]
# 编写数据集
train_ds = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_path, test_labels))


# 读取图片路径并做预处理
def load_img(path, label):
    # 读取图片
    image = tf.io.read_file(path)
    # 解码
    image = tf.image.decode_jpeg(image, channels=3)
    # 重构图片大小
    image = tf.image.resize(image, [256, 256])
    # 转换数据类型
    image = tf.cast(image, tf.float32)
    # 归一化
    image = image / 255
    return image, label
#
