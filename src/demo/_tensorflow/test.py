import tensorflow as tf
import pathlib

print(tf.__version__)
# 获取当前路径
data_root = pathlib.Path.cwd()
print(data_root)

# # 获取指定目录下的文件路径（返回是一个列表，每一个元素是一个PosixPath对象）
# all_image_paths = list(data_root.glob('*/*/*'))
# print(type(all_image_paths[0]))
# # 将PosixPath对象转为字符串
# all_image_paths = [str(path) for path in all_image_paths]
# print(all_image_paths[0])
# print(data_root)
