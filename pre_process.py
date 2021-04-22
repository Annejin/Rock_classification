# -*-coding:utf-8-*-
# 1.根据rock_label_1.csv文件对数据集进行划分：将不同种类的岩石图片划分到不同的文件夹中
import os
import shutil
from collections import Counter

data_dir = "/home/xyjin/PycharmProjects/data_mining/data_test/rock"  # 数据集的根目录
label_file = 'rock_label_1.csv'  # 根目录中csv的文件名加后缀
train_dir = 'rock_data'  # 根目录中的训练集文件夹的名字
input_dir = 'classed_data'  # 用于存放拆分数据集的文件夹的名字，可以不用先创建，会自动创建


def reorg_rock_data(data_dir, label_file, train_dir, input_dir):
    # 读取训练数据标签，label.csv文件读取标签以及对应的文件名,加上enconding='gb2312'防止UTF8报错
    with open(os.path.join(data_dir, label_file), 'r', encoding='gb2312') as f:
        # 跳过文件头行（栏名称）
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
    labels = set(idx_label.values())

    # 获取训练集的数量便于数据集的分割
    num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
    # print(num_train)  #140

    # 训练集中数量最少一类的岩石的数量
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])

    # print(min_num_train_per_label)  #3

    # 判断是否有存放拆分后数据集的文件夹，没有就创建一个
    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    # 整理训练集，将数据集进行拆分复制到预先设置好的存放文件夹中。
    for train_file in os.listdir(os.path.join(data_dir, train_dir)):
        idx = train_file.split('.')[0]
        idx1 = idx[:-2]
        label = idx_label[idx1]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))


# 载入数据，进行数据的拆分
reorg_rock_data(data_dir, label_file, train_dir, input_dir)

#2.扩充数据集，遍历train_valid文件夹中的所以子文件夹进行扩充
#(暂时不知道如何遍历文件夹中的子文件以及子文件中的文件，调用7次函数**==后期需要优化**)

import os
import random
from PIL import Image

def random_cut_image(w,h,num,file_pathname,filename,after_path):
    """

    :param w: 设置裁剪的小图的宽度
    :param h: 设置裁剪的小图的高度
    :param num: 随机裁剪为num张的小图
    :param file_pathname: 文件位置
    :param filename: 文件名称
    :return:
    """
    im = Image.open(file_pathname + '/' + filename)
    img_size = im.size
    m = img_size[0]  # 读取图片的宽度
    n = img_size[1]  # 读取图片的高度
    for i in range(1, num+1):  # 裁剪为num张随机的小图
        x = random.randint(0, m - w)  # 裁剪起点的x坐标范围
        y = random.randint(0, n - h)  # 裁剪起点的y坐标范围
        stem, suffix = os.path.splitext(filename)
        name = stem
        region = im.crop((x, y, x + w, y + h))  # 裁剪区域
        region.save(after_path + "/" + name + "_" + str(i) + ".jpg")  # 存储剪裁后数据，str(i)是裁剪后的编号，此处是1到100

def read_path(file_pathname,after_path):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        num = int(filename[-5])
        if (num == 1):
            random_cut_image(768, 768, 200, file_pathname, filename,after_path)
        else:
            continue

#(暂时不知道如何遍历文件夹中的子文件以及子文件中的文件，调用7次函数**==后期需要优化**)
before_path1="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/浅灰色细砂岩"
after_path1="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/浅灰色细砂岩"
read_path(before_path1,after_path1)

before_path2="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/深灰色泥岩"
after_path2="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/深灰色泥岩"
read_path(before_path2,after_path2)

before_path3="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/深灰色粉砂质泥岩"
after_path3="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/深灰色粉砂质泥岩"
read_path(before_path3,after_path3)

before_path4="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/灰色泥质粉砂岩"
after_path4="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/灰色泥质粉砂岩"
read_path(before_path4,after_path4)

before_path5="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/灰色细砂岩"
after_path5="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/灰色细砂岩"
read_path(before_path5,after_path5)

before_path6="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/灰黑色泥岩"
after_path6="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/灰黑色泥岩"
read_path(before_path6,after_path6)

before_path7="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/train_valid/黑色煤"
after_path7="/home/xyjin/PycharmProjects/data_mining/data_process/rock/classed_data/augmentated_data/黑色煤"
read_path(before_path7,after_path7)

#3.手动剔除干扰成分

#4.划分测试集和验证集
import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 指向classed_data文件夹
    cwd = os.getcwd()
    data_root = os.path.join(cwd)   #split_data所在位置
    origin_rock_path = os.path.join(data_root, "augmentated_data")  #进行数据集划分的文件夹
    assert os.path.exists(origin_rock_path), "path '{}' does not exist.".format(origin_rock_path)

    rock_class = [cla for cla in os.listdir(origin_rock_path)
                    if os.path.isdir(os.path.join(origin_rock_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in rock_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in rock_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in rock_class:
        cla_path = os.path.join(origin_rock_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()



