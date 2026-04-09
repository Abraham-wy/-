import pandas as pd
import glob
import matplotlib.pyplot as plt
import os, shutil
from sklearn.model_selection import train_test_split
img_path = glob.glob('D:\\work_data\\2408flower\\*\\*')
print(img_path[:5])
f_class = [p.split('\\')[-2] for p in img_path]
f_num = pd.Series(f_class).value_counts()
print(f_num)
plt.pie(f_num,
        autopct='%.2f%%',
        labels= f_num.index)
plt.show()
def flower_split(input_path, output_path, train_size, seed):
        # 创建新文件夹
        try:
                os.makedirs(output_path)
                print('flower_split文件夹已创建')
        except:
                print('flower_split文件夹已存在')

        # 创建train文件夹
        try:
                os.makedirs(output_path + 'train')
                print('train文件夹已创建')
        except:
                print('train文件夹已存在')

        # 创建val文件夹
        try:
                os.makedirs(output_path + 'val')
                print('val文件夹已创建')
        except:
                print('val文件夹已存在')

        # 创建各种花的train文件夹
        for i in os.listdir(input_path):
                try:
                        os.makedirs(output_path + 'train' + '\\' + i)
                except:
                        print('%s的train文件夹已存在' % i)

        # 创建各种花的val文件夹
        for i in os.listdir(input_path):
                try:
                        os.makedirs(output_path + 'val' + '\\' + i)
                except:
                        print('%s的val文件夹已存在' % i)

        # 划分 + 复制copy或者移动move
        for class_name in os.listdir(input_path):
                pic = glob.glob(input_path + class_name + '\\*')
                train, val = train_test_split(pic, train_size=train_size, shuffle=True, random_state=seed)
                for i in train:
                        pic_index_i = i.split(sep='\\')[-1]
                        shutil.copy(i, output_path + 'train' + '\\' + class_name + '\\' + pic_index_i)
                for j in val:
                        pic_index_j = j.split(sep='\\')[-1]
                        shutil.copy(j, output_path + 'val' + '\\' + class_name + '\\' + pic_index_j)
        return print('已完成')
flower_split(input_path = 'D:\\work_data\\2408flower\\',
            output_path = 'D:\\work_data\\flower_split\\',
            train_size = 0.8,
            seed = 42)