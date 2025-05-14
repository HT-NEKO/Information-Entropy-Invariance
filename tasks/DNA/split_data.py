from sklearn.model_selection import train_test_split
import pandas as pd
import os

import ipdb

def split_data(input_df,y, output_dir, test_size=0.2, random_state=42):
    # 读取数据
    # 分割数据集
    train_df, valid_df = train_test_split(input_df, test_size=test_size, random_state=random_state,stratify=y)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练集和测试集
    # train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    # valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)   
    return train_df, valid_df