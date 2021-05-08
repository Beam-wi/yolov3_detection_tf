import os
import sys
import numpy as np

from utils.get_data_from_web import get_data_from_web_and_save

if __name__ == '__main__':
    batch_path = "Q:/ai_model_ckpt/sjht/20210407_model_lbhy_v1.0/20210407_lbhy_data_batch.txt"
    class_name_path = "Q:/ai_model_ckpt/sjht/20210407_model_lbhy_v1.0/class_names.txt"
    class_name_writer = open(class_name_path, "w")
    train_num, val_num, class_num_dic = get_data_from_web_and_save(data_batch_path=batch_path,
                                                                   data_train_save_path="Q:/ai_model_ckpt/sjht/20210407_model_lbhy_v1.0/train_data.txt",
                                                                   data_val_save_path="Q:/ai_model_ckpt/sjht/20210407_model_lbhy_v1.0/val_data.txt",
                                                                   nfs_mount_path="Q:/",
                                                                   project_name="sjht")



    print(class_num_dic)

    for key, value in class_num_dic.items():
        if value > 200:
            class_name_writer.write(key)
            class_name_writer.write("\n")
        else:
            print(key)

    class_name_writer.close()

