import os
import numpy as np
import random
import math
import copy


def parse_line(line):
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    pic_path = s[0]
    s = s[1:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = s[i * 5], float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)

    boxes = np.asarray(boxes, np.float32)
    return pic_path, boxes, labels


def calculate_class_info(data_lines, name2id):
    classes_line_point = {classname:[] for classname in list(name2id.keys())}
    classes_line_point_set = {}
    classes_num = []
    for line in data_lines:
        pic_path, boxes, labels = parse_line(line)
        for label in labels:
            if label not in ["wxqy", "hscfxpmgp", "background"]:
                classes_line_point[label].append(line)

    for key in classes_line_point.keys():
        classes_line_point_set[key] = list(set(classes_line_point[key]))

    for key in classes_line_point.keys():
        classes_num.append(len(classes_line_point[key]))


    return classes_line_point, classes_line_point_set, classes_num

def random_select_data(lines, class_name, ask_num):
    lines_ori = copy.deepcopy(lines)
    select_line = []
    select_num = 0
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    while select_num < ask_num:
        line = lines.pop(0)
        pic_path, boxes, labels = parse_line(line)
        select_num = select_num + labels.count(class_name)
        select_line.append(line)
        if len(lines) == 0:
            lines = copy.deepcopy(lines_ori)
            random.shuffle(lines)
    print("%s:ori line num %d, select line num %d"%(class_name, len(lines_ori), len(select_line)))

    return select_line

def get_data_batch_name(lines):
    batch_names = []
    batch_datas = {}
    for line in lines:
        image_path = line.split("\n")[0].split(" ")[0]
        batch_name = image_path.split("/")[-2]
        if len(batch_name) < 5:
            batch_name = image_path.split("/")[-3]
        if batch_name not in batch_names:
            batch_names.append(batch_name)
        if batch_name not in list(batch_datas.keys()):
            batch_datas[batch_name] = []
            batch_datas[batch_name].append(line)
        else:
            batch_datas[batch_name].append(line)
    return batch_names, batch_datas

def get_background_data(lines, hard_background_batch, batch_select_num = 0):
    background_lines = []
    background_batch_lines = {}
    background_select_lines = []
    for line in lines:
        if len(line.split("\n")[0].split(" ")) < 5:
            background_lines.append(line)
            image_path = line.split("\n")[0].split(" ")[0]
            batch_name = image_path.split("/")[-2]
            if len(batch_name) < 5:
                batch_name = image_path.split("/")[-3]
            if batch_name not in list(background_batch_lines.keys()):
                background_batch_lines[batch_name] = []
            background_batch_lines[batch_name].append(line)

    for key in list(background_batch_lines.keys()):
        batch_lines = background_batch_lines[key]
        if key in hard_background_batch:
            batch_select_num_temp = len(batch_lines)
        else:
            batch_select_num_temp = batch_select_num
        random.shuffle(batch_lines)
        repetion_times = max(math.floor(batch_select_num_temp / len(batch_lines)), 1)
        add_num = int(max(batch_select_num_temp - repetion_times * len(batch_lines), 0))
        for time in range(repetion_times):
            if len(batch_lines) > batch_select_num_temp:
                background_select_lines = background_select_lines + batch_lines[:batch_select_num_temp]
            else:
                background_select_lines = background_select_lines + batch_lines
        background_select_lines = background_select_lines + batch_lines[: add_num]


    return background_lines, background_batch_lines, background_select_lines


def build_data(data_ori_path, data_select_path, classes_file, classes_select_num=None, hard_background_batch=None, select_num_last=None):
    # data_ori_path = "data/train_data/sjht_common_20200717_train_data_combine_add.txt"
    data_ori_lines = open(data_ori_path).readlines()
    # data_select_path = "./data/train_data/sjht_common_20200717_train_data_combine_selected.txt"
    data_select_write = open(data_select_path, "w")
    # classes_file = "./data/train_data/20200715classes_combine_sjht-sy-common.txt"
    classes_name = []
    for line in open(classes_file, "r").readlines():
        classes_name.append(line.split("\n")[0].split(" ")[0])
    classes_name = classes_name[1:]
    name2id = {}
    id = 0
    combine_name = {}
    for line in open(classes_file, "r").readlines():
        if len(line.split("\n")[0].split(" ")) > 2:
            for name in line.split("\n")[0].split(" ")[1:]:
                combine_name[name] = line.split("\n")[0].split(" ")[0]
        for name in line.split("\n")[0].split(" "):
            name2id[name] = id
        id = id + 1

    batch_name_ori, batch_datas_ori = get_data_batch_name(data_ori_lines)
    background_lines, background_batch_lines, background_select_lines = get_background_data(data_ori_lines, hard_background_batch, batch_select_num=50)
    classes_line_point, classes_line_point_set, classes_num = calculate_class_info(data_ori_lines, name2id)
    sorted_nums = sorted(enumerate(classes_num), key=lambda x: x[1])

    select_lines = []
    balance_num_ori = 2000

    classes_id_selected = []
    find_all = False
    id = 0
    while not find_all:
        if sorted_nums[id][1] == 0 or sorted_nums[id][0] == 0:
            pass
        else:
            classes_line_point_temp, classes_line_point_set_temp, classes_num_temp = calculate_class_info(select_lines, name2id)
            classes_num = sorted_nums[id][1]
            class_name = list(classes_line_point_set.keys())[sorted_nums[id][0]]
            class_line = classes_line_point_set[list(classes_line_point_set.keys())[sorted_nums[id][0]]]
            random.shuffle(class_line)
            if class_name in list(combine_name.keys()):
                balance_num = int(balance_num_ori / 1.)
            elif class_name in list(classes_select_num.keys()):
                balance_num = classes_select_num[class_name]
            else:
                balance_num = balance_num_ori
            select_num = (balance_num - classes_num_temp[sorted_nums[id][0]]) if classes_num_temp[sorted_nums[id][0]] < balance_num else 0
            select_lines = select_lines + random_select_data(class_line, class_name, select_num)

        classes_id_selected.append(sorted_nums[id][0])
        id = id + 1
        if len(classes_id_selected) == len(list(name2id.keys())):
            find_all = True
    select_lines = select_lines + background_select_lines
    classes_line_point_result, classes_line_point_set_result, classes_num_result = calculate_class_info(select_lines, name2id)
    batch_name_select, batch_datas_select = get_data_batch_name(select_lines)

    # syht2_batch = "./data/train_data/20200924_syht2_batch_list.txt"
    # syht2_batch_data = open(syht2_batch, "w")
    # for batch in batch_name_ori:
    #     syht2_batch_data.write(batch)
    #     syht2_batch_data.write("\n")
    # syht2_batch_data.close()

    batch_name_not_select = list(set(batch_name_ori) - set(batch_name_select))
    print("not select bacth name:", batch_name_not_select)

    print("start select not select batch data")

    batch_select_lines = []
    for batch_name in batch_name_not_select:
        batch_lines = batch_datas_ori[batch_name]
        random.shuffle(batch_lines)
        batch_select_lines = batch_select_lines + batch_lines[:400]

    select_lines = select_lines + batch_select_lines
    batch_name_select, batch_datas_select = get_data_batch_name(select_lines)

    batch_name_not_select = list(set(batch_name_ori) - set(batch_name_select))
    print("not select bacth name after again:", batch_name_not_select)

    print("select line first step num:", len(select_lines))
    random.shuffle(select_lines)
    if select_num_last is not None:
        if len(select_lines) >= select_num_last:
            select_lines = select_lines[:select_num_last]
        else:
            select_lines = select_lines + select_lines[:(select_num_last-len(select_lines))]
    print("select line second step num:", len(select_lines))

    for key in list(classes_line_point_result.keys()):
        print("class_name:%s    ori_point_num %d    select_point_num %d"%(key, len(classes_line_point[key]), len(classes_line_point_result[key])))

    for line in select_lines:
        data_select_write.write(line)

if __name__ == '__main__':
    data_ori_path = "data/train_data/20200924_syht2_data_combine.txt"
    data_select_path = "data/train_data/20200924_syht2_data_combine_selected.txt"
    classes_file = "./data/train_data/20200924classes_combine_sjht-sy-common.txt"
    class_select_num_path = "./data/train_data/class_select_num_20200924.txt"
    classes_select_num = {}
    for line in open(class_select_num_path).readlines():
        key, value = line.split("\n")[0].split(" ")
        classes_select_num[key] = int(value)
    # classes_select_num = {"heisejsqj03":10000, "hssjqj04h":10000, "hsjsqj05-fm":6000, "yhuijsqj01":6000, "hsjsqj05-zm":6000, "hsslqj04h":6000, "hsslqj05h-zm":6000}
    hard_background_batch = ["sjht-20191105144929324966", "20191128170440-22", "20191215132549-57", "20191215143341-58", "20191216084334-98", "20191215103835-71", "20191215154025-97"]
    build_data(data_ori_path, data_select_path, classes_file, classes_select_num=classes_select_num, hard_background_batch=hard_background_batch, select_num_last=None)




# ['sjht-20191030101730679267', '20200608150648-50', '20191128122241-21', '20200612151426-50', '20191128120808-21', '20200628134138-46', '20200402104219-11', 'sjht-20191016143048683165', '20200611083015-50', '20191128124236-21', '20200401154524-3']
