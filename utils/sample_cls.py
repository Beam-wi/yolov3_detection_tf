def parseData(line, clsList, clsNum, imSize, anchors, mode, dataSavePathTemp, nfsMountPath, classNumDic, trainWithGray):
    # if 'str' not in str(type(classNumDic)):
    if not isinstance(classNumDic, str):
        classNumDic = classNumDic.decode()
    classNumDic = json.loads(classNumDic.replace("'", "\""))

    classes_list_temp = []
    for class_name in clsList:
        classes_list_temp.append(class_name.decode())
    pic_path, boxes, labels = parse_line(line, classes_list_temp, prefix=None)

    img = readImg(pic_path, dataSavePathTemp, nfsMountPath)

    if img is None:
        glo.globalvar.logger.warning(f"read img is None: {pic_path} fill with 0")
        img = np.zeros((3648, 5472, 3), dtype=np.float32)
        # img = np.zeros((2048, 2448, 3), dtype=np.float32)
        boxes = np.empty((0), dtype=np.float32)
        labels = []

    setResize = glo.globalvar.config.data_set["detResize"]
    img, boxes = detResize(img, boxes, setResize) if isinstance(setResize, (list, tuple)) else (img, boxes)
    fill_zero_label_names = globalvar.globalvar.config.data_set["fill_zero_label_names"]
    img, boxes, labels = adjust_data(img, boxes, labels, fill_zero_label_names)

    h, w, c = img.shape[0], img.shape[1], img.shape[2]
    boxes_temp = []
    labels_temp = []
    for i, box in enumerate(boxes):
        box[0] = max(0, box[0])
        box[1] = max(0, box[1])
        box[2] = min(w, box[2])
        box[3] = min(h, box[3])
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:  # 过滤小于2的像素
            pass
        else:
            boxes_temp.append(box)
            labels_temp.append(labels[i])
    boxes = np.array(boxes_temp)
    labels = labels_temp
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # do data augmentation here
    img, boxes, labels, confs = data_augmentation(pic_path, img, boxes, labels, imSize, classNumDic, trainWithGray, False)

    img = img.astype(np.float32)
    img = img / 255.
    y_true_13, y_true_26, y_true_52 = process_box(pic_path, img, boxes, labels, imSize, clsNum, anchors)

    return img, y_true_13, y_true_26, y_true_52, pic_path