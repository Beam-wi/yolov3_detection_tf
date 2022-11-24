## Describe
sjht嵌件检测项目，AGX-4.2与AGX-4.4 tensorflow版本嵌件检测和分类模型训练及测试。

### 安装依赖

```txt
pip3 install -r requirements.txt
```

### 目录结构

```txt
./data：数据
./data/config/config_common.yaml：默认参数配置文件
./data/config/config_project.yaml：工程参数配置文件，每个工厂都不相同
./data/train_data：训练、测试需要的一些通用缓存文件
./data/darknet_weights/yolov3.ckpt：检测模型初始化参数加载模型
./data/darknet_weights/yolov3_classfication.ckpt：分类模型初始化参数加载模型
./utils：工具箱   
./utils/data_augmentation.py：训练过程中数据扩充接口  
./utils/data_utils.py：训练前处理，读取数据，扩充等操作   
./utils/eval_utils.py：训练过程中测试用的接口  
./utils/focal_loss.py：部分损失函数  
./utils/layer_utils.py：网络各层实现  
./utils/misc_utils.py：数据随机排序等实现  
./utils/nms_utils.py：nms实现  
./utils/plot_utils.py：测试结果可视化接口
./utils/data_select_balance.py：数据赛选
./utils/model.py：模型结构
./utils/cut_box_data_mutil.py：多进程数据裁剪
./utils/config_parse.py：配置文件读取脚本
./utils/globalvar.py：全局参数文件
./utils/logging_util.py：日志文件
./utils/classify_utils.py：分类模型一些接口
./utils/test_utils.py：测试模型用到得一些接口
./：可执行文件及其他说明
./auto_train_demo.txt：自动化训练及测试按理
./get_data_class_names.py：获取数据里面的嵌件名称
./combine_test_auto.py：检测模型结合分类模型自动化测试脚本
./combine_test.sh：检测模型结合分类模型自动化测试脚本
./train_detection.sh：检测模型自动化训练脚本
./train_detection_auto.py：检测模型自动化训练脚本
./train_classify.sh：分类模型自动化训练脚本
./train_classify_auto.py：分类模型自动化训练脚本
./classify_test_auto.py：分类模型嵌件自动化测试脚本
./classify_test.sh：分类模型自动化测试脚本
```

### Distributed Train | 模型分布式训练
训练工程配有两个参数脚本分别为[默认参数][默认参数]和[常用参数][常用参数]，其中常用参数优先级高于默认参数，但默认参数中包含了所有参数。多GPU参数也在参数脚本中选择。另外需要指定CUDA相关环境变量。
- 检测模型  
    [训练脚本][检测脚本]执行指令如下。  
    
    ```angular2html
    export PATH=/usr/local/cuda-10.0/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export CUDA_HOME=/usr/local/cuda
  
    sudo python3 train_det.py --config_project ./data/config/project.yaml --config_common ./data/config/common.yaml
    ```  
    全部参数理解可参考[默认参数][默认参数]和[常用参数][常用参数]，下面是正常训练一个脚本时所需要用到的参数案例。  
    常用参数  
    `nfs_mount_path` nfs挂载路径，由于标注平台获取的图片路径一般都是相对于nfs挂载路径的相对路径，所以这里要提供。  
    `data_save_path_temp` 本地数据缓存路径，训练检测模型必须设置。  
    `detection_train_file_path` 选练集绝对路径，必须给定。  
    `detection_val_file_path` 测试集绝对路径可以填多个，也可以填多个，测试集内容可以为空，但文件名至少指定一个。  
    `anchors` yolov3的anchor配置3组每组3个值分别对应三个特征层三个输出通道，实际计算要乘参数*anchorRotate*，该参数在[默认参数][默认参数]中修改。  
    `class_name_path` 分类模型嵌件类别信息。  
    `model_and_temp_file_save_path` 模型训练测试以及模型存储路径。  
    `clsDirName` 指定分类数据集名称，同一个路径下可以存放多个分类数据集，通过名称区分。  
    `need_cut_object` 如果已经生成好分类数据集这里填False，当为True时训练前会通过训练集生成分类数据集覆盖已有数据集。  
    `detection_model_save_name` 检测模型名称。  
    `detection_train_gpu_device` 目标检测指定GPU序号，字符串以英文逗号相连。  
    `detection_total_epoches` 检测模型训练总轮数。  
    `detection_save_freq` 检测模型存储轮数。  
  
    默认参数  
    `oriSize` 输入图片的原始尺寸。  
    `mergeLabel` 是否合并所有类别，世纪华通嵌件检测需合并，工件检测一般不合并。  
    `detResize` 目标检测时是否resize，None or [w, h], 填充resize, 专为裁截时尺寸超出原始尺寸设置。  
    `detRotate` 目标检测时数据增强旋转角度范围，一般按照工件大小和特征点特性确定旋转角度。  
    `anchorRotate` *anchors*乘以该比值后表示对应到原图目标的大小。  
    `detFilterSize` 目标检测过滤尺寸大小[w, h]。  
    `image_size` 我们训练目标检测的规则是在原来的图片上裁剪一个尺寸然后将裁剪图片热resize到image_size进行训练达到多尺度效果，裁剪的大小是*image_size*与*scale*乘积。  
    `scale` 与*image_size*连用实现多尺度训练。  
    `train_with_gray` 检测模型是否按照灰度训练，注意模型转化时要保持一致。  
    检测模型会存储在*model_and_temp_file_save_path/detection_model_save_dir_name*。  
    注意：当嵌件不能旋转平移等操作时需手动到[数据增强脚本][检测数据增强]屏蔽掉函数*dataAugmentation*中不适合的操作代码。  


- 分类模型  
    分布式[训练脚本][分类脚本]执行指令如下，也可以使用单GPU[训练脚本][单分类脚本]。    
    ```angular2html
    export PATH=/usr/local/cuda-10.0/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export CUDA_HOME=/usr/local/cuda
  
    sudo python3 train_cls_mutil.py --config_project ./data/config/project.yaml --config_common ./data/config/common.yaml
    ```  
    全部参数理解可参考[默认参数][默认参数]和[常用参数][常用参数]，下面是正常训练一个脚本时所需要用到的参数案例。  
    常用参数  
    `nfs_mount_path` nfs挂载路径，由于标注平台获取的图片路径一般都是相对于nfs挂载路径的相对路径，所以这里要提供。  
    `data_save_path_temp` 本地数据缓存路径，训练检测模型必须设置。  
    `detection_train_file_path` 如果分类数据集已存在，无需给定。  
    `detection_val_file_path` 测试集绝对路径可以填多个，也可以填多个，测试集内容可以为空，但文件名至少指定一个。  
    `class_name_path` 分类模型嵌件类别信息。  
    `model_and_temp_file_save_path` 模型训练测试以及模型存储路径。  
    `clsDirName` 指定分类数据集名称，同一个路径下可以存放多个分类数据集，通过名称区分。  
    `need_cut_object` 如果已经生成好分类数据集这里填False，当为True时训练前会通过训练集生成分类数据集覆盖已有数据集。  
    `classify_model_save_name` 分类模型名称。  
    `other_gpu_device` 分类训练时指定多GPU序号，字符串以英文逗号相连。  
    `extension_ratio` 分类训练目标扩充比例。  
    `cutRatio` 分类训练随机裁剪比例，为0时不裁, 一般为(extension_ratio-0.95*test_ext_ratio)/extension_ratio，test_ext_ratio为测试时的扩充比例。  
    `fill_box2square` 是否把嵌件扩充到矩形再进行分类，训练分类和测试时使用，默认为True。  
    `classify_total_epochs` 训练分类总轮数。  
    默认参数  
    `classify_size` 分类模型输入尺寸。  
    分类模型会存储在*model_and_temp_file_save_path/classify_model_save_dir_name*。  
    注意：当嵌件不能旋转平移等操作时需手动到[数据增强脚本][分类数据增强]屏蔽掉函数*data_augmentation*中不适合的操作代码。  

### Model Test Analysis | 模型测试分析  
- 联合测试  
    支持有标注和无标注测试集测试，[联合测试][测试脚本]执行指令如下。  
    ```angular2html
    export PATH=/usr/local/cuda-10.0/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64
    export CUDA_HOME=/usr/local/cuda
  
    sudo python3 combine_test_workpiece.py --config_project ./data/config/project.yaml --config_common ./data/config/common.yaml
    ```  
    全部参数理解可参考[默认参数][默认参数]和[常用参数][常用参数]，下面是正常训练一个脚本时所需要用到的参数案例。  
    常用参数   
    `nfs_mount_path` nfs挂载路径，由于标注平台获取的图片路径一般都是相对于nfs挂载路径的相对路径，所以这里要提供。  
    `data_save_path_temp` 本地数据缓存路径，训练检测模型必须设置。  
    `detection_val_file_path` 测试集绝对路径可以填多个，也可以填多个。  
    `anchors` 与训练保持一致。  
    `class_name_path` 分类模型嵌件类别信息。  
    `model_and_temp_file_save_path` 模型训练测试以及模型存储路径。  
    `detection_model_save_name` 检测模型名称。  
    `classify_model_save_name` 分类模型名称。  
    `detection_train_gpu_device` 目标检测指定GPU序号，字符串以英文逗号相连。  
    `other_gpu_device` 指定测试GPU，给定多个时默认使用第一个。  
    `extension_ratio` 测试扩充比例，一般为 *extension_ratio(1-cutRatio)/0.95*, *extension_ratio*为训练时扩充比。  
    `do_classify` True表示使用联合模型。  
    `writeWBox` 是否把测试过程中分类错误和漏检的box存储下来，以便分析。  
    `writeWFull` 是否只存测试错误的整图, 否则存所有图片。  
    默认参数  
    `classify_size` 分类模型输入尺寸。  
    `new_size_detection`: 测试时输入尺寸*[w, h]*，这个尺寸理论上要在多尺度泛化范围里。  
    测试类容会存储在 *model_and_temp_file_save_path/combine_test_result_save_dir_name*。  








[默认参数]: data/config/common.yaml
[常用参数]: data/config/project.yaml
[检测脚本]: train_det.py
[分类脚本]: train_cls_mutil.py
[单分类脚本]: train_cls.py
[测试脚本]: combine_test_workpiece.py
[检测数据增强]: utils/data_augmentation.py
[分类数据增强]: utils/classify_utils.py