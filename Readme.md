# sjht嵌件检测项目，嵌件检测模型训练及测试

## 安装依赖

```txt
pip3 install -r requirements.txt
```

## 目录结构

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

## 模型训练及测试  
config参数解释请看./data/config/config_common.yaml,./data/config/config_project.yaml,config_common.yaml是通用配置文件，
非必要参数在里面都有默认设置，如果要修改，请在工程配置文件config_project.yaml中修改、添加，如果参数在config_project.yaml设置，
以config_project.yaml为准，如果参数不在config_project.yaml设置，以config_common.yaml参数为准。
### 训练检测模型  
##### 用sh脚本训练  
sudo bash ./train_detection.sh ./data/config/config_project.yaml ./data/config/config_common.yaml  
##### 用python脚本训练  
python3 train_detection_auto.py --config_project=./data/config/config_project.yaml --config_common=./data/config/config_common.yaml  

### 分类模型训练  
##### 用sh脚本训练  
sudo bash ./train_classify.sh ./data/config/config_project.yaml ./data/config/config_common.yaml  
##### 用python脚本训练
python3 train_classify_auto.py --config_project=./data/config/config_project.yaml --config_common=./data/config/config_common.yaml  

### 综合测试  
##### 用sh脚本测试  
sudo bash ./combine_test.sh ./data/config/config_project.yaml ./data/config/config_common.yaml  
##### 用python脚本测试
python3 combine_test_auto.py --config_project=./data/config/config_project.yaml --config_common=./data/config/config_common.yaml  

### 分类模型测试  
##### 用sh脚本测试  
sudo bash ./classify_test.sh ./data/config/config_project.yaml ./data/config/config_common.yaml  
##### 用python脚本测试  
python3 classify_test_auto.py --config_project=./data/config/config_project.yaml --config_common=./data/config/config_common.yaml  

## 注意事项  
注意训练检测模型时候可以采用多gpu训练，其余默认选择设置得第一个gpu。
