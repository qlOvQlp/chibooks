## CHIBOOKS 是什么
chibooks 是一个简易视觉分类模型单机多卡训练框架，仅需简单的几步即可开展模型对数据的拟合，快来试试吧😘

## 如何使用CHIBOOKS
### 1. 设定一个模型实例

|方法名称|说明|输入|返回值|是否自动调用|
|:---:|:---:|:---:|:---:|:---:|
|train_step|模型如何使用数据并进行损失计算|(batch, batch_idx)|loss|✅
|val_step|进行损失计算与评估指标计算|(batch, batch_idx)|loss, meta|✅
|test_step|在有标签的数据上进行评估|(batch, batch_idx)|loss, meta|
|inference_step|在用户输入的无标签数据上进行评估|(batch, batch_idx)|meta|

同时chibooks预定义了一些模型🤗，可以直接调用：
|模型名称|链接|预定义权重|状态|
|:---:|:---:|:---:|:---:|
|vit||dinov2|✅|
|tinyVit|||❓
|swin|||❓
|convnext|||❓


### 2. 设定待拟合的数据
chibooks中预定义了一些数据集模板，如果待训练的数据满足格式可以直接传参使用
|内置数据集格式|示例|输入设定|状态
|:---:|:---|:--:|:--:|
|ImageFolder|root_dir<br>-- CAT<br>---- cat01.jpg<br>---- cat02.jpg||❓|
|CSV_list|(clss_name, sample_path)||❓|

### 3. 配置训练参数

chibooks会按照设定开始训练,如果没有给定训练配置文件，则默认参数如下
|参数项目|默认设定|说明|
|:---:|:---:|:---:|
|optim|adamW||
|lr_scheduler|warmup_cosine_epoch|依epoch，先线性递增，到设定值后余弦衰减|
|pretrain_weight|True|将加载该模型的预训练权重，否则将正态初始|

### 4. 开训🎇

训练模型需要准备一个任务文件以及一个配置文件。在任务文件中需要继承taskbook来自定义模型训练行为。配置文件则用于调整超参数

