# CLIP 模型层次化训练与推理说明<br>
1. CLIP 模型训练时，如果使用精细的损失，那么需要提供标签，用于区分batch中哪些样本属于同一类别
如果一个类别有多个层次标签需要训练，那么需要建立映射关系,建议直接建立如下形式的映射表，这样在推理时同样可以用到。<br>
    ```json
    {
        "class_A":{
            "global_idx":[1,1,3,4,12]
            "inlayer_idx":[1,1,1,2,3]
            "layer_cls":[
                "Aves",
                "Galliformes",
                "Numididae",
                "Acryllium",
                "Acryllium vulturinum"
            ]
        }
    }
    ```
    此时可以通过 dataloader 提供的的类别名来检索需要的各层文本及其全局索引值，用于clip训练

2. CLIP模型推理时有两种方法，分别是直接使用某一次的文本进行直接推理，以及从根节点逐层逻辑推理两种形式。
    - 直接推理模型在特定层级的精度<br>
        这种模式下，需要使用上面json中inlayer_idx作为样本真值，
        而输入CLIP的文本则来自另一个json文件，该json文件中记录了每层中都包含哪些类别，基本形式如下：<br>
        ```json
        "layer_class": [
            "Aves",
            "Mammalia"
        ],
        "layer_order": [
            "Artiodactyla",
            "Carnivora",
            "Cingulata",
            "Columbiformes",
            "Didelphimorphia",              
        ], 
        ```
        其中每个layer内的列表的顺序就是各层级内inlayer的顺序。<br>
        这样，在推理时，根绝输入batch可以获取图像image 以及 meta 类别名，通过类别名可以获取该类别对应的层次类别，CLIP模型的输入为 image 以及指定层次所有的类别对应的文本
        ```python
        prefix = "A photo of an animal, which belongs to "
        text = [f"{prefix}{x}" for x in layer_order ]
        target = layer_info[class_name]["inlayer_idx"][layer]
        pred = ...

        ```
        这样通过 `softmax(pred)` 计算得到的索引就可以得到预测的类别结果，因为两者的索引是对齐的

    - 层次化逻辑推理<br>
    这种模式下希望通过逐层推理来做出分类决策。

    实际上，上述两种测试形式只是对相似度结果的不同处理方式，两者都可以先将图像与所有层次的文本计算相似度，然后再处理处理结果，
    