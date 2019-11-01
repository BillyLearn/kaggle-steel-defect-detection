## classification 文件夹：训练Resnet34 模型作为分类器

我的pytorch GPU 相关运行配置：

> torch.\_\_version\_\_                             = 1.1.0
> torch.version.cuda             			 = 10.0.130
> torch.backends.cudnn.version()   = 7501
> os['CUDA_VISIBLE_DEVICES']     	= 0
> torch.cuda.device_count()      		= 1



训练之前，请确定:

1. 下载[数据集](<https://www.kaggle.com/c/severstal-steel-defect-detection/data>) , [resnet34 预训练模型](https://download.pytorch.org/models/resnet34-333f7ec4.pth)
2. 设置文件路径训练数据集，npy,  resnet34预训练模型，训练文件模型文件等。所有相关文件路径设置在 include.py

3. 制作训练二进制数据集 npy

   > python make_train_npy.py

4. 训练

   > python train.py

## segmentation文件夹: 训练 Efficientnet B5 作为分割器 

## input 文件夹: 数据集，预训练模型，模型保存等相关文件

## kaggle_ensemble_submit 文件夹: 生成csv 文件，提交kaggle 