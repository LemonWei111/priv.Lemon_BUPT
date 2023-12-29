报告文档：
cnnlagru_caption.ipynb//VGG19整体表示+GRU模型训练与功能验证报告
train.ipynb                   //ResNet101局部表示+LSTM+注意力机制模型训练报告
caption.ipynb              //ResNet101局部表示+LSTM+注意力机制模型功能验证报告

说明与其他附件：
--.ipynb_checkpoints   //JupterNotebook运行自动生成的.ipynb的检查点文件

--images
    --example.jpg         //系统实时拍摄存储图像
    --start.jpg               //系统主页面展示图像

#因平台上传文件大小限制，取消以下skip内内容上传
##########################skip###########################
--model                     //VGG19整体表示+GRU模型
    --MYMODL
        --last.ckpt         //最后一次训练的模型相关参数state = {
                                    //'epoch': epoch,
                                    //'step': i,
                                    //'model': model,
                                    //'optimizer': optimizer
                                //}
        --mymodl.pth   //系统使用的模型参数
##########################skip###########################

--api.py                    //翻译api

--checkpoint1_DeepFashion_MultiModal.pth.tar
                               //ResNet101局部表示+LSTM+注意力机制模型检查点

--datasets.py           //基于getdata.py做小变化的数据集加载

--evaluate.py           //实现bleu_4、rouge_l评测指标，实现序列转化为文本函数

--getdata.py            //数据集加载（训练+测试/训练+验证+测试），同时提供词典生成等数据预处理函数

--log.txt                  //VGG19整体表示+GRU模型最近一次训练记录

--menu.jpg             //系统初始图片

--menu.py              //系统主页面

--modelcr.py          //VGG19整体表示+GRU模型定义

--models.py           //ResNet101局部表示+LSTM+注意力机制模型定义

--pages.py             //系统初始注册登录页面

--solver.py             //ResNet101局部表示+LSTM+注意力机制模型的训练和验证函数

--upload.py           //图像可视化界面示例

--utils.py               //文本生成与可视化工具