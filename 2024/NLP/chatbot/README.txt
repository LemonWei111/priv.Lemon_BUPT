# python==3.10.5
运行步骤：
1、配置环境，设置环境变量
2、python app.py

--Message_example    系统测试对话记录文本
   |--message（机密窃取意图）.txt
   |--message（人工服务唤醒）.txt
   |--message（商家诉求）.txt
   |--message（推荐与下单）.txt
   |--message（退款与补偿）.txt
   |--message（外卖员诉求）.txt
   |--message（违法行为意图）.txt
   |--message（未定义行为）.txt
--spa
   |--AmbiguousCustomerSentences_Reformatted.csv    ①情感极性分析-测试数据（50）
   |--classifiers.py                                                           ①情感极性分析-模型定义
   |--feature_extraction.py                                             ①情感极性分析-数据特征提取
   |--svm_model.pkl                                                      ①情感极性分析-模型
   |--used_for_choose.zip                                             ①情感极性分析-方法对比和选择
   |--waimai_10k.csv                                                   ①情感极性分析-测试数据（10000）
--api.py                                  ②大模型-阿里云模型api调用
--app.py                                 系统应用APP
--MaShanZheng-Regular.ttf    ③关键词提取与可视化-词云字体文件
--metric.py                              ②大模型-评测指标
--my_wordcloud.png               ③关键词提取与可视化-词云示例
--OFL.txt                                 ③关键词提取与可视化-词云字体说明
--pass.txt                                 系统环境配置密钥、APP管理员密码
--prompt0608.txt                     ②大模型-最新系统提示词
--README.txt
--report.txt                              ③关键词提取与可视化-服务报告示例
--requirements.txt
--result.xls                               系统测试结果
--stopwords.txt                        ③关键词提取与可视化-停用词
--test_llm.py                            ②大模型-测试
--test_svm.py                          ①情感极性分析-测试
--Testdata.xlsx                       ②大模型-测试数据
--train_svm.py                      ①情感极性性分析-模型训练
--trywordcloud.py              ③关键词提取与可视化