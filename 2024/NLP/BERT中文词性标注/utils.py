# 监控与绘制
import matplotlib.pyplot as plt

def show(log, log_name='loss'):
    '''
    绘制并保存模型日志数据变化图像

    :param log: list 你的日志数据列表
    :param log_name: str 数据名
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(log)  # 直接使用列表进行绘图
    plt.title(f'Log of {log_name}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{log_name}')
    plt.grid(True)
    plt.show()
    plt.savefig(f'log_{log_name}.png')

'''
## Use ##
loss = [1, 2, 4, 3, 5]
show(loss)
'''
