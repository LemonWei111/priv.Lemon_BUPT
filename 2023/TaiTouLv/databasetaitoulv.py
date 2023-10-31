import time
import mysql.connector
import matplotlib.pyplot as plt
import datetime

# 连接MySQL数据库
mydb = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  password="yourpassword",
  database="yourdatabase"
)

# 获取光标
mycursor = mydb.cursor()

# 创建数据表
mycursor.execute("CREATE TABLE IF NOT EXISTS headpose (id INT AUTO_INCREMENT PRIMARY KEY, rate FLOAT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

# 定义一个函数来获取抬头率数据并将其存入数据库
def save_data():
    # 计算抬头率
    rate = rate_cal()

    # 插入抬头率数据到数据表中
    sql = "INSERT INTO headpose (rate) VALUES (%s)"
    val = (rate,)
    mycursor.execute(sql, val)

    # 提交更改
    mydb.commit()

    # 打印插入的数据行数
    print(mycursor.rowcount, "record inserted.")

# 定义一个函数来生成并保存散点图
def save_plot():
    # 获取当前时间
    now = datetime.datetime.now()

    # 获取最近3小时内的数据
    time_limit = now - datetime.timedelta(hours=3)
    time_limit_str = time_limit.strftime('%Y-%m-%d %H:%M:%S')
    query = f"SELECT * FROM headpose WHERE timestamp > '{time_limit_str}'"
    mycursor.execute(query)

    # 获取所有记录
    records = mycursor.fetchall()

    # 分离时间戳和抬头率
    timestamps = [record[2] for record in records]
    rates = [record[1] for record in records]

    # 绘制散点图
    plt.scatter(timestamps, rates)
    plt.xlabel('Timestamp')
    plt.ylabel('Head Pose Rate')

    # 生成文件名和路径
    filename = f"{str(class_room_chosen.get()+course_time_chosen.get())}anlyse.jpg"
    filepath = f"./analyse/{filename}"

    # 保存散点图并覆盖旧的图像文件
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    if arates==1:
        plt.show()
    arates=0
    

# 循环执行save_data函数，并每隔1分钟调用save_plot函数
while True:
    save_data()
    time.sleep(60)
    if datetime.datetime.now().minute % 1 == 0:
        save_plot()
