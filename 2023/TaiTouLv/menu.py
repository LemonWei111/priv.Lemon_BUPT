import cv2, sys, os, glob, time, xlrd, xlrd2, datetime, hashlib
import mysql.connector, tkinter.messagebox

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from skimage import io
from PIL import Image, ImageTk
from tkinter import ttk
from tkinter import IntVar
from src.mtcnn import mtcnnimage

# 连接MySQL数据库
cnx = mysql.connector.connect(
    user='taitoulv',
    password='1',
    host='localhost',
    database='taitoulvuser'
)

# 创建用户表
def create_user_table():
    cursor = cnx.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL
        )
        """
    )
    cursor.close()

# 注册用户
def register_user(username, password):
    cursor = cnx.cursor()
    select_query = "SELECT username FROM users WHERE username = %s"
    cursor.execute(select_query, (username,))
    result = cursor.fetchone()
    if result:
        tkinter.messagebox.showinfo('错误提示','用户名已存在')
    else:
        #哈希处理密码
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        insert_query = "INSERT INTO users (username, password_hash) VALUES (%s, %s)"
        user_data = (username, password_hash)
        cursor.execute(insert_query, user_data)
        cnx.commit()
        tkinter.messagebox.showinfo('提示','用户注册成功')
    cursor.close()


# 登录验证
def login_user(username, password):
    cursor = cnx.cursor()
    # 获取密码哈希值
    select_query = "SELECT password_hash FROM users WHERE username = %s"
    cursor.execute(select_query, (username,))
    result = cursor.fetchone()
    if result:
        # 对输入的密码进行哈希处理，并与数据库中的哈希值进行比对
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash == result[0]:
            cursor.close()
            return True
    cursor.close()
    return False

# 创建用户表
create_user_table()

##主窗口
def get_in():
    # GUI代码

    #消除小弹窗
    root=tk.Tk()
    root.withdraw()
    
    window = tk.Toplevel()  # 这是一个窗口object
    window.title('抬头率监测系统')
    window.geometry('600x400')  # 窗口大小

    def read_data():
        path = r'.\py_excel.xlsx'

        # 打开文件
        data = xlrd.open_workbook(path)
        # path + '/' +file 是文件的完整路径
        # 获取表格数目
        # nums = len(data.sheets())
        # for i in range(nums):
        #     # 根据sheet顺序打开sheet
        #     sheet1 = data.sheets()[i]

        # 根据sheet名称获取
        sheet1 = data.sheet_by_name('Sheet1')
        sheet2 = data.sheet_by_name('Sheet2')
        # 获取sheet（工作表）行（row）、列（col）数
        nrows = sheet1.nrows  # 行
        ncols = sheet1.ncols  # 列
        # print(nrows, ncols)

        # 获取教室名称列表
        global room_name, time_name
        room_name = sheet2.col_values(0)
        time_name = sheet2.col_values(1)
        ##print(room_name)
        ##print(time_name)
        # 获取单元格数据
        # 1.cell（单元格）获取
        # cell_A1 = sheet2.cell(0, 0).value
        # print(cell_A1)
        # 2.使用行列索引
        # cell_A2 = sheet2.row(0)[1].value

    read_data()

    def gettime():  # 当前时间显示
        timestr = time.strftime('%Y.%m.%d %H:%M', time.localtime(time.time()))
        lb.configure(text=timestr)
        window.after(1000, gettime)

    lb = tk.Label(window, text='', font=("黑体", 20))
    lb.grid(column=0, row=0)
    gettime()

    # 选择教室标签加下拉菜单
    choose_classroom = tk.Label(window, text="选择地点", width=15, height=2, font=("黑体", 12)).grid(column=0, row=1,
                                                                                               sticky='w')
    class_room = tk.StringVar()
    class_room_chosen = ttk.Combobox(window, width=20, height=10, textvariable=class_room, state='readonly')
    class_room_chosen['values'] = room_name
    class_room_chosen.grid(column=0, row=1, sticky='e')

    # 选择课时标签加下拉菜单
    choose_time = tk.Label(window, text="选择时间", width=15, height=2, font=("黑体", 12)).grid(column=0, row=2, sticky='w')
    course_time = tk.StringVar()
    course_time_chosen = ttk.Combobox(window, width=20, height=10, textvariable=course_time, state='readonly')
    course_time_chosen['values'] = time_name
    course_time_chosen.grid(column=0, row=2, sticky='e')

    pic_tip = tk.Label(window, text="所选地点实时图像", width=16, height=2, font=("黑体", 12)).grid(column=1, row=2, sticky='s')

    img = r'.\faces\start.jpg'
    ##初始化所选教室实时图像
    img_open = Image.open(img)
    # 显示图片的代码
    (x, y) = img_open.size  # read image size
    x_s = 200  # define standard width
    y_s = y * x_s // x  # calc height based on standard width
    img_adj = img_open.resize((x_s, y_s), Image.Resampling.LANCZOS)
    img_png = ImageTk.PhotoImage(img_adj)

    Image2 = tk.Label(window, bg='white', bd=20, height=y_s * 0.83, width=x_s * 0.83,image=img_png)  ##0.83用来消除白框
    Image2.grid(column=1, row=4, sticky='w')

    flag = IntVar()
    flag.set(0)

    '''
        if(flag.get()!=0):
        pic_path = str(flag.get())+'.jpg'

        img_open = Image.open(img)
        # 显示图片的代码
        (x, y) = img_open.size  # read image size
        x_s = 200  # define standard width
        y_s = y * x_s // x  # calc height based on standard width
        img_adj = img_open.resize((x_s, y_s), Image.ANTIALIAS)
        img_png = ImageTk.PhotoImage(img_adj)
        Image2 = tk.Label(window, bg='black', bd=20, height=y_s * 0.83, width=x_s * 0.83, imagevariable=img_png)  ##0.83用来消除白框
        Image2.grid(column=1, row=4, sticky='w')
    '''
    def cameraget():
        ##x='Here'
        ##y='Now'
        ##if x=='Here':
        m=0
        ##控制更新本地当前时段数据
        if class_room_chosen.get()=='Here':
        ##获取摄像头图像并保存更新到数据库
            cap = cv2.VideoCapture(0)##0代表电脑摄像头，1代表外接摄像头(usb摄像头)
            ##控制参数来调用教室监控
            ##print(cap.isOpened())
            ret, img = cap.read()
            ##print(ret, img)
            ##cv2.imshow("Image", img)
            try:
                os.remove(os.path.join('./faces/',str(class_room_chosen.get()+course_time_chosen.get())+'.jpg'))
            except:
                ##print("写入数据")
                m=1
            ##os.remove(os.path.join('./faces/',str(x+y)+'.jpg'))
            ##更新本地实时状态（用于展示时模拟教室监控摄像头）
            cv2.imwrite(r".\faces\example.png",img,[cv2.IMWRITE_JPEG_QUALITY,100])
            src = os.path.join('./faces/example.png')
            dst = os.path.join('./faces/',str(class_room_chosen.get()+course_time_chosen.get()))
            ##dst = os.path.join('./faces/',str(x+y))
            try:
                os.rename(src, dst + '.jpg')
            except:
                ##print("更新数据失败")
                tkinter.messagebox.showinfo('错误提示','更新数据失败')
            ##cv2.waitKey(0)
            cap.release()# 释放摄像头资源
        ##暂时只实现本地测试

    def rate_cal():
        face = 0
        cameraget()

        def inspect():  
            nonlocal face
            str1 = "教室"
            str2 = "课上的抬头率为："
            path = r'.\faces'
            pic_path = str(class_room_chosen.get()) + str(course_time_chosen.get()) + '.jpg'
            p = path + '/' + pic_path
            '''
            img = cv2.imread(p)
            color = (0, 255, 0)

            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            classfier = cv2.CascadeClassifier(
                r".\haarcascade_frontalface_alt2.xml")
            faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            '''
            ##mtcnn人脸检测函数
            a = mtcnnimage(p)
            
            face = a
            str3 = str(a)
        inspect()
        path = r'.\py_excel.xlsx'
        data = xlrd.open_workbook(path)
        sheet1 = data.sheet_by_name('Sheet1')
        nrows = sheet1.nrows  # 行
        ncols = sheet1.ncols  # 列
        total = 0
        for i in range(nrows):
            if (sheet1.cell(i, 0).value == class_room_chosen.get() and sheet1.cell(i,
                                                                                   1).value == course_time_chosen.get()):
                total = sheet1.cell(i, 2).value
        ##print(total)
        global rate
        ##print(face)
        rate = face /total
        ##print(rate)        
        
        str1 = "教室"
        str2 = "课上的抬头率为："
        str3 = str(rate)
        var.set(class_room_chosen.get() + str1 + course_time.get() + str2 + str3)
        return rate

    ##控制展示变化图
    global arates
    arates=0
    
    # 连接MySQL数据库
    mydb = mysql.connector.connect(host="localhost",user="taitoulv",password="1",database="taitoulv")

    # 获取光标
    mycursor = mydb.cursor()

    # 创建数据表
    mycursor.execute("CREATE TABLE IF NOT EXISTS headpose (id INT AUTO_INCREMENT PRIMARY KEY, rate FLOAT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")

    # 定义一个函数来获取抬头率数据并将其存入数据库
    def save_data():
        # 计算抬头率
        ratee = rate_cal()

        # 插入抬头率数据到数据表中
        sql = "INSERT INTO headpose (rate) VALUES (%s)"
        val = (ratee,)
        mycursor.execute(sql, val)

        # 提交更改
        mydb.commit()

        # 打印插入的数据行数
        ##print(mycursor.rowcount, "record inserted.")

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

        # 绘制折线图
        plt.plot(timestamps, rates)
        plt.xlabel('Timestamp')
        plt.ylabel('Head Pose Rate')

        # 生成文件名和路径
        filename = f"{str(class_room_chosen.get()+course_time_chosen.get())}anlyse.jpg"
        filepath = f"./analyse/{filename}"

        # 保存散点图并覆盖旧的图像文件
        plt.savefig(filepath, bbox_inches='tight')
        global arates
        if arates==1:
            plt.show()
            arates=0
        plt.close()

    def analyse():
        # 循环执行save_data函数，并每隔1秒钟调用save_plot函数    
        save_data()
        if datetime.datetime.now().minute % 1 == 0:
            save_plot()
        time.sleep(1)

    def pic_re():
        cameraget()
        if (flag.get() == 0):
            pic_path = str(class_room_chosen.get()) + str(course_time_chosen.get()) + '.jpg'
            img = os.path.join(r'.\faces', pic_path) #图片的命名需按规则来命名，具体规则可参考示例图片名称
            img_open = Image.open(img)
            # 显示图片的代码
            (x, y) = img_open.size  # read image size
            global x_s
            global y_s
            x_s = 200  # define standard width
            y_s = y * x_s // x  # calc height based on standard width
            img_adj = img_open.resize((x_s, y_s), Image.Resampling.LANCZOS)
            global img_png  ##这里一定要设置为全局变量，不然图片无法正常显示！！！！！！！！！！！
            img_png = ImageTk.PhotoImage(img_adj)
            Image2.configure(image=img_png)
        window.update_idletasks()
        ##受系统性能限制，后台不能一直运行记录数据
        analyse()
                
    def showanalyse():
        global arates
        arates=1
        analyse()
        
    var = tk.StringVar()  # tkinter中的字符串
    display = tk.Label(window, textvariable=var, font=('Arial', 12), width=38, height=10)
    display.grid(column=0, row=4, sticky='n')

    # Adding a Button
    rate_button = ttk.Button(window, text="查看当前抬头率", command=rate_cal).grid(column=0, row=4, sticky='s')
    pic_button = ttk.Button(window, text="查看实时图像并记录", command=pic_re).grid(column=0, row=5)
    ana_button=ttk.Button(window, text="查看近三小时抬头率变化图", command=showanalyse).grid(column=0, row=6)
    window.mainloop()


class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        ##登录界面
        self.title('欢迎进入抬头率检测系统！')
        self.geometry('600x420')
        #增加背景图片
        img = Image.open(r".\menu.jpg")
        img2 = img.resize((600, 420), )
        photo = ImageTk.PhotoImage(img2)
        theLabel = tk.Label(self,
                 text="",#内容
                 justify=tk.LEFT,#对齐方式
                 image=photo,#加入图片
                compound = tk.CENTER,#关键:设置为背景图片
                font=("华文新魏",20),#字体和字号
                fg = "white")#前景色
        theLabel.place(x=0,y=0)

        name = tk.Label(self, text="请输入用户名:", width=16, height=1)
        name.place(x=50, y=220)
        name_tap = tk.Entry(self,  width=16)
        name_tap.place(x=250, y=220)
        ##name_tap.pack()

        code = tk.Label(self, text="请输入密码:", width=16, height=1)
        code.place(x=50, y=250)
        code_tap = tk.Entry(self,  width=16,show="*")
        code_tap.place(x=250, y=250)
        ##code_tap.pack()

        def login():
            if login_user(name_tap.get(), code_tap.get()):
                ##print("登录成功")
                self.destroy()
                get_in()
            else:
                tkinter.messagebox.showinfo('错误提示','用户名或密码错误')
        
        self.button1 = tk.Button(self, text='注册', command=self.open_subwindow).place(x=330,y=300)
        ##self.button1.pack()
        self.button2 = tk.Button(self, text='登录', command=login).place(x=250,y=300)
        ##self.button2.pack()

        self.subwindow = None  # 用于存储子界面的引用

    def open_subwindow(self):
        if self.subwindow is None:  # 如果子界面尚未打开
            self.withdraw()  # 隐藏主界面
            self.subwindow = SubWindow(self)  # 创建子界面
            self.subwindow.protocol("WM_DELETE_WINDOW", self.on_subwindow_close)  # 监听子界面的关闭事件

    def on_subwindow_close(self):
        self.subwindow.destroy()  # 关闭子界面
        self.subwindow = None  # 清空子界面引用
        self.deiconify()  # 重新显示主界面

        

class SubWindow(tk.Toplevel):
    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.title('抬头率监测系统用户注册')
        self.geometry('600x400')  # 窗口大小
        name = tk.Label(self, text="请输入用户名:", width=16, height=1)
        name.place(x=50, y=220)
        name_tap = tk.Entry(self,  width=16)
        name_tap.place(x=250, y=220)
        ##name_tap.pack()
    
        code = tk.Label(self, text="请输入密码:", width=16, height=1)
        code.place(x=50, y=250)
        code_tap = tk.Entry(self,  width=16,show="*")
        code_tap.place(x=250, y=250)
        ##code_tap.pack()
        codee = tk.Label(self, text="再次输入密码:", width=16, height=1)
        codee.place(x=50, y=280)
        codee_tap = tk.Entry(self,  width=16,show="*")
        codee_tap.place(x=250, y=280)
        ##codee_tap.pack()
        def register():
            if(code_tap.get()==codee_tap.get()):
                # 注册用户
                register_user(name_tap.get(), code_tap.get())
            else:
                tkinter.messagebox.showinfo('错误提示','两次密码不一致，请重新输入')
        get_up_done = tk.Button(self, text='提交', command=register).place(x=250,y=330)
        ##get_up_done.pack()
        self.button = tk.Button(self, text="返回登录", command=self.close).place(x=310,y=330)
        ##self.button.pack()

    def close(self):
        self.master.on_subwindow_close()  # 调用主界面的方法来关闭子界面并返回主界面
        self.destroy()  # 关闭当前窗口

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()

