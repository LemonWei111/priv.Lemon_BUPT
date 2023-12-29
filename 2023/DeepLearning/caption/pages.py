# 服装图像描述生成系统初始注册登录页面
# Version: 1.0
# Author: [魏靖]

#使用tkinter库创建图形用户界面，实现用户登录和注册功能。
#连接了MySQL数据库，用于存储用户信息。

import tkinter as tk
import hashlib
import mysql.connector, tkinter.messagebox
from menu import get_in
from PIL import Image, ImageTk

# 连接MySQL数据库
cnx = mysql.connector.connect(
    user='caption',
    password='Caption1229!',
    host='localhost',
    database='caption'
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

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        ##登录界面
        self.title('欢迎进入服装图片描述生成系统！')
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
            username = name_tap.get()
            if login_user(name_tap.get(), code_tap.get()):
                ##print("登录成功")
                self.destroy()
                get_in(username)
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
        self.title('服装图片描述生成系统用户注册')
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
