import mysql.connector
import hashlib
'''
import cv2
import numpy as np
import sys, os, glob, numpy
from skimage import io
from PIL import Image, ImageTk
import tkinter as tk
import time
from tkinter import ttk
from tkinter import IntVar
import xlrd
import xlrd2
import matplotlib.pyplot as plt
import mysql.connector
import datetime
from src.mtcnn import mtcnnimage
##消除tk小框
root = tk.Tk()
root.withdraw()
##登录界面
root = tk.Toplevel()
root.title('欢迎进入抬头率检测系统！')
root.geometry('600x420')
#增加背景图片
img = Image.open(r".\menu.jpg")
img2 = img.resize((600, 420), )
photo = ImageTk.PhotoImage(img2)
theLabel = tk.Label(root,
                 text="",#内容
                 justify=tk.LEFT,#对齐方式
                 image=photo,#加入图片
                compound = tk.CENTER,#关键:设置为背景图片
                font=("华文新魏",20),#字体和字号
                fg = "white")#前景色
theLabel.place(x=0,y=0)

name = tk.Label(root, text="请输入用户名:", width=16, height=1)
name.place(x=50, y=220)
name_tap = tk.Entry(root,  width=16)
name_tap.place(x=250, y=220)
name_tap.pack()

code = tk.Label(root, text="请输入密码:", width=16, height=1)
code.place(x=50, y=250)
code_tap = tk.Entry(root,  width=16,show="*")
code_tap.place(x=250, y=250)
code_tap.pack()

def showuser():
    print(name_tap.get())
    print(code_tap.get())
get_into = ttk.Button(root, text='显示', command=showuser).place(x=250,y=300)
'''
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
        print("用户名已存在")
    else:
        #哈希处理密码
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        insert_query = "INSERT INTO users (username, password_hash) VALUES (%s, %s)"
        user_data = (username, password_hash)
        cursor.execute(insert_query, user_data)
        cnx.commit()
        print("用户注册成功")
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

def register_userprocessing():
    root.withdraw()
    window1 = tk.Toplevel()  # 这是一个窗口object
    window1.title('抬头率监测系统用户注册')
    window1.geometry('600x400')  # 窗口大小
    name = tk.Label(window1, text="请输入用户名:", width=16, height=1)
    name.place(x=50, y=220)
    name_tap = tk.Entry(window1,  width=16)
    name_tap.place(x=250, y=220)
    name_tap.pack()
    
    code = tk.Label(window1, text="请输入密码:", width=16, height=1)
    code.place(x=50, y=250)
    code_tap = tk.Entry(window1,  width=16,show="*")
    code_tap.place(x=250, y=250)
    code_tap.pack()
    codee = tk.Label(window1, text="再次输入密码:", width=16, height=1)
    codee.place(x=50, y=280)
    codee_tap = tk.Entry(window1,  width=16,show="*")
    codee_tap.place(x=250, y=280)
    codee_tap.pack()
    def register():
        if(code_tap.get()==codee_tap.get()):
            # 注册用户
            register_user(name_tap.get(), code_tap.get())
        else:
            print("两次密码不一致，请重新输入")
    get_up_done = ttk.Button(window1, text='提交', command=register).place(x=250,y=330)
    window1.mainloop()
# 登录验证
if login_user('Alice', 'password123'):
    print("登录成功")
    get_in()
else:
    print("用户名或密码错误")
