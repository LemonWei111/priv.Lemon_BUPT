import tkinter as tk

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("主界面")

        self.button1 = ttk.Button(self, text='注册', command=self.open_subwindow).place(x=50,y=300)
        self.button1.pack()

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
        name_tap.pack()
    
        code = tk.Label(self, text="请输入密码:", width=16, height=1)
        code.place(x=50, y=250)
        code_tap = tk.Entry(self,  width=16,show="*")
        code_tap.place(x=250, y=250)
        code_tap.pack()
        codee = tk.Label(self, text="再次输入密码:", width=16, height=1)
        codee.place(x=50, y=280)
        codee_tap = tk.Entry(self,  width=16,show="*")
        codee_tap.place(x=250, y=280)
        codee_tap.pack()
        def register():
            if(code_tap.get()==codee_tap.get()):
                # 注册用户
                register_user(name_tap.get(), code_tap.get())
            else:
                print("两次密码不一致，请重新输入")
        get_up_done = ttk.Button(self, text='提交', command=register).place(x=250,y=330)
        get_up_done.pack()
        self.button = tk.Button(self, text="返回登录", command=self.close).place(x=250,y=380)
        self.button.pack()

    def close(self):
        self.master.on_subwindow_close()  # 调用主界面的方法来关闭子界面并返回主界面
        self.destroy()  # 关闭当前窗口

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
