# 图像可视化界面示例
# Version: 1.0
# Author: [魏靖]

#作为menu.py函数的示例，实际在本例实现中并未调用以下代码

import tkinter as tk
from tkinter import filedialog
import cv2, os, tkinter.messagebox
from PIL import Image, ImageTk
from tkinter import IntVar

# 打开图像文件并调整大小
def display_image(file_path):
    img_open = Image.open(file_path)
    (x, y) = img_open.size
    x_s = 200
    y_s = y * x_s // x
    img_adj = img_open.resize((x_s, y_s), Image.Resampling.LANCZOS)
    # 转换为Tkinter的PhotoImage格式
    img_png = ImageTk.PhotoImage(img_adj)
    # 在Label中显示图像
    Image2.configure(image=img_png)
    Image2.image = img_png  # Keep a reference to avoid garbage collection issues

# 弹出文件选择对话框，获取选择的图片文件路径
def upload_action():
    file_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    # 如果用户选择了文件，则进行相关处理
    if file_path:
        print("已选择文件:", file_path)
        # 在这里可以添加处理上传文件的代码，比如显示图片等操作
        display_image(file_path)

# 执行拍摄操作，将图像保存到指定路径
def capture_action():
    print("选择了拍摄")
    i = 1

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
    #if i ==1 :
        tkinter.messagebox.showerror("错误", "无法打开摄像头")

        return

    ret, img = cap.read()

    if not ret:
        tkinter.messagebox.showerror("错误", "无法从摄像头读取图像")
        return

    try:
        os.remove(os.path.join('./images/example.jpg'))
    except:
        print("写入数据")

    os.makedirs('./images/', exist_ok=True)
    cv2.imwrite(r"./images/example.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cap.release()

# 下拉菜单选项变化时的回调函数
def on_option_change(*args):
    selected_option = option_var.get()
    # 根据选择的操作执行相应的动作
    if selected_option == "上传":
        upload_action()
    elif selected_option == "拍摄":
        capture_action()

# 创建主窗口
root = tk.Tk()
root.title("选择操作")

# 创建下拉菜单选项
options = ["上传", "拍摄"]
option_var = tk.StringVar()
option_var.set(options[0])  # 默认选择第一个选项
option_var.trace_add("write", on_option_change)

# 创建下拉菜单
option_menu = tk.OptionMenu(root, option_var, *options)
#option_menu.pack(pady=20)
option_menu.grid(row=10, column=10, pady=20)

# 创建用于显示图像的Label
pic_tip = tk.Label(root, text="实时图像", width=16, height=2, font=("黑体", 12))
pic_tip.grid(column=1, row=2, sticky='s')

# Initialize with a default image path
default_image_path = r'.\images\start.jpg'
img_open = Image.open(default_image_path)
(x, y) = img_open.size
x_s = 200
y_s = y * x_s // x
img_adj = img_open.resize((x_s, y_s), Image.Resampling.LANCZOS)
img_png = ImageTk.PhotoImage(img_adj)

Image2 = tk.Label(root, bg='white', bd=20, height=int(y_s * 1.5), width=int(x_s * 0.75), image=img_png)
Image2.grid(column=1, row=4, sticky='w')

# 创建用于显示图像描述的Label
flag = IntVar()
flag.set(0)

def display_caption(caption):
    caption_label.config(text=f"Caption: {caption}")

# Example usage:
caption = "A beautiful image caption"

# Create a label to display the caption
caption_label = tk.Label(root, text="")
caption_label.grid(row=0, column=10, pady=20)

# Call the function to display the caption
display_caption(caption)

# 启动主循环
root.mainloop()
