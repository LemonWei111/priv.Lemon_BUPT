# 服装图像描述生成系统主页面
# Version: 1.0
# Author: [魏靖]

# 本代码是一个基于 tkinter 的服装图像描述生成系统的 GUI 程序。
# 主要功能包括上传或拍摄服装图像，通过预训练模型生成图像描述，并提供保存描述到数据库的功能。
#     图形用户界面：创建了一个图形用户界面，其中包括显示实时图像的部分，可以通过上传图片文件或拍摄实时图像。
#     实时图像显示：根据用户选择的上传或拍摄，显示相应的实时图像。
#     模型生成图片描述：通过预训练的深度学习模型（在./model/MYMODL/mymodl.pth中），对输入的图像生成描述，并显示在界面上。
#     保存到数据库：用户可以将生成的图片描述保存到MySQL数据库中，包括用户名、描述内容和图片数据。
#     查看历史记录：用户可以查看之前保存的图片描述历史记录，最多显示最近10条记录。 

# 导入必要的库
import mysql.connector, tkinter.messagebox
import cv2, sys, os, glob, time, xlrd, xlrd2, torch

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import ttk
from tkinter import IntVar
from tkinter import filedialog
from skimage import io
from PIL import Image, ImageTk
from io import BytesIO
from modelcr import *
from torchvision import transforms
from api import translate
from evaluate import fromtexttosentence
from getdata import words

##主窗口
def get_in(username):
    # GUI代码

    #消除小弹窗
    root=tk.Tk()
    root.withdraw()
    
    window = tk.Toplevel()  # 这是一个窗口object
    window.title('服装图像描述生成系统')
    window.geometry('600x400')  # 窗口大小

    # Initialize with a default image path
    global default_image_path
    default_image_path = r'.\images\start.jpg'
    global default_caption
    default_caption = "将在此处为您生成图片描述"
    
    #用户上传图片文件（有选择上传或拍摄两个选项）
    def display_image(file_path):
        img_open = Image.open(file_path)
        (x, y) = img_open.size
        x_s = 200
        y_s = y * x_s // x
        img_adj = img_open.resize((x_s, y_s), Image.Resampling.LANCZOS)
        img_png = ImageTk.PhotoImage(img_adj)
        Image2.configure(image=img_png)
        Image2.image = img_png  # Keep a reference to avoid garbage collection issues

    def upload_action():
        file_path = filedialog.askopenfilename(title="选择图片文件", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            print("已选择文件:", file_path)
            # 在这里可以添加处理上传文件的代码，比如显示图片等操作
            display_image(file_path)
            global default_image_path
            default_image_path = file_path

    def capture_action():
        print("选择了拍摄")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            tkinter.messagebox.showerror("错误", "无法打开摄像头")
            return
        #cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)    # 创建一个用于显示画面的窗口
        #cv2.imshow(window_name, cap.read()[1])
        #暂停10秒
        time.sleep(10)
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
        display_image("./images/example.jpg")
        global default_image_path
        default_image_path = r'.\images\example.jpg'

    def on_option_change(*args):
        selected_option = option_var.get()
        if selected_option == "上传":
            upload_action()
        elif selected_option == "拍摄":
            capture_action()
            
    def gettime():  # 当前时间显示
        timestr = time.strftime('%Y.%m.%d %H:%M', time.localtime(time.time()))
        lb.configure(text=timestr)
        window.after(1000, gettime)

    lb = tk.Label(window, text='', font=("黑体", 20))
    lb.grid(column=0, row=1)
    gettime()

    # 创建下拉菜单选项
    options = ["上传", "拍摄"]
    option_var = tk.StringVar()
    option_var.set(options[0])  # 默认选择第一个选项
    option_var.trace_add("write", on_option_change)

    # 创建下拉菜单
    option_menu = tk.OptionMenu(window, option_var, *options)
    option_menu.grid(row=10, column=10, pady=20)
    
    pic_tip = tk.Label(window, text="实时图像", width=16, height=2, font=("黑体", 12))
    pic_tip.grid(column=1, row=2, sticky='s')

    img_open = Image.open(default_image_path)
    (x, y) = img_open.size
    x_s = 200
    y_s = y * x_s // x
    img_adj = img_open.resize((x_s, y_s), Image.Resampling.LANCZOS)
    img_png = ImageTk.PhotoImage(img_adj)

    Image2 = tk.Label(window, bg='white', bd=20, height=int(y_s * 1.5), width=int(x_s * 0.75), image=img_png)
    Image2.grid(column=1, row=4, sticky='w')

    flag = IntVar()
    flag.set(0)
    
    # Example usage:
    # Create a label to display the caption
    caption_label = tk.Label(window, text="")
    caption_label.grid(row=0, column=10, pady=20)

    # Call the function to display the caption
    caption_label.config(text=f"描述: {default_caption}")
    train_path = './deepfashion-multimodal/train_captions.json'
    all_words = words(train_path)

    def get_image(file_path):
        # 读取图像
        image = Image.open(file_path)
        image = image.convert('RGB')
        tx = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        #try2
        if tx is not None:
            image = tx(image).unsqueeze(0)
        return image

    model = torch.load('./model/MYMODL/mymodl.pth')  # Replace with the actual path to your model.pth
    model.eval()

    def caption_cal():
        try:
            # Load the model
            beam_k = 5
            max_lenth = 50

            image = get_image(default_image_path)

            # Perform forward pass
            with torch.no_grad():
                output = model.generate_by_beamsearch(image, beam_k, max_lenth+2)

            # Get the caption
            caption = translate(fromtexttosentence(output, all_words))

            # Do something with the caption, such as printing it
            caption_label.config(text=f"Caption: {caption}")
            global default_caption
            default_caption = caption
            print(default_caption)

        except Exception as e:
            print("Error:", e)

    def caption_save():
        try:
            # Connect to the MySQL database
            mydb = mysql.connector.connect(
                host="localhost",
                user="caption",
                password="Caption1229!",
                database="caption"
            )
            cursor = mydb.cursor()

            # 创建数据表
            cursor.execute("CREATE TABLE IF NOT EXISTS caption (id INT AUTO_INCREMENT PRIMARY KEY,username VARCHAR(255) NOT NULL,caption VARCHAR(255) NOT NULL,image_blob LONGBLOB NOT NULL)")

            # Read the image as bytes
            with open(default_image_path, 'rb') as f:
                image_blob = f.read()
            '''
            with open(default_image_path, "rb") as image_file:
                # Read the binary data of the image file
                binary_data = image_file.read()
                # Encode the binary data using base64
                encoded_data = base64.b64encode(binary_data)
                # Convert the encoded data to a string (optional)
                image_blob = encoded_data.decode('utf-8')
            '''
            # Insert data into the 'caption' table
            query = "INSERT INTO caption (username, caption, image_blob) VALUES (%s, %s, %s)"

            values = (username, default_caption, image_blob)
            print(default_caption)
            cursor.execute(query, values)

            # Commit the changes
            mydb.commit()

            print("Data saved successfully!")

        except Exception as e:
            print("Error:", e)

        finally:
            # Close the connection
            if mydb:
                mydb.close()

    def get_captions():
        try:
            # Connect to the MySQL database
            mydb = mysql.connector.connect(
                host="localhost",
                user="caption",
                password="Caption1229!",
                database="caption"
            )
            cursor = mydb.cursor()

            # Retrieve up to ten captions (ordered by the latest first)
            query = f"SELECT username, caption, image_blob FROM caption WHERE username = '{username}' ORDER BY id DESC LIMIT 10"
            cursor.execute(query)
            result = cursor.fetchall()

            return result

        except Exception as e:
            print("Error:", e)
            return []

        finally:
            # Close the connection
            if mydb:
                mydb.close()

    def display_caption_and_image(caption_data):
        if not caption_data:
            return

        username, caption_text, image_blob = caption_data

        # Display the caption
        caption_label.config(text=f"Username: {username}\nCaption: {caption_text}")

        # Display the image
        image = Image.open(BytesIO(image_blob))
        (x, y) = image.size
        x_s = 200
        y_s = y * x_s // x
        img_adj = image.resize((x_s, y_s), Image.Resampling.LANCZOS)
        img_png = ImageTk.PhotoImage(img_adj)
        Image2.configure(image=img_png)
        Image2.image = img_png  # Keep a reference to avoid garbage collection issues

    def show_record():
        captions = get_captions()

        def on_caption_change(*args):
            selected_caption = caption_var.get()#tkinter.messagebox.askoption("选择一条记录", "选择记录并查看:", options=caption_options)

            if selected_caption:
                index = int(selected_caption.split(".")[0]) - 1
                selected_caption_data = captions[index]
                display_caption_and_image(selected_caption_data)

        # Display up to ten captions in messagebox
        caption_options = [f"{i+1}. {caption[1]}" for i, caption in enumerate(captions)]

        caption_var = tk.StringVar()
        caption_var.set(caption_options[0])  # 默认选择第一个选项
        caption_var.trace_add("write", on_caption_change)

        # 创建下拉菜单
        caption_menu = tk.OptionMenu(window, caption_var, *caption_options)
        caption_menu.grid(row=4, column=10, pady=20)
        '''
        selected_caption = caption_var.get()#tkinter.messagebox.askoption("选择一条记录", "选择记录并查看:", options=caption_options)

        if selected_caption:
            index = int(selected_caption.split(".")[0]) - 1
            selected_caption_data = captions[index]
            display_caption_and_image(selected_caption_data)
        '''
    # Adding a Button
    rate_button = ttk.Button(window, text="生成图片描述", command=caption_cal).grid(column=2, row=3, sticky='s')
    pic_button = ttk.Button(window, text="保存到我的数据库", command=caption_save).grid(column=2, row=4)
    ana_button=ttk.Button(window, text="查看我保存的历史记录", command=show_record).grid(column=2, row=5)
    window.mainloop()

#get_in("Lemon")
