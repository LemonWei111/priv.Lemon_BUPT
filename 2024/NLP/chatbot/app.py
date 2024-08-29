import os
import gradio as gr

from api import new_call_with_messages, get_report, get_score

import time
st = time.time()

prompt_path = 'prompt0608.txt'
# 打开文本：指定使用utf-8编码读取
with open(prompt_path, 'r', encoding='utf-8') as f:
    prompt = f.read()
    
messages = [{'role': 'system', 'content': prompt}]

from spa.classifiers import SVMClassifier
svm = SVMClassifier()

def validate_password(password_input):
    '''
    验证输入的密码是否与预设的管理员密码匹配。
    
    :param password_input (str): 用户输入的密码字符串。
    
    :return bool: 如果密码匹配则返回True，否则返回False。
    '''
    # 这里只是一个示例验证逻辑，实际应替换为您的密码验证机制
    correct_password = "bupt2024nlp_wwx"  # 示例密码，请替换为真实的管理员密码
    if password_input == correct_password:
        return True
    else:
        return False

def chatbot_response(history, user_input):
    '''
    处理用户输入并生成聊天机器人的响应。
    
    :param history (List[List[str]]): 聊天记录，每个子列表包含一条用户消息和对应的机器人回复。
    :param user_input (str): 用户的当前输入。
    
    :return Tuple[List[List[str]], str]: 更新后的聊天记录（包含新对话）以及清空后的输入框内容。
    '''
    # history是之前的聊天记录，user_input是当前输入
    # 假设history是一个列表的列表，每个子列表包含两条消息：[用户消息, 机器人回复]
    if user_input.strip():
        score = get_score(user_input)
        global messages
        messages.append({'role': 'user', 'content': f'{score}'+' '+user_input})
        messages, response = new_call_with_messages(messages)
        
        # response = f"你刚才说了：{user_input}"
        history.append([user_input, response]) # 添加新的对话到历史记录
    return history, "" # 返回更新后的聊天记录，并清空输入框内容

def chat_app():
    '''
    创建并启动一个基于Gradio的聊天应用界面。
    
    功能包括：
    - 用户与聊天机器人的交互界面。
    - 管理员密码验证功能，用于生成服务报告、查看服务报告文本和历史服务情况图像。
    - 可视化展示聊天历史、生成的服务报告和基于历史对话的服务情况图像。
    '''
    
    demo = gr.Blocks()
    with demo:
        gr.Markdown("# 饿了没智能客服")

        chat_history = gr.Chatbot([], label="聊天记录") # 初始化为空列表的列表

        msg = gr.Textbox(label="输入消息", placeholder="我是您的智能客服，很高兴为您服务...") # , submit_event="send")
        with gr.Row():
            send = gr.Button("发送")
            clear = gr.Button("清除聊天记录")

        send.click(chatbot_response, inputs=[chat_history, msg], outputs=[chat_history, msg], queue=False)
        msg.submit(chatbot_response, inputs=[chat_history, msg], outputs=[chat_history, msg], queue=False)

        clear.click(lambda _: [], None, chat_history) # 清空聊天记录的回调函数

        with gr.Row():
            password = gr.Textbox(label="管理员密码", type="password")
            with gr.Row():
                generate_report = gr.Button("生成服务报告")
                show_txt_button = gr.Button("查看服务报告")
                show_image_button = gr.Button("查看历史服务情况")

        image_output = gr.Image(type="filepath", visible=False) # 初始化图片组件为隐藏状态
        txt_output = gr.Textbox(label="服务报告", lines=10, interactive=False, visible=False) # 初始化文本框组件用于显示内容
       
        # 定义处理密码验证及报告生成的函数
        def validate_and_report(password_input):
            is_valid = validate_password(password_input)
            if is_valid:
                try:
                    get_report(messages)
                except:
                    return "", "抱歉， 当前没有服务记录。"
                return "", "报告已生成。"
            else:
                return "", "抱歉， 您不能进行权限外的操作。"


        # 定义处理密码验证及历史图片查看的函数
        def display_local_image(password_input):
            is_valid = validate_password(password_input)
            if is_valid:
                # 当前执行目录
                rootdir = os.getcwd()
                print(rootdir)
                pic_path = os.path.join(rootdir, "my_wordcloud.png")
                print(pic_path)
                if not os.path.exists(pic_path):
                    try:
                        get_report(messages)
                    except:
                        return "", gr.update(visible=False), "抱歉， 当前没有服务记录。"
                return "", gr.update(value=pic_path, visible=True), ""
            else:
                return "", gr.update(visible=False), "抱歉， 您不能进行权限外的操作。"
        
        # 定义处理密码验证及历史服务报告
        def read_specified_txt_file(password_input):
            is_valid = validate_password(password_input)
            if is_valid:
                # 当前执行目录
                rootdir = os.getcwd()
                print(rootdir)
                file_path = os.path.join(rootdir, "report.txt")
                if not os.path.exists(file_path):
                    return "", gr.update(visible=False), "抱歉， 当前没有服务报告，请先生成。"

                from trywordcloud import read_text
                content = read_text(file_path)
                return "", gr.update(value=content, visible=True), ""
            else:
                return "", gr.update(visible=False), "抱歉， 您不能进行权限外的操作。"

        generate_report.click(validate_and_report, inputs=[password], outputs=[password, msg], queue=False)
        show_image_button.click(display_local_image, inputs=[password], outputs=[password, image_output, msg], queue=False)
        show_txt_button.click(read_specified_txt_file, inputs=[password], outputs=[password, txt_output, msg], queue=False)

    print(time.time()-st)
    demo.launch(inbrowser=True)
    
if __name__ == "__main__":
    chat_app()

    #get_report(messages)
    
