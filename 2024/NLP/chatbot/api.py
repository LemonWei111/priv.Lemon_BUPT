# python
# 业务空间模型调用请参考文档传入workspace信息: https://help.aliyun.com/document_detail/2746874.html    
        
# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
import re
import time
import jieba
import logging

from http import HTTPStatus
from dashscope import Generation
from retrying import retry  # 需要安装retrying库

# 配置日志
logging.basicConfig(level=logging.INFO)

# 'llama3-8b-instruct'
# 'llama2-7b-chat-v2' 可训练，但部署要钱
# 'qwen1.5-1.8b-chat'
def call_with_messages(user_input, model_name='llama3-8b-instruct', use_prompt=True):
    '''
    messages：用户与模型的对话历史。array中的每个元素形式为{"role":角色, "content": 内容}，角色当前可选值：system、user、assistant和tool。
    
    system：表示系统级消息，用于指导模型按照预设的规范、角色或情境进行回应。是否使用system角色是可选的，如果使用则必须位于messages的最开始部分。
    user和assistant：表示用户和模型的消息。它们应交替出现在对话中，模拟实际对话流程。
    tool：表示工具的消息。在使用function call功能时，如果要传入工具的结果，需将元素的形式设为{"content":"工具返回的结果", "name":"工具的函数名", "role":"tool"}。其中name是工具函数的名称，需要和上轮response中的tool_calls[i]['function']['name']参数保持一致；content是工具函数的输出。
    '''
    prompt = " "
    if use_prompt:
        prompt_path = 'prompt0608.txt'
        # 打开文本：指定使用utf-8编码读取
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
    
    messages = [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_input}]

    response = Generation.call(
        model=model_name,
        messages=messages,
        result_format='message'
    )
    if response.status_code == HTTPStatus.OK:
        # print(response)
        return response.output.choices[0].message['content']
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        return "亲亲，抱歉，服务器连接错误，请您稍后再试！"

def multi_round():
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': '如何做西红柿炖牛腩？'}]
    response = Generation.call(model='qwen1.5-1.8b-chat',
                               messages=messages,
                               # 将输出设置为"message"格式
                               result_format='message')
    if response.status_code == HTTPStatus.OK:
        print(response)
        # 将assistant的回复添加到messages列表中
        messages.append({'role': response.output.choices[0]['message']['role'],
                         'content': response.output.choices[0]['message']['content']})
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        # 如果响应失败，将最后一条user message从messages列表里删除，确保user/assistant消息交替出现
        messages = messages[:-1]
    # 将新一轮的user问题添加到messages列表中
    messages.append({'role': 'user', 'content': '不放糖可以吗？'})
    # 进行第二轮模型的响应
    response = Generation.call(model='qwen1.5-1.8b-chat',
                               messages=messages,
                               result_format='message',  # 将输出设置为"message"格式
                               )
    if response.status_code == HTTPStatus.OK:
        print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))

# Function call的使用涉及到参数解析功能，因此对大模型的响应质量要求较高
from datetime import datetime
import random
import json

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 获取当前时刻的时间
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {}  # 因为获取当前时间无需输入参数，因此parameters为空字典
        }
    },  
    # 工具2 获取指定城市的天气
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {  # 查询天气时需要提供位置，因此参数设置为location
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                            }
                        }
            },
            "required": [
                "location"
            ]
        }
    }
]

# 模拟天气查询工具。返回结果示例：“北京今天是晴天。”
def get_current_weather(location):
    return f"{location}今天是晴天。 "

# 查询当前时间的工具。返回结果示例：“当前时间：2024-04-15 17:15:18。“
def get_current_time():
    # 获取当前日期和时间
    current_datetime = datetime.now()
    # 格式化当前日期和时间
    formatted_time = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    # 返回格式化后的当前时间
    return f"当前时间：{formatted_time}。"

# 封装模型响应函数
def get_response(messages):
    response = Generation.call(
        model='llama3-8b-instruct', # 'qwen1.5-1.8b-chat',
        messages=messages,
        tools=tools,
        seed=random.randint(1, 10000),  # 设置随机数种子seed，如果没有设置，则随机数种子默认为1234
        result_format='message'  # 将输出设置为message形式
    )
    return response

@retry(stop_max_attempt_number=3, wait_fixed=5000)
def new_call_with_messages(messages):
    t = 0
    try:
        while True:
            # 模型的第一轮调用
            messages.append({'role': 'user', 'content': f'{score}'+' '+user_input})
            first_response = get_response(messages)
            if first_response.status_code == HTTPStatus.OK:
                assistant_output = first_response.output.choices[0].message
                break
            else:
                # 防止频繁调用API引起的调用失败
                time.sleep(15 + t)
                logging.info('waiting')
                t += 5
                logging.error('ERROR!!!\n Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                first_response.request_id, first_response.status_code,
                first_response.code, first_response.message
                ))
                if t > 15:
                    # 如果响应失败，将最后一条user message从messages列表里删除，确保user/assistant消息交替出现
                    messages = messages[:-1]
                    break
    except Exception as e:
        logging.error(f"Request processing failed: {e}")
        raise  # 重新抛出异常以触发重试
    #print(f"\n大模型第一轮输出信息：{first_response}\n")
    messages.append(assistant_output)
    if 'tool_calls' not in assistant_output:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        #print(f"最终答案：{assistant_output.content}") # 此处直接返回模型的回复，您可以根据您的业务，选择当无需调用工具时最终回复的内容
        return messages, assistant_output.content
    # 如果模型选择的工具是get_current_weather
    elif assistant_output.tool_calls[0]['function']['name'] == 'get_current_weather':
        tool_info = {"name": "get_current_weather", "role":"tool"}
        # print(json.loads(assistant_output.tool_calls[0]['function']['arguments']))
        location = json.loads(assistant_output.tool_calls[0]['function']['arguments'])['location']
        tool_info['content'] = get_current_weather(location)
    # 如果模型选择的工具是get_current_time
    elif assistant_output.tool_calls[0]['function']['name'] == 'get_current_time':
        tool_info = {"name": "get_current_time", "role":"tool"}
        tool_info['content'] = get_current_time()
    # print(f"工具输出信息：{tool_info['content']}\n")
    messages.append(tool_info)

    # 模型的第二轮调用，对工具的输出进行总结
    second_response = get_response(messages)
    # print(f"大模型第二轮输出信息：{second_response}\n")
    #print(f"最终答案：{second_response.output.choices[0].message['content']}")
    return messages, second_response.output.choices[0].message['content']
# 随message增多选择工具能力变弱（注意力问题？）
# 所以不要存太多历史记录，而是选择记忆比较好，或者每几轮删除

def get_keywords(file_path='keyword_list.json'):
    from trywordcloud import load_keywords
    existing_keywords = load_keywords(file_path)
    text = ''
    for key, value in existing_keywords.items():
        text += f"{key} {value}" + "\n"
    return text

def get_report(messages):
    with open('message.txt', 'w', encoding='utf-8') as file:    
        for message in messages[1:]:
            # print(message['content'])
            file.write(message['content']+ '\n')

    # 用户退出，产出服务报告
    p_a = "我是平台管理者，这是最近几次服务中的关键词，忽略情感分数，请用中文回答："

    '''
    # v1
    from trywordcloud import get_keywords
    get_keywords('message.txt')
    text_path = 'keyword_tfidf_freq.txt'

    # 打开文本：指定使用utf-8编码读取
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    '''
    #v2
    from trywordcloud import process_text_and_update_keywords
    process_text_and_update_keywords('message.txt')
    text = get_keywords()
    
    messages.append({'role': 'user', 'content': p_a+text})
    messages, response = new_call_with_messages(messages)

    with open('report.txt', 'w', encoding='utf-8') as file:
        file.write(response)

def is_chinese(text):
    pattern = re.compile(r'^[\u4e00-\u9fa5]+$')
    return True #bool(pattern.match(text))

def get_score(user_input):
    if is_chinese(user_input):
        try:
            score, _ = svm.classify([word for word in jieba.lcut(user_input)])
        except Exception as e:
            score = 0.5
            logging.error(f"classify processing failed: {e}")
    else:
        score = 0.5
        logging.info('get none chinese')
    return score

if __name__ == '__main__':
    # multi_round()
    # call_with_messages()
    # new_call_with_messages()
    
    prompt_path = 'prompt0608.txt'
    # 打开文本：指定使用utf-8编码读取
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    messages = [{'role': 'system', 'content': prompt}]
    svm_times = []
    llm_times = []
    
    from spa.classifiers import SVMClassifier
    svm = SVMClassifier()

    # 您可以自定义设置对话轮数，当前为3
    for i in range(30):
        user_input = input("请输入：（q退出）")
        if user_input == "q":
            break
        
        st_svm = time.time()
        score = get_score(user_input)
        end_svm = time.time()
        svm_time = end_svm - st_svm
        svm_times.append(svm_time)
        
        print("Score", score)
        # score = input("Score:")
        
        st_llm = time.time()
        messages, response = new_call_with_messages(messages)
        end_llm = time.time()
        llm_time = end_llm - st_llm
        llm_times.append(llm_time)

        print(response)

    st_rep = time.time()
    get_report(messages)

    print("av_svm", sum(svm_times) / len(svm_times))
    print("av_llm", sum(llm_times) / len(llm_times))
    print("av_rep", time.time() - st_rep)
