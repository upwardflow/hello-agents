"""
第一章学习复盘：旅行 Agent 的核心机制

1. 用户输入、工具调用和 Observation 如何流转？
   标准答案：
   用户输入先加入 prompt_history,再拼接成 full_prompt 发送给大语言模型。
   模型根据系统提示词生成 Action 文本;Python 解析 Action,从
   available_tools 中找到并执行真实函数。工具结果作为 Observation
   追加到 prompt_history,随后完整历史再次传给模型。

2. 为什么不能第一次调用模型就直接生成最终答案？
   标准答案：
   模型本身通常不知道实时天气。它需要先决定调用工具,由外部服务获取
   真实信息,再根据 Observation 继续决策。多轮循环的本质是：
   模型决策 -> 外部执行 -> 结果反馈 -> 模型继续决策。

3. 为什么需要 available_tools 工具字典？
   标准答案：
   模型输出的工具名只是字符串。available_tools 将工具名映射到真实的
   Python 函数,使程序能够调度工具。它也便于扩展,并构成工具白名单,
   限制模型只能调用明确注册的函数。

4. 模型请求调用未注册工具时应如何处理？为什么不能使用 eval()?
   标准答案：
   程序应拒绝调用,并将“未知工具”作为 Observation 反馈给模型。
   Python 的 eval() 会把字符串当作代码执行;模型输出是不可信输入,
   直接执行可能导致读取密钥、删除文件或运行系统命令。安全方案是：
   工具白名单 + 参数校验 + 权限限制。

5. 为什么要限制 Agent 的最大循环次数？
   标准答案：
   防止模型反复调用同一工具或因格式错误陷入无限循环,同时控制上下文
   长度、响应时间、API 费用和外部服务限流风险。

6. 为什么模型可能不遵守 Thought-Action 格式？程序如何处理？
   标准答案：
   系统提示词是自然语言软约束,而模型是概率式生成,因此可能输出解释、
   Markdown 或多组 Thought-Action。程序使用正则表达式截取第一组结果,
   再检查 Action;解析失败时,将错误作为 Observation 反馈给模型重试。
   正式项目通常优先使用 JSON Schema、结构化输出或原生 tool calling。

7. response.raise_for_status() 和 response.json() 分别做什么？
   标准答案：
   raise_for_status() 检查 HTTP 状态码,并在 4xx、5xx 时抛出异常;
   json() 将响应正文解析成 Python 字典或列表。前者避免继续处理错误响应,
   后者使程序能够按字段读取天气数据。

8. 大语言模型、系统提示词、Python 主循环和外部工具分别负责什么？
   标准答案：
   - 大语言模型：根据上下文选择下一步行动或生成最终答案。
   - 系统提示词：定义角色、可用工具、行为边界和输出协议。
   - Python 主循环：调用模型、解析 Action、调度工具、记录 Observation,
     并决定继续循环还是结束。
   - 外部工具：访问天气、搜索等外部数据源,执行真实操作。

总结：
这个 Agent 本质上是一个由大语言模型负责决策、Python 负责控制和执行、
外部工具负责获取真实信息的循环系统。模型生成的是行动文本,真正的函数
调用始终由 Python 程序完成。
"""

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求,并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式,包含一对Thought和Action:

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一:
1. 调用工具:function_name(arg_name="arg_value")
2. 结束任务:Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行,不要换行
- 当收集到足够信息可以回答用户问题时,必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""


import requests

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点,我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status() 
        # 解析返回的JSON数据
        data = response.json()
        
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc},气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败,可能是城市名称无效 - {e}"



import os
from tavily import TavilyClient

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气,使用Tavily Search API搜索并返回优化后的景点推荐。
    """

    # 从环境变量或主程序配置中获取API密钥
    api_key = os.environ.get("TAVILY_API_KEY") # 推荐方式
    # 或者,我们可以在主循环中传入,如此处代码所示

    if not api_key:
        return "错误:未配置TAVILY_API_KEY。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        # 4. 调用API,include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. Tavily返回的结果已经非常干净,可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性回答,则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉,没有找到相关的旅游景点推荐。"

        return "根据搜索,为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"


# 将所有工具函数放入一个字典,方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

from openai import OpenAI

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"

import re
from pathlib import Path
from dotenv import load_dotenv

# --- 1. 配置LLM客户端 ---
# 从项目根目录加载配置,避免依赖启动命令所在的目录。
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

missing_config = [
    name for name, value in {
        "API_KEY": API_KEY,
        "BASE_URL": BASE_URL,
        "MODEL_ID": MODEL_ID,
        "TAVILY_API_KEY": TAVILY_API_KEY,
    }.items() if not value
]
if missing_config:
    raise RuntimeError(f".env 缺少配置: {', '.join(missing_config)}")

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 初始化 ---
user_prompt = "你好,请帮我查询一下今天武汉的天气,然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "="*40)

# --- 3. 运行主循环 ---
for i in range(5): # 设置最大循环次数
    print(f"--- 循环 {i+1} ---\n")
    
    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)
    
    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action,需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)
    
    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"任务完成,最终答案: {final_answer}")
        break
    
    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误:未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
