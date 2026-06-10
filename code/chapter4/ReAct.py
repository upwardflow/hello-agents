"""
第四章学习复盘：智能体经典范式

1. ReAct 和 Plan-and-Solve 如何处理同一个多步骤任务？
   标准答案：
   ReAct 在每次获得 Observation 后再决定下一步,属于边执行、边观察、
   边规划;Plan-and-Solve 则先生成较完整的计划,再由执行器逐项完成。
   ReAct 更适合工具结果不可预测、需要动态调整的任务;Plan-and-Solve
   更适合步骤明确、需要保持整体结构的复杂任务。

2. 第一章已经实现了类似 ReAct 的流程,第四章为什么还要进行封装？
   标准答案：
   第一章用单个脚本展示完整流程,第四章则将职责拆分：
   - LLM 客户端统一管理模型、接口、超时和流式响应。
   - 工具模块统一管理工具描述、注册和调用。
   - Agent 模块集中管理提示词、Action 解析、Observation 和终止条件。
   这种封装降低了模块耦合,使模型和工具更容易替换、扩展、复用和测试。

3. Plan-and-Solve 为什么要将 Planner 和 Executor 分开？
   标准答案：
   Planner 专注于把复杂问题拆分成有顺序、可执行的步骤;Executor 专注于
   结合原始问题、计划和已有结果完成当前步骤。直接一次性生成答案容易
   漏掉步骤、顺序错误、推理跳跃或偏离目标,也难以定位具体出错环节。

4. 基础 Plan-and-Solve 在前序步骤出错时有什么不足？如何改进？
   标准答案：
   基础实现通常仍会继续执行原计划,导致错误传播到后续步骤。与 ReAct
   相比,它根据执行结果动态调整计划的能力较弱。可增加每步结果验证、
   当前步骤重试、失败后重新规划、最大重规划次数,以及保留已成功步骤。
   改进后的流程可以是：计划 -> 执行 -> 验证 -> 失败时修订计划。

5. Reflection 为什么需要 Memory?
   标准答案：
   Memory 用于保存每一版执行结果和每一轮反思反馈,从而记录完整的优化
   轨迹、获取最新结果、追踪已发现的问题,并支持调试、审计和版本比较。
   但 Memory 不能保证结果一定持续变好;模型仍可能重复建议或把结果改差。
   当前示例主要使用 get_last_execution(),get_trajectory() 尚未真正用于
   后续提示词,因此完整历史的价值还没有被完全发挥。

6. Reflection 为什么在反馈为“无需改进”时终止？如何防止过早终止？
   标准答案：
   及时终止可以减少模型调用成本、等待时间和无意义的重复修改。但模型的
   自我评价属于软判断,可能遗漏正确性、性能或安全问题。可结合单元测试、
   性能指标、代码执行检查、结构化评分或独立模型复核。更可靠的条件是：
   反思认为无需改进,并且程序测试和客观指标均通过。

7. ReAct、Plan-and-Solve 和 Reflection 的主要成本与风险是什么？
   标准答案：
   - ReAct:多轮模型和工具调用增加延迟及费用;错误 Observation 可能
     污染后续决策,也可能出现重复工具调用。
   - Plan-and-Solve:：需要额外生成计划并逐步执行;错误的初始计划可能
     影响全部后续步骤,基础实现又缺少动态修订能力。
   - Reflection:每轮通常包含反思和优化两次调用;反思可能不正确,
     导致重复修改、结果退化或过早终止。

8. 如何为“基于实时资料生成研究报告并检查事实和逻辑”选择范式？
   标准答案：
   可以组合三种范式：
   - Plan-and-Solve 负责规划研究问题、资料检索和报告结构。
   - ReAct 负责执行实时搜索,并根据实际结果动态补充或调整检索方向。
   - Reflection 负责核查报告中的事实、逻辑、引用和表达,再驱动修订。
   三种范式不是互斥选项,而是可以分别承担规划、动态执行和质量控制。

总结：
ReAct 解决“如何根据观察动态行动”,Plan-and-Solve 解决“如何先规划再
执行复杂任务”,Reflection 解决“如何检查并迭代改进已有结果”。模型给出
的是决策和反馈,Python 程序负责流程控制、工具执行、验证和硬性约束。
"""

import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
请注意,你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程,用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动,必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息,能够回答用户的最终问题时,你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在,请开始解决以下问题:
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误:LLM未能返回有效响应。"); break

            thought, action = self._parse_output(response_text)
            if thought: print(f"🤔 思考: {thought}")
            if not action: print("警告:未能解析出有效的Action,流程终止。"); break
            
            if action.startswith("Finish"):
                # 如果是Finish指令,提取最终答案并结束
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                self.history.append("Observation: 无效的Action格式,请检查。"); continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            observation = tool_function(tool_input) if tool_function else f"错误:未找到名为 '{tool_name}' 的工具。"
            
            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数,流程终止。")
        return None

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时,应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
