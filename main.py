"""
DeepAgents 生产级智能体示例
基于 LangGraph 和 DeepAgents 构建具备自动文件管理、子任务委派以及人机协同能力的智能体
"""

import os
import getpass
import uuid
from typing import Literal

from tavily import TavilyClient
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command


# ==================== 3.3 配置密钥 ====================
def setup_api_keys():
    """配置 LLM 和搜索工具的 API Key"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter DeepSeek API Key: ")

    if not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter Tavily API Key: ")


# ==================== 4.2 定义搜索工具 ====================
tavily_client = TavilyClient()


def internet_search(
    query: str,
    max_results: int = 2,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = True,
):
    """
    执行互联网搜索。
    注意：此工具会返回网页的 HTML 原始内容，数据量较大。
    """
    print(f"\n[Tool Call] 正在搜索: {query}...")
    response = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic
    )

    # 数据清洗与填充逻辑
    # 目的：确保数据量超过 20000 Token，以触发 DeepAgents 的文件拦截机制
    if "results" in response:
        for res in response["results"]:
            raw_content = res.get("raw_content") or ""
            # 如果真实内容不足，人工填充数据以用于演示
            if len(raw_content) < 20000:
                res["raw_content"] = raw_content + (" [PADDING_DATA] " * 5000)

    return response


# ==================== 5.2 混合存储后端 ====================
# 初始化全局存储（生产环境建议替换为 PostgresStore）
global_store = InMemoryStore()


def hybrid_backend_factory(runtime):
    """
    后端工厂函数：DeepAgents 运行时会自动调用它来创建存储实例。
    """
    return CompositeBackend(
        # 默认路由：临时文件存入 StateBackend (内存)
        default=StateBackend(runtime),

        # 特定路由：以 /memories/ 开头的路径存入 StoreBackend (持久化)
        routes={
            "/memories/": StoreBackend(runtime)
        }
    )


# ==================== 6.2 子智能体配置 ====================
research_subagent_config = {
    "name": "deep_researcher",
    "description": "专门用于执行复杂的互联网信息检索和分析任务。",
    "system_prompt": """你是一个严谨的研究员。
    你的任务是：
    1. 使用 internet_search 工具搜索信息。
    2. 如果搜索结果被存入文件（Output saved to file...），请务必使用 read_file 读取关键部分。
    3. 将分析结果整理为摘要返回。""",

    # 搜索工具只赋予子智能体
    "tools": [internet_search],
    "model": "deepseek-chat"
}


# ==================== 7.2 人机协同配置 ====================
def create_agent():
    """创建配置好的 DeepAgent"""
    model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        temperature=0
    )
    checkpointer = MemorySaver()

    agent = create_deep_agent(
        model=model,
        tools=[],
        store=global_store,
        backend=hybrid_backend_factory,
        subagents=[research_subagent_config],
        interrupt_on={
            "task": {"allowed_decisions": ["approve", "reject"]},
            "write_file": {"allowed_decisions": ["approve", "reject", "edit"]}
        },
        checkpointer=checkpointer,
        system_prompt="""你是项目经理。
        1. 遇到调研任务，必须使用 task 工具委派给 deep_researcher。
        2. 将调研的草稿文件保存在根目录（如 /draft.md）。
        3. 将最终的重要结论，必须写入 /memories/ 目录。
        """
    )
    return agent


# ==================== 8. 完整运行流程 ====================
def main():
    """主函数：运行完整的 Agent 流程"""
    # 配置 API Keys
    setup_api_keys()

    # 创建 Agent
    agent = create_agent()

    # 创建线程 ID 和配置
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 执行任务
    print("=" * 60)
    print("开始执行任务...")
    print("=" * 60)
    
    result = agent.invoke(
        {"messages": [("user", "请调研 LangGraph 的核心架构优势，整理成简报，并保存到我的长期记忆库中。")]},
        config=config
    )

    # 处理中断（人机协同）
    if result.get("__interrupt__"):
        print("\n" + "=" * 60)
        print("检测到中断，需要人工审核...")
        print("=" * 60)
        print("\n中断信息:", result.get("__interrupt__"))
        
        # 自动批准所有决策（实际应用中应该由用户交互决定）
        print("\n自动批准所有决策...")
        final_result = agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config=config
        )
        
        print("\n" + "=" * 60)
        print("任务执行完成！")
        print("=" * 60)
        print("\n最终结果:")
        print(final_result)
    else:
        print("\n" + "=" * 60)
        print("任务执行完成！")
        print("=" * 60)
        print("\n结果:")
        print(result)


if __name__ == "__main__":
    main()
