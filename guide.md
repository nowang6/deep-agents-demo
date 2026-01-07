## 一、LangGraph 通过图结构来编排 LLM 的工作流存在什么问题？

LangGraph 通过图结构来编排 LLM 的工作流，可以为我们提供了极大的灵活性。

但是，当我们尝试将 Agent 从简单的演示推向生产环境时，往往会面临两个具体的工程挑战：

首先是**上下文管理难题**。在执行深度调研或代码分析时，Agent 调用的工具可能会返回大量数据（例如网页 HTML 源码或长文档）。这些数据如果直接填入对话历史，会迅速消耗 Token 预算，导致模型推理速度下降，甚至因为超出上下文窗口而报错。

其次是**任务执行的稳定性**。面对复杂的长流程任务，Agent 容易陷入局部细节而偏离初始目标，或者因为缺乏长期记忆，无法在跨会话中保持行为的一致性。

针对这些问题，LangChain 官方推出了 DeepAgents。它不仅仅是一个工具库，更是一套标准化的运行时环境（Harness）。本文将通过代码实战，展示如何利用 DeepAgents 构建一个具备自动文件管理、子任务委派以及人机协同能力的生产级智能体。

---

## 二、DeepAgents 核心架构解析

DeepAgents 的核心在于它预装了一套 Middleware（中间件）体系，这使得开发者无需从零编写 Prompt 来教 Agent 如何规划或管理内存。

它主要包含三个核心组件：

- **文件系统中间件（Filesystem）**：当工具返回的数据量过大时，中间件会自动拦截数据并写入文件，只在上下文中保留文件路径。这让 Agent 能够处理远超其上下文窗口的数据量。
- **子智能体中间件（Subagents）**：通过将复杂任务委派给拥有独立上下文的子智能体，保持主 Agent 上下文的整洁。
- **混合存储后端（Composite Backend）**：通过路径路由，将不同类型的数据分别存储在内存（临时数据）或数据库（持久化数据）中。

---

## 三、环境搭建

### 3.1 构建环境

```bash
$ conda create -n py312 python=3.12       # 创建新环境
$ source activate py312                  # 激活环境
````

### 3.2 安装依赖

本教程基于 Python 环境。DeepAgents 依赖 LangGraph 构建，并推荐使用 Tavily 进行搜索。

```bash
$ pip install deepagents tavily-python langchain-openai langchain-anthropic langgraph \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed
```

### 3.3 配置密钥

```python
import os
import getpass

# 配置 LLM 和搜索工具的 API Key
# 建议在生产环境中使用 .env 文件管理
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter OpenAI API Key: ")

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter Tavily API Key: ")
```

---

## 四、虚拟文件系统与自动拦截

### 4.1 场景描述

在调研场景中，我们需要 Agent 阅读大量网页。

* **传统的做法**：将网页内容全部塞给 LLM，这很容易导致 Token 溢出。
* **DeepAgents 的做法**：拦截大结果，转存为文件。

### 4.2 代码实现：定义搜索工具

```python
from tavily import TavilyClient
from typing import Literal

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
```

### 4.3 代码解析

* `include_raw_content=True`：关键参数，使 Tavily 返回完整 HTML。
* 数据填充逻辑：DeepAgents 默认的文件拦截阈值是 20,000 Tokens，演示中通过填充确保触发。在生产环境通常不需要。

---

## 五、混合存储后端

### 5.1 场景描述

我们希望 Agent 能够区分“临时记忆”和“长期记忆”。

> 例如，调研过程中的草稿应该随会话结束而销毁，但用户的偏好或最终报告应该持久化保存。

### 5.2 代码实现：配置路由

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

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
```

### 5.3 代码解析

* **StateBackend**：瞬时存储，线程结束即释放。
* **StoreBackend**：持久化存储，跨线程永久保存。

---

## 六、子智能体

### 6.1 场景描述

如果主 Agent 亲自处理所有搜索、阅读和整理工作，其上下文会变得非常混乱。需要一个专门的“研究员”子智能体。

### 6.2 代码实现：配置子智能体

```python
# 子智能体配置
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
    "model": "gpt-4o"
}
```

### 6.3 代码解析

* **工具隔离**：主 Agent 不直接拥有搜索工具，必须通过子智能体委派，保证上下文整洁。

---

## 七、人机协同

### 7.1 场景描述

通过人机协同机制，对关键操作进行人工审核，确保安全与可控。

### 7.2 代码实现：配置中断策略

```python
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

model = ChatOpenAI(model="gpt-4o", temperature=0)
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
```

### 7.3 代码解析

* `interrupt_on`：定义哪些工具调用需要人工介入。
* `allowed_decisions`：支持 `approve / reject / edit`。
* `checkpointer`：用于暂停与恢复执行。

---

## 八、完整运行流程

由于引入了人机协同，运行流程为：**执行 → 中断 → 批准 → 恢复**。

```python
import uuid
from langgraph.types import Command

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {"messages": [("user", "请调研 LangGraph 的核心架构优势，整理成简报，并保存到我的长期记忆库中。")]},
    config=config
)

if result.get("__interrupt__"):
    final_result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config
    )
```

---

## 总结

至此，我们完成了一个从“玩具 Demo”到“生产级应用”的改造，具备以下特征：

1. **无限上下文**：通过文件系统中间件处理海量数据。
2. **职责清晰**：主 Agent 负责规划，子智能体负责执行。
3. **分级存储**：临时数据与长期记忆分离。
4. **安全可控**：关键操作引入人机协同。

DeepAgents 的真正价值，在于它将 Agent 的设计思路从“堆 Prompt”升级为“工程化架构设计”，这是智能体走向生产环境的关键一步。

```
```

