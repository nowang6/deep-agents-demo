# DeepAgents 生产级智能体示例

基于 LangGraph 和 DeepAgents 构建的具备自动文件管理、子任务委派以及人机协同能力的生产级智能体。

## 功能特性

1. **无限上下文**：通过文件系统中间件处理海量数据
2. **职责清晰**：主 Agent 负责规划，子智能体负责执行
3. **分级存储**：临时数据与长期记忆分离
4. **安全可控**：关键操作引入人机协同

## 环境要求

- Python >= 3.12
- DeepSeek API Key
- Tavily API Key

## 安装依赖

```bash
# 使用 uv 安装（推荐）
uv sync

# 或使用 pip
pip install deepagents tavily-python langchain-openai langchain-anthropic langgraph
```

## 配置 API Keys

在运行程序前，需要设置以下环境变量：

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

或者在运行程序时通过交互式输入配置。

## 运行示例

```bash
python main.py
```

## 代码结构

- `main.py`: 完整的 DeepAgents 实现，包含：
  - 虚拟文件系统与自动拦截
  - 混合存储后端
  - 子智能体配置
  - 人机协同机制
  - 完整运行流程

## 详细文档

请参考 `guide.md` 了解详细的实现原理和使用说明。

