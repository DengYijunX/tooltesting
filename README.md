# ToolTesting - 智能问题生成系统

基于多Agent协作的智能问题生成与解决方案系统，采用RAG（检索增强生成）技术，支持多学科自动生成高质量的问题、解决方案和测试用例。

## 项目简介

本项目是一个基于大语言模型的智能问题生成平台，通过多个专业Agent的协作，能够自动从知识库中检索相关信息，生成符合规范的问题、提供可执行的解决方案代码，并自动验证其正确性。

## 核心特性

- **多Agent协作**：采用模块化Agent设计，每个Agent负责特定功能
- **智能检索**：基于Faiss向量数据库和重排序模型的精准文档检索
- **多学科支持**：支持心理学、物理、数学、金融等多个学科
- **自动验证**：包含沙箱执行环境和多维度验证器
- **审计日志**：完整的操作日志记录，便于追溯和调试
- **反馈循环**：通过验证器反馈实现迭代优化

## 技术栈

- **Python 3.8+**
- **LangChain**：LLM应用框架
- **OpenAI API**（通过SiliconFlow）：大语言模型接口
- **Faiss**：向量数据库
- **FastAPI + Uvicorn**：Web服务框架
- **PyPDF**：PDF文档处理

## 项目结构

```
tooltesting/
├── app/                      # 核心应用代码
│   ├── agents/              # Agent模块
│   │   ├── retrieval_agent.py      # 检索Agent
│   │   ├── background_agent.py     # 背景分析Agent
│   │   ├── problem_agent.py        # 问题生成Agent
│   │   ├── solution_agent.py      # 解决方案生成Agent
│   │   ├── task_modeling_agent.py # 任务建模Agent
│   │   ├── testcase_agent.py      # 测试用例生成Agent
│   │   ├── modeling_agent.py      # 知识建模Agent
│   │   └── drafting_agent.py      # 问题起草Agent
│   ├── workflows/           # 工作流模块
│   │   ├── problem_generation_workflow.py  # 问题生成工作流
│   │   └── problem_workflow.py             # 问题处理工作流
│   ├── validators/          # 验证器模块
│   │   ├── consistency.py   # 一致性验证
│   │   ├── knowledge.py     # 知识充分性验证
│   │   └── review.py        # 最终评审
│   ├── executors/           # 执行器模块
│   │   └── sandbox.py       # 代码沙箱
│   ├── utils/               # 工具模块
│   │   └── audit_logger.py  # 审计日志
│   ├── schemas/             # 数据模型
│   ├── prompts/             # 提示词模板
│   └── config.py            # 配置文件
├── data/                    # 数据目录
├── storage/                 # 存储目录
│   └── faiss_index/        # Faiss向量索引
├── audit_logs/             # 审计日志目录
├── scripts/                # 脚本目录
├── frontend/              # 前端代码
├── .env                   # 环境变量配置
├── requirements.txt       # Python依赖
└── README.md             # 项目说明
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd tooltesting

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，并配置以下变量：

```env
# SiliconFlow API配置
SILICONFLOW_API_KEY=your_api_key_here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

# 模型配置
SF_CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
SF_EMBED_MODEL=BAAI/bge-m3
SF_RERANK_MODEL=BAAI/bge-reranker-v2-m3

# 数据目录配置
DATA_DIR=data
INDEX_DIR=storage/faiss_index

# 检索参数
RECALL_K=8
FINAL_TOP_N=2
```

**获取API Key**：
- 访问 [SiliconFlow](https://siliconflow.cn) 注册账号
- 在控制台获取API Key
- 将API Key填入 `.env` 文件

### 4. 准备数据

将你的知识库文档（PDF格式）放入 `data/` 目录，并按学科分类：

```
data/
├── psychology/
├── physics/
├── math/
└── finance/
```

### 5. 构建向量索引

```python
from app.utils.build_index import build_faiss_index

# 为指定学科构建索引
build_faiss_index(subject="math")
```

## 使用方法

### 方式一：使用工作流API

```python
from app.workflows.problem_generation_workflow import run_problem_generation_workflow

# 生成问题和解决方案
result = run_problem_generation_workflow(
    topic="计算复利",
    subject="finance"
)

# 查看结果
print(result["final_result"])
```

### 方式二：使用简单工作流

```python
from app.workflows.problem_workflow import run_problem_workflow

# 只生成问题
result = run_problem_workflow(
    topic="牛顿第二定律",
    subject="physics"
)

# 查看生成的问题
print(result["final_problem"])
```

### 方式三：直接使用Agent

```python
from app.agents.retrieval_agent import run_retrieval_agent

# 检索相关文档
retrieval_result = run_retrieval_agent(
    topic="心理学实验设计",
    subject="psychology"
)

print(retrieval_result["retrieved_docs"])
```

## 工作流程

### 问题生成工作流（完整版）

1. **检索阶段**：从知识库中检索相关文档
2. **背景分析**：分析文档并提取背景信息
3. **任务建模**：构建任务模型和知识规则
4. **问题生成**：基于任务模型生成问题
5. **解决方案生成**：生成可执行的代码解决方案
6. **测试用例生成**：生成测试用例
7. **沙箱执行**：在安全环境中执行代码
8. **验证阶段**：检查一致性和知识充分性
9. **反馈循环**：根据验证结果进行迭代优化

### 问题处理工作流（简化版）

1. **检索阶段**：检索相关文档
2. **建模阶段**：构建知识模型
3. **起草阶段**：生成问题草案

## 支持的学科

- `psychology` - 心理学
- `physics` - 物理学
- `math` - 数学
- `finance` - 金融学
- `mixed` - 混合学科
- `all` - 所有学科

## 配置说明

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SF_CHAT_MODEL` | 对话模型 | Qwen/Qwen3-8B |
| `SF_EMBED_MODEL` | 嵌入模型 | BAAI/bge-m3 |
| `SF_RERANK_MODEL` | 重排序模型 | BAAI/bge-reranker-v2-m3 |

### 检索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `RECALL_K` | 初始检索数量 | 8 |
| `FINAL_TOP_N` | 最终保留数量 | 2 |

## 审计日志

所有操作都会记录在 `audit_logs/` 目录中，日志文件命名格式为：`{timestamp}_{topic}.json`

日志包含每个步骤的输入、输出和错误信息，便于调试和追溯。

## 常见问题

### 1. API Key错误

```
ValueError: 缺少 SILICONFLOW_API_KEY，请检查 .env
```

**解决方案**：检查 `.env` 文件是否正确配置了 `SILICONFLOW_API_KEY`

### 2. 向量索引不存在

```
FileNotFoundError: faiss_index not found
```

**解决方案**：运行索引构建脚本为相应学科构建向量索引

### 3. 模型调用失败

**解决方案**：
- 检查网络连接
- 确认API Key有效
- 检查模型名称是否正确

## 开发指南

### 添加新的Agent

1. 在 `app/agents/` 目录下创建新的Agent文件
2. 实现Agent的主函数，返回标准格式结果
3. 在工作流中集成新的Agent

### 添加新的验证器

1. 在 `app/validators/` 目录下创建验证器文件
2. 实现验证逻辑，返回问题列表
3. 在工作流中调用验证器

### 自定义提示词

在 `app/prompts/` 目录下修改或添加新的提示词模板。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件至：[3014099915@qq.com]

## 致谢

感谢以下项目的支持：
- [LangChain](https://github.com/langchain-ai/langchain)
- [Faiss](https://github.com/facebookresearch/faiss)
- [OpenAI](https://openai.com/)
- [SiliconFlow](https://siliconflow.cn)