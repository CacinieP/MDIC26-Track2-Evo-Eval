# MinerU DataAgent Competition - Track 2

> Data Agent 数据智能体 — 基于 MinerU 的智能文档处理系统

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-green)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Tests](https://img.shields.io/badge/Tests-318%20passed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 项目概述

本项目为 **MinerU 竞赛赛道二** 参赛方案，聚焦于构建一个具备自动化数据处理能力的 Data Agent，能够：
- 理解复杂任务需求，自动拆解多步子任务
- 调用 MinerU 等工具完成高质量文档解析与结构化处理
- 输出可验证的结构化结果与完整执行日志

## 核心难点场景

| 场景 | 描述 | 技术方案 |
|------|------|----------|
| 财务报表数字解析 | 密集数字表格、合并单元格、跨页连续性 | 表格结构识别 + LLM 校验 + 数值一致性检查 |
| 跨页合并与指代消解 | 跨页表格/段落续接、代词/缩写解析 | 页面关联分析 + 实体注册表 + LLM 消解 |
| 低质量文档鲁棒性 | 模糊、光照不均、手写签章重叠 | 自适应增强管线(7步) + 多轮 OCR 投票 |
| 复杂图表解析 | 多级嵌套图表、混合类型图 | 18 种图表分类 + LLM 视觉 + 数据提取 |
| 国标文件结构化 | GB/T 行业标准规范提取 | MinerU 解析 + 后处理结构化 |

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                      API Gateway (FastAPI)                    │
│  POST /tasks/upload │ GET /tasks/{id} │ GET /capabilities    │
├──────────────────────────────────────────────────────────────┤
│                  LangGraph Agent Core (6 节点)                │
│                                                                │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐              │
│  │ analyze  │──▶│  plan    │──▶│ execute_step │──┐           │
│  │ _task    │   │ _exec    │   │   (loop)     │  │            │
│  └──────────┘   └──────────┘   └──────┬───────┘  │           │
│                                       │          │            │
│                              error? ──┤◀─────────┘            │
│                                ↓      │                        │
│                        ┌─────────────┐│  (重试3次)             │
│                        │error_handler││                        │
│                        └──────┬──────┘│                       │
│                               ↓                                │
│                       ┌──────────────┐                        │
│                       │ verify_result│ (质量阈值 0.7)          │
│                       └──────┬───────┘                        │
│                     passed?  │  failed? → 重新规划执行          │
│                       ↓ yes  │                                │
│                ┌──────────────┐                               │
│                │format_output │                               │
│                └──────────────┘                               │
├──────────────────────────────────────────────────────────────┤
│                    Tool Registry (5+2 工具)                    │
│                                                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ MinerU   │ │ Table    │ │ Chart    │ │ Image    │        │
│  │ Parser   │ │ Parser   │ │ Analyzer │ │Enhancer  │        │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │Cross-Page│ │ Verifier │ │ Exporter │                     │
│  │ Merger   │ │(built-in)│ │(built-in)│                     │
│  └──────────┘ └──────────┘ └──────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 11.8+ (GPU 推荐)
- MinerU >= 1.0

### 安装

```bash
# 克隆仓库
git clone https://github.com/CacinieP/MIDC26-Track2-Evo-Eval.git
cd MIDC26-Track2-Evo-Eval

# 创建虚拟环境
conda create -n mineru-agent python=3.10 -y
conda activate mineru-agent

# 安装 PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt

# 配置环境
cp configs/config.example.yaml configs/config.yaml
# 编辑 config.yaml 填入 API keys（可选，不填则使用启发式模式）
```

### 运行

```bash
# 启动 API 服务
python main.py serve --port 8000

# CLI 解析单个文件
python main.py parse ./report.pdf --output-dir ./output

# 批量处理
python main.py batch ./documents/ --output-dir ./output

# 运行演示
python main.py demo

# 运行测试（318 个用例，默认跳过集成测试）
python -m pytest tests/ -v

# 运行集成测试
python -m pytest tests/ -v -m integration
```

### 快速验证

```bash
# 启动服务
python main.py serve &

# 检查健康状态
curl http://localhost:8000/health

# 上传并解析文档
curl -X POST http://localhost:8000/tasks/upload \
  -F "file=@./data/samples/financial_report_sample.pdf" \
  -F "task_description=解析财务报告，提取所有表格"

# 查看系统能力
curl http://localhost:8000/capabilities
```

## 示例输出

解析 411 页金融公开披露材料（15.87 秒）的执行日志：

```
[09:15:03][parse_1780173063] Assessment: types=[document_parse,table_extract,financial_report,cross_page_merge], difficulty=hard
[09:15:03][parse_1780173063] Plan: 4 steps — [step_000_extract, step_001_table, step_002_merge, step_003_verify]
[09:15:19][parse_1780173063] Step step_000_extract SUCCESS (15.87s) — 411 pages, 28 tables, 12 images
[09:15:19][parse_1780173063] Step step_001_table SUCCESS (0.00s)
[09:15:19][parse_1780173063] Step step_002_merge SUCCESS (0.00s)
[09:15:19][parse_1780173063] Step step_003_verify SUCCESS (0.00s)
[09:15:19][parse_1780173063] Verify: score=0.85, passed=True
```

## 项目结构

```
├── main.py                          # CLI 入口 (serve/parse/batch/demo)
├── requirements.txt                 # Python 依赖
├── pytest.ini                       # 测试配置
├── LICENSE                          # MIT License
├── configs/
│   └── config.example.yaml          # 配置模板（复制为 config.yaml 使用）
├── src/
│   ├── agents/                      # Agent 核心
│   │   ├── graph.py                 # LangGraph 状态图编排引擎（6 节点 + 3 条件路由）
│   │   └── planner.py               # 任务规划器（分析/分解/工具选择/启发式）
│   ├── tools/                       # 工具层（可插拔架构）
│   │   ├── mineru_parser.py         # MinerU 文档解析（PDF/图片/DOCX/PPTX/HTML）
│   │   ├── table_parser.py          # 财务表格解析 + 中文数字 + 数值一致性验证
│   │   ├── chart_analyzer.py        # 图表分类(18种) + 数据提取 + 描述生成
│   │   ├── crosspage_merger.py      # 跨页合并 + 实体注册表 + 指代消解
│   │   └── image_enhancer.py        # 低质量图像增强管线（7 步自适应）
│   ├── pipeline/                    # 数据处理管线（预留扩展）
│   ├── utils/                       # 通用工具
│   │   ├── config.py                # YAML 配置加载 + ${ENV_VAR} 环境变量展开
│   │   ├── logger.py                # loguru 结构化日志 + 文件轮转
│   │   └── llm_client.py           # 通用 LLM 客户端（GLM/StepFun/Anthropic/OpenAI）
│   └── api/                         # API 服务
│       ├── main.py                  # FastAPI REST API（10+ 端点 + SSE 日志流）
│       └── task_store.py            # 线程安全内存任务存储
├── tests/                           # 测试（318 用例，10 模块）
│   ├── test_api.py                  # TaskStore + API 端点（21 tests）
│   ├── test_config.py               # 配置加载/校验/环境变量展开（19 tests）
│   ├── test_graph.py                # Agent 图/路由/合并/验证（27 tests）
│   ├── test_planner.py              # 任务规划/关键词检测/依赖链（15 tests）
│   ├── test_table_parser.py         # 表格解析/数值/类型分类（14 tests）
│   ├── test_mineru_parser.py        # MinerU 解析/回退/图像预处理（38 tests）
│   ├── test_chart_analyzer.py       # 图表分类/视觉/数据提取（56 tests）
│   ├── test_crosspage_merger.py     # 跨页合并/实体/指代消解（50 tests）
│   ├── test_image_enhancer.py       # 图像增强/质量评估/OCR 投票（58 tests）
│   └── test_llm_client.py           # LLM 客户端/提供商检测/错误处理（20 tests）
├── scripts/
│   └── run_demo.py                  # 独立演示脚本（自动生成样本 PDF）
├── data/
│   ├── samples/                     # 测试样本（5 个文档：3 PDF + 1 DOCX）
│   └── output/                      # 解析结果 JSON（含输出格式说明）
├── logs/
│   └── samples/                     # 示例运行日志（2 份完整执行记录）
└── docs/
    ├── technical-report.md          # 竞赛技术报告
    ├── deployment.md                # 部署与运行文档（1173 行）
    └── demo/                        # 演示材料
        ├── demo-video.mp4           # 演示视频（2m40s）
        ├── slides.html              # HTML 幻灯片（8 slides）
        └── audio/                   # MiniMax TTS 旁白（8 段）
```

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 核心解析引擎 | MinerU (magic-pdf) v1.3 |
| Agent 框架 | LangGraph StateGraph |
| 任务规划/验证 LLM | Claude / Qwen3 / GLM / StepFun（通用客户端） |
| 表格解析 | 自研（HTML→结构化 + 中文数字 + 数值一致性验证） |
| 图像增强 | OpenCV + scikit-image（CLAHE/降噪/去倾斜/去印章/超分/二值化） |
| OCR | PaddleOCR / MinerU 内置 |
| 图表解析 | LLM 视觉 + OpenCV 备选（18 种图表类型） |
| API 服务 | FastAPI + Uvicorn |
| 日志 | loguru 结构化日志 |

## 团队

Team CacinieP

## License

MIT License — see [LICENSE](LICENSE) for details.
