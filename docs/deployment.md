# MinerU DataAgent 系统部署文档

> MinerU 竞赛赛道二 -- Data Agent 数据智能体完整部署与复现指南
>
> 版本: 1.0.0 | 更新日期: 2026-05-31

---

## 目录

1. [System Architecture Overview / 系统架构总览](#1-system-architecture-overview--系统架构总览)
2. [Hardware Requirements / 硬件需求](#2-hardware-requirements--硬件需求)
3. [Software Requirements / 软件需求](#3-software-requirements--软件需求)
4. [Installation Steps / 安装步骤](#4-installation-steps--安装步骤)
5. [Configuration / 配置说明](#5-configuration--配置说明)
6. [Running the System / 运行系统](#6-running-the-system--运行系统)
7. [API Documentation / 接口文档](#7-api-documentation--接口文档)
8. [Testing & Verification / 测试与验证](#8-testing--verification--测试与验证)
9. [Troubleshooting / 常见问题排查](#9-troubleshooting--常见问题排查)
10. [Directory Structure / 目录结构说明](#10-directory-structure--目录结构说明)

---

## 1. System Architecture Overview / 系统架构总览

本系统采用 **LangGraph 状态图 + 工具注册表** 的 Agent 架构，围绕 MinerU 文档解析引擎构建多阶段智能处理管线。

```
                           MinerU DataAgent Architecture
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        Client Layer                                  │
  │  ┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐   │
  │  │  CLI parse │  │  CLI batch   │  │ REST API │  │  Demo Mode   │   │
  │  │  (main.py) │  │  (main.py)   │  │ (FastAPI)│  │ (scripts/)   │   │
  │  └─────┬──────┘  └──────┬───────┘  └────┬─────┘  └──────┬───────┘   │
  │        └────────────────┼───────────────┼───────────────┘           │
  ├─────────────────────────┼───────────────┼───────────────────────────┤
  │                   LangGraph Agent Core                              │
  │                                                                     │
  │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐    │
  │   │ analyze  │──>│  plan    │──>│ execute  │──>│   verify     │    │
  │   │ _task    │   │_execut-  │   │ _step    │   │  _result     │    │
  │   │          │   │  ion     │   │  (loop)  │   │              │    │
  │   └──────────┘   └──────────┘   └────┬─────┘   └──────┬───────┘    │
  │                                      │   ^             │            │
  │                             error    │   │ retry       │            │
  │                            handler ──┘   └─────────────┘            │
  │                                      │                              │
  │                               ┌──────┴──────┐                       │
  │                               │   format    │                       │
  │                               │   _output   │                       │
  │                               └─────────────┘                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │                      Tool Registry                                  │
  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐   │
  │  │  MinerU    │ │   Table    │ │   Chart    │ │   Image        │   │
  │  │  Parser    │ │   Parser   │ │  Analyzer  │ │  Enhancer      │   │
  │  │  (magic-   │ │  (HTML/Num │ │  (LLM/CV)  │ │  (CLAHE/Denoise│   │
  │  │   pdf)     │ │  eric/CN)  │ │            │ │   /Deskew)     │   │
  │  └────────────┘ └────────────┘ └────────────┘ └────────────────┘   │
  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                      │
  │  │  Cross-    │ │ Verifier   │ │  Exporter  │                      │
  │  │  Page      │ │  (Quality  │ │  (Output   │                      │
  │  │  Merger    │ │   Gate)    │ │  Format)   │                      │
  │  └────────────┘ └────────────┘ └────────────┘                      │
  ├─────────────────────────────────────────────────────────────────────┤
  │                    Infrastructure                                    │
  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐   │
  │  │   MinerU   │ │ PaddleOCR  │ │  OpenCV /  │ │   LLM API      │   │
  │  │  Models    │ │            │ │ scikit-    │ │  (Claude/Qwen) │   │
  │  │  (GPU/CPU) │ │            │ │  image     │ │                │   │
  │  └────────────┘ └────────────┘ └────────────┘ └────────────────┘   │
  └─────────────────────────────────────────────────────────────────────┘
```

### 核心处理管线 (Pipeline)

系统对每个文档处理任务执行以下 5 阶段管线:

```
  Preprocess → Extract → Specialize → Verify → Export
  (图像增强)   (MinerU)   (表格/图表/    (质量门)  (JSON输出)
                         跨页合并)
```

**数据流:**
1. **analyze_task** -- 分析任务请求，识别文档类型与难度，确定所需工具
2. **plan_execution** -- 生成有序执行计划 (SubTask 列表)，包含依赖关系
3. **execute_step** (循环) -- 依次执行每个子任务，支持重试 (最多 3 次)
4. **verify_result** -- 质量验证 (LLM 或启发式)，不合格则触发重规划 (最多 2 轮)
5. **format_output** -- 合并结果、生成结构化 JSON 输出与执行日志

---

## 2. Hardware Requirements / 硬件需求

### 最低配置 (CPU 模式)

| 组件 | 要求 |
|------|------|
| CPU | 8 核 x86_64 (推荐 Intel 8代+ / AMD Zen2+) |
| 内存 | 16 GB RAM |
| 硬盘 | 20 GB 可用空间 (模型 + 依赖 + 临时文件) |
| GPU | 不要求 (可用 CPU 推理，速度较慢) |

### 推荐配置 (GPU 模式)

| 组件 | 要求 |
|------|------|
| CPU | 16 核 x86_64 |
| 内存 | 32 GB RAM |
| 硬盘 | 50 GB 可用空间 (SSD) |
| GPU | NVIDIA GPU，显存 >= 8 GB (如 RTX 3060/4060+) |
| CUDA | 11.8 或 12.x |

### 性能参考

| 场景 | CPU 模式 | GPU 模式 |
|------|---------|---------|
| 10 页 PDF 解析 | ~60s | ~15s |
| 30 页财务报表 (含表格) | ~180s | ~45s |
| 单张图片 OCR + 增强 | ~10s | ~3s |
| 批量 20 文件处理 | 视文件大小 | 视文件大小 |

---

## 3. Software Requirements / 软件需求

| 软件 | 版本要求 | 说明 |
|------|---------|------|
| 操作系统 | Ubuntu 20.04+ / Windows 10+ / macOS 12+ | 推荐 Ubuntu 22.04 LTS |
| Python | 3.10, 3.11, 3.12 | **必须 3.10+** |
| CUDA Toolkit | 11.8+ 或 12.x | 仅 GPU 模式需要 |
| NVIDIA Driver | >= 525.60.13 (Linux) / >= 528.33 (Windows) | 仅 GPU 模式需要 |
| Git | >= 2.30 | 用于克隆仓库 |
| poppler-utils | 任意版本 | pdf2image 依赖 (Linux: `apt install poppler-utils`) |

### Python 核心依赖

```
fastapi >= 0.110.0          # Web 框架
uvicorn >= 0.29.0           # ASGI 服务器
pydantic >= 2.6.0           # 数据验证
pyyaml >= 6.0               # 配置文件
magic-pdf[full] >= 1.0.0   # MinerU 文档解析引擎
langgraph >= 0.2.0          # Agent 状态图框架
paddleocr >= 2.8.0          # OCR 引擎
paddlepaddle >= 2.6.0       # PaddlePaddle 深度学习框架
opencv-python >= 4.9.0      # 图像处理
Pillow >= 10.3.0            # 图像 I/O
loguru >= 0.7.0             # 日志
```

完整依赖列表参见 `requirements.txt`。

---

## 4. Installation Steps / 安装步骤

### 4.1 克隆代码仓库

```bash
git clone https://github.com/CacinieP/MIDC26-Track2-Evo-Eval.git
cd MIDC26-Track2-Evo-Eval
```

### 4.2 创建 Conda 环境

```bash
# 创建 Python 3.10 虚拟环境
conda create -n mineru-agent python=3.10 -y
conda activate mineru-agent
```

> **Windows 用户注意:** 如果没有 conda，可以使用 `python -m venv venv` 创建虚拟环境。

### 4.3 安装 PyTorch (GPU 版)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (无 GPU 时)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4.4 安装项目依赖

```bash
# 安装全部依赖
pip install -r requirements.txt

# 如果 magic-pdf[full] 安装失败，可分步安装:
pip install magic-pdf>=1.0.0
pip install paddleocr paddlepaddle
pip install fastapi uvicorn pydantic pyyaml
pip install langgraph langchain-core
pip install opencv-python Pillow scikit-image
pip install loguru structlog
pip install anthropic openai
pip install pandas numpy matplotlib
pip install beautifulsoup4 lxml
pip install python-docx python-pptx pdf2image
pip install pytest pytest-asyncio httpx
```

### 4.5 下载 MinerU 模型

MinerU 首次运行时会自动下载模型文件 (~1.5 GB)。如果需要手动下载或配置离线模型:

```bash
# 方式 1: 使用 MinerU 自带命令自动下载模型到默认位置
python -c "from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze; print('Models will download on first use')"

# 方式 2: 手动指定模型目录 (编辑 configs/config.yaml 中的 mineru.model_dir)
mkdir -p ./models
```

模型文件下载位置默认为 `~/.cache/huggingface/` 或项目目录下的 `./models`。 MinerU 需要以下模型:
- **DocLayout-YOLO** -- 文档版面分析模型
- **TableMaster** -- 表格结构识别模型
- **PaddleOCR** -- OCR 文字识别模型 (自动下载)

### 4.6 配置 magic-pdf.json (MinerU 配置)

MinerU 需要一个 `magic-pdf.json` 配置文件，通常位于用户目录下:

```bash
# Linux/macOS
~/.magic-pdf.json

# Windows
%USERPROFILE%\.magic-pdf.json
```

创建该文件:

```json
{
    "device-mode": "cuda",
    "layout-config": {
        "model": "doclayout_yolo"
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": true
    },
    "table-config": {
        "model": "tablemaster",
        "enable": true,
        "max_time": 400
    },
    "ocr-config": {
        "enable": true
    }
}
```

> 如果没有 GPU，将 `"device-mode"` 改为 `"cpu"`。

### 4.7 验证安装

```bash
# 验证 Python 版本
python --version
# 输出: Python 3.10.x 或更高

# 验证关键依赖
python -c "import magic_pdf; print('MinerU OK:', magic_pdf.__version__)"
python -c "import paddleocr; print('PaddleOCR OK')"
python -c "import langgraph; print('LangGraph OK')"
python -c "import fastapi; print('FastAPI OK:', fastapi.__version__)"
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"

# 验证 GPU 可用性 (可选)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## 5. Configuration / 配置说明

### 5.1 主配置文件 config.yaml

```bash
# 从模板复制
cp configs/config.example.yaml configs/config.yaml
```

编辑 `configs/config.yaml`，按需修改以下配置项:

```yaml
# ============================================================
# API 服务器配置
# ============================================================
server:
  host: "0.0.0.0"        # 监听地址
  port: 8000              # 监听端口
  workers: 4              # 工作进程数
  timeout: 300            # 请求超时 (秒)

# ============================================================
# LLM 配置 (任务规划与质量验证)
# ============================================================
llm:
  # 主规划器 LLM
  planner:
    provider: "anthropic"                          # anthropic | openai | openai_compatible
    model: "claude-sonnet-4-20250514"
    api_key: "${ANTHROPIC_API_KEY}"                # 从环境变量读取
    max_tokens: 4096
    temperature: 0.1

  # 验证器 LLM (可用较轻量模型)
  verifier:
    provider: "anthropic"
    model: "claude-haiku-4-5-20251001"
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 2048
    temperature: 0.0

  # 本地模型 (备选，可选)
  local:
    provider: "openai_compatible"
    base_url: "http://localhost:8001/v1"
    model: "qwen3-4b"
    max_tokens: 2048

# ============================================================
# MinerU 解析配置
# ============================================================
mineru:
  model_dir: "./models"        # 模型存放目录
  device: "cuda"               # cuda | cpu
  batch_size: 4
  use_gpu: true
  table_recognition:
    enabled: true
    model: "tablemaster"
  ocr:
    enabled: true
    engine: "paddleocr"        # paddleocr | easyocr
    lang: "ch"                 # ch | en | ch_en
  layout:
    enabled: true
    model: "doclayout_yolo"

# ============================================================
# 处理管线配置
# ============================================================
pipeline:
  max_retries: 3               # 子任务最大重试次数
  retry_delay: 5               # 重试间隔 (秒)
  max_concurrent_tasks: 8      # 最大并发任务数
  preprocess:
    image_enhancement: true    # 启用图像增强
    denoise: true              # 启用降噪
    deskew: true               # 启用倾斜校正
    dpi_target: 300            # 目标 DPI
  postprocess:
    cross_page_merge: true     # 启用跨页合并
    reference_resolution: true # 启用指代消解
    format_validation: true    # 启用格式验证

# ============================================================
# 日志配置
# ============================================================
logging:
  level: "INFO"                # DEBUG | INFO | WARNING | ERROR
  format: "json"               # json | text
  file: "./logs/agent.log"
  rotation: "100 MB"
  retention: "30 days"

# ============================================================
# 存储配置
# ============================================================
storage:
  output_dir: "./data/output"  # 结果输出目录
  temp_dir: "./data/temp"      # 临时文件目录
  cleanup_after: 3600          # 临时文件清理间隔 (秒)
```

### 5.2 环境变量

系统支持在 `config.yaml` 中使用 `${VAR_NAME}` 语法引用环境变量，也支持 `${VAR_NAME:-default}` 提供默认值:

```bash
# 设置 LLM API Key (必须)
export ANTHROPIC_API_KEY="sk-ant-xxx..."

# 或使用 OpenAI 兼容 API
export OPENAI_API_KEY="sk-xxx..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 5.3 无 LLM 模式运行

如果未配置 LLM API Key，系统会自动降级为 **启发式模式**:
- 任务分析使用关键词匹配
- 质量验证使用规则引擎
- 参考消解使用规则 + 模糊匹配

此模式下所有功能仍然可用，但分析精度会有所降低。

---

## 6. Running the System / 运行系统

### 6.1 API Server 模式

启动 REST API 服务，接受 HTTP 请求处理文档:

```bash
# 默认启动 (0.0.0.0:8000)
python main.py serve

# 自定义地址和端口
python main.py serve --host 127.0.0.1 --port 9000

# 开发模式 (启用热重载)
python main.py serve --reload

# 启动后访问 API 文档:
# http://localhost:8000/docs          (Swagger UI)
# http://localhost:8000/redoc         (ReDoc)
```

启动成功后会看到:

```
[START] Starting MinerU DataAgent API server at http://0.0.0.0:8000
   API docs: http://0.0.0.0:8000/docs
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 6.2 CLI 单文件解析

直接解析单个文档文件:

```bash
# 基本用法
python main.py parse ./report.pdf

# 指定输出目录
python main.py parse ./report.pdf --output-dir ./results

# 添加任务描述 (帮助 Agent 更好地规划)
python main.py parse ./financial.xlsx \
    --description "提取所有资产负债表数据，验证数值一致性"

# 处理其他文件类型
python main.py parse ./contract.docx
python main.py parse ./slide.pptx
python main.py parse ./page.html
python main.py parse ./photo.jpg
```

支持的文件类型:
- **PDF**: `.pdf`
- **图片**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`
- **Office**: `.docx`, `.pptx`
- **网页**: `.html`, `.htm`

输出示例:

```
[FILE] Parsing: annual_report.pdf

[OK] Completed in 23.5s
[SAVE] Results saved to: ./data/output/annual_report_result.json

[STATS] Execution Summary (5 steps):
  [OK] step_000_extract: mineru_parser [completed]
  [OK] step_001_table: table_parser [completed]
  [OK] step_002_merge: cross_page_merger [completed]
  [OK] step_003_verify: verifier [completed]

[VERIFY] Quality Score: 0.85
```

### 6.3 Batch 批量处理

批量处理目录下所有支持的文件:

```bash
# 处理整个目录
python main.py batch ./documents/

# 指定输出目录
python main.py batch ./documents/ --output-dir ./batch_results
```

输出示例:

```
[DIR] Found 12 files to process in ./documents/

============================================================
[1/12] Processing: report_01.pdf
  Status: completed | Time: 18.3s | Errors: 0

============================================================
[12/12] Processing: slide_final.pptx
  Status: completed | Time: 5.2s | Errors: 0

============================================================
[STATS] Batch Summary: 12 files processed
  [OK] report_01.pdf: completed (18.3s)
  [OK] report_02.pdf: completed (22.1s)
  ...

[SAVE] Summary saved to: ./batch_results/batch_summary.json
```

### 6.4 Demo 演示模式

运行内置演示场景 (无需真实文件，自动生成样本 PDF):

```bash
# 运行所有演示场景
python main.py demo

# 或使用独立演示脚本
python scripts/run_demo.py

# 运行指定场景
python scripts/run_demo.py --scenario 1

# 保留生成的样本文件
python scripts/run_demo.py --keep-files
```

演示场景:
1. **财务报表结构化提取** -- 多页财务报告，含资产负债表和利润表
2. **跨页长表格合并** -- 跨页产品规格表，需合并和消解指代
3. **低质量拍照件处理** -- 模糊合同扫描件，含手写签名和印章

---

## 7. API Documentation / 接口文档

所有 API 端点均为 RESTful JSON 接口。启动 API Server 后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 7.1 健康检查

```bash
curl http://localhost:8000/health
```

响应:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "tools_loaded": [
        "chart_analyzer", "cross_page_merger", "exporter",
        "image_enhancer", "mineru_parser", "table_parser", "verifier"
    ],
    "agent_ready": true,
    "active_tasks": 0
}
```

### 7.2 系统能力查询

```bash
curl http://localhost:8000/capabilities
```

响应:
```json
{
    "supported_file_types": [".bmp", ".docx", ".htm", ".html", ".jpeg", ".jpg", ".pdf", ".png", ".pptx", ".tif", ".tiff", ".webp"],
    "max_file_size_mb": 100,
    "processing_capabilities": {
        "document_parsing": { "description": "..." },
        "table_extraction": { "description": "..." },
        "chart_analysis": { "description": "..." },
        "cross_page_merge": { "description": "..." },
        "image_enhancement": { "description": "..." },
        "reference_resolution": { "description": "..." }
    },
    "output_formats": ["json", "markdown", "csv"]
}
```

### 7.3 提交任务 (JSON Body)

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "解析这份财务报告，提取所有表格数据并验证数值一致性",
    "file_url": "/path/to/report.pdf",
    "options": {}
  }'
```

响应:
```json
{
    "task_id": "task_a1b2c3d4e5f6",
    "status": "accepted",
    "message": "Task submitted. Use GET /tasks/task_a1b2c3d4e5f6 to check status."
}
```

### 7.4 上传文件处理 (Multipart Form)

```bash
curl -X POST http://localhost:8000/tasks/upload \
  -F "file=@./report.pdf" \
  -F "task_description=解析此文档并提取结构化数据" \
  -F 'options={}'
```

响应:
```json
{
    "task_id": "task_f6e5d4c3b2a1",
    "status": "accepted",
    "message": "File 'report.pdf' uploaded. Use GET /tasks/task_f6e5d4c3b2a1 to check status."
}
```

### 7.5 查询任务状态

```bash
curl http://localhost:8000/tasks/{task_id}
```

响应:
```json
{
    "task_id": "task_a1b2c3d4e5f6",
    "status": "completed",
    "progress": 1.0,
    "file_name": "report.pdf",
    "current_step": "",
    "total_steps": 5,
    "completed_steps": 5,
    "execution_plan": [
        {
            "step_id": "step_000_extract",
            "tool_name": "mineru_parser",
            "description": "Extract content using MinerU pipeline",
            "status": "completed",
            "retries": 0
        },
        {
            "step_id": "step_001_table",
            "tool_name": "table_parser",
            "description": "Extract and structure tables with numeric verification",
            "status": "completed",
            "retries": 0
        }
    ],
    "result": { ... },
    "verification": {
        "quality_score": 0.85,
        "passed": true
    },
    "logs": ["[HH:MM:SS][task_a1b2c3d4e5f6] Step step_000_extract SUCCESS (8.23s)", ...],
    "errors": [],
    "duration": 23.52
}
```

### 7.6 获取最终结果

```bash
curl http://localhost:8000/tasks/{task_id}/result
```

响应:
```json
{
    "task_id": "task_a1b2c3d4e5f6",
    "status": "completed",
    "result": {
        "task_id": "task_a1b2c3d4e5f6",
        "status": "completed",
        "request": "...",
        "file_info": { "name": "report.pdf", "suffix": ".pdf", "size": 2456789 },
        "assessment": { "task_types": [...], "difficulty": "hard" },
        "execution_summary": [ ... ],
        "raw_content": { "pages": 30, "tables": [...], "images": [...], "markdown": "..." },
        "structured_content": { "tables": [...], "charts": [...], "content_list": [...] },
        "verification": { "quality_score": 0.85, "passed": true, "issues": [] },
        "errors": [],
        "logs": [ ... ]
    },
    "errors": [],
    "duration": 23.52
}
```

### 7.7 获取执行日志

```bash
# 获取最近 200 条日志
curl http://localhost:8000/tasks/{task_id}/logs

# 限制日志数量
curl "http://localhost:8000/tasks/{task_id}/logs?limit=50"
```

### 7.8 实时日志流 (SSE / NDJSON)

```bash
curl -N http://localhost:8000/tasks/{task_id}/logs/stream
```

每行输出为一个 JSON 对象:
```
{"ts": "14:30:15", "message": "[14:30:15][task_xxx] Entering analyze_task node"}
{"ts": "14:30:16", "message": "[14:30:16][task_xxx] Step step_000_extract SUCCESS (8.23s)"}
{"ts": "14:30:25", "message": "Task task_xxx finished with status: completed", "level": "done"}
```

### 7.9 批量上传

```bash
curl -X POST http://localhost:8000/tasks/batch \
  -F "files=@./doc1.pdf" \
  -F "files=@./doc2.docx" \
  -F "files=@./chart.png" \
  -F "task_description=批量解析文档" \
  -F 'options={}'
```

响应:
```json
{
    "task_ids": ["task_111", "task_222", "task_333"],
    "status": "accepted",
    "message": "3 file(s) submitted for processing."
}
```

### 7.10 列出所有任务

```bash
curl "http://localhost:8000/tasks?limit=20&offset=0"
```

---

## 8. Testing & Verification / 测试与验证

### 8.1 运行测试

```bash
# 运行所有单元测试 (93 tests, 默认跳过集成测试)
python -m pytest tests/ -v

# 运行并显示详细输出
python -m pytest tests/ -v -s

# 运行指定测试模块
python -m pytest tests/test_api.py -v
python -m pytest tests/test_graph.py -v
python -m pytest tests/test_planner.py -v
python -m pytest tests/test_table_parser.py -v
python -m pytest tests/test_config.py -v

# 运行集成测试 (需要完整环境, 可能较慢)
python -m pytest tests/ -v -m integration

# 运行所有测试 (包括集成测试)
python -m pytest tests/ -v -m ""
```

### 8.2 功能验证步骤

**步骤 1: 验证服务健康**

```bash
# 启动 API 服务
python main.py serve &

# 等待服务就绪 (约 5-10 秒)
sleep 10

# 检查健康状态
curl http://localhost:8000/health
# 确认: agent_ready == true
```

**步骤 2: 验证文档解析**

```bash
# 上传测试文件
curl -X POST http://localhost:8000/tasks/upload \
  -F "file=@./data/samples/test.pdf" \
  -F "task_description=测试文档解析" \
  -o submit_result.json

# 提取 task_id
TASK_ID=$(python -c "import json; print(json.load(open('submit_result.json'))['task_id'])")

# 等待处理完成
sleep 30

# 查看结果
curl http://localhost:8000/tasks/$TASK_ID | python -m json.tool
```

**步骤 3: 验证 Demo 运行**

```bash
# 运行演示脚本
python main.py demo

# 验证输出文件存在
ls ./data/output/demo/
# 预期输出:
# demo_financial_001_result.json
# demo_crosspage_002_result.json
# demo_lowquality_003_result.json
# all_demo_results.json
```

**步骤 4: 验证 CLI 解析**

```bash
# 准备测试 PDF (演示脚本会生成)
python scripts/run_demo.py --scenario 1 --keep-files

# 使用 CLI 解析生成的文件
python main.py parse ./data/samples/financial_report_2025.pdf

# 验证输出
ls ./data/output/financial_report_2025_result.json
```

### 8.3 查看日志

```bash
# 实时查看服务日志
tail -f ./logs/agent.log

# 查看 API 请求日志 (loguru 格式)
grep "INFO" ./logs/agent.log | tail -50

# 查看错误日志
grep "ERROR" ./logs/agent.log

# 通过 API 查看任务日志
curl http://localhost:8000/tasks/{task_id}/logs
```

### 8.4 检查输出结果

每个任务的输出 JSON 包含以下关键字段用于验证:

```json
{
    "status": "completed",
    "execution_summary": [
        { "step_id": "step_000_extract", "status": "completed", "tool_name": "mineru_parser" },
        { "step_id": "step_001_table", "status": "completed", "tool_name": "table_parser" }
    ],
    "verification": {
        "quality_score": 0.85,
        "passed": true
    },
    "structured_content": {
        "tables": [...],
        "charts": [...],
        "content_list": [...]
    }
}
```

验证要点:
- `status` 应为 `completed` 或 `completed_with_errors`
- `execution_summary` 中各步骤 `status` 应为 `completed`
- `verification.quality_score` 应 >= 0.7 (质量阈值)
- `structured_content` 应包含解析出的内容

---

## 9. Troubleshooting / 常见问题排查

### 9.1 GPU / CUDA 相关

**问题: `torch.cuda.is_available()` 返回 False**

排查步骤:
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 支持
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

解决方案:
- 确保 NVIDIA 驱动版本 >= 525 (Linux) / >= 528 (Windows)
- 重新安装与 CUDA 版本匹配的 PyTorch (参见 4.3 节)
- 如果无 GPU，在 `configs/config.yaml` 中设置 `mineru.device: "cpu"` 和 `magic-pdf.json` 中设置 `"device-mode": "cpu"`

**问题: CUDA Out of Memory**

解决方案:
```yaml
# 在 configs/config.yaml 中调整:
mineru:
  batch_size: 1        # 减小批处理大小
  device: "cuda"
  use_gpu: true
```

### 9.2 MinerU 模型下载

**问题: MinerU 模型下载失败或超时**

解决方案:
```bash
# 方法 1: 设置 HuggingFace 镜像 (中国大陆用户)
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 手动下载模型到指定目录
# 在 configs/config.yaml 中设置 mineru.model_dir: "./models"
# 从 https://huggingface.co/opendatalab 下载模型文件到该目录

# 方法 3: 使用离线模式
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**问题: `ImportError: No module named 'magic_pdf'`**

解决方案:
```bash
pip install magic-pdf[full]
# 如果 [full] 安装失败，尝试:
pip install magic-pdf
pip install pymupdf
```

### 9.3 PaddleOCR 相关

**问题: PaddleOCR 初始化缓慢或报错**

```bash
# 检查 PaddlePaddle 安装
python -c "import paddle; print(paddle.__version__)"

# 如果使用 GPU，确保安装了 GPU 版 PaddlePaddle
python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pip/simple

# CPU 版本
python -m pip install paddlepaddle -i https://mirror.baidu.com/pip/simple
```

### 9.4 依赖冲突

**问题: `ImportError` 或 `ModuleNotFoundError`**

```bash
# 验证所有关键模块
python -c "
import magic_pdf
import paddleocr
import langgraph
import fastapi
import cv2
import numpy
import pandas
from PIL import Image
from bs4 import BeautifulSoup
print('All imports OK')
"

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

**问题: `langgraph` 版本不兼容**

```bash
pip install langgraph>=0.2.0 langchain-core>=0.3.0 --upgrade
```

### 9.5 API 服务相关

**问题: 启动时报 `Address already in use`**

```bash
# 查找占用端口的进程
# Linux/macOS:
lsof -i :8000
kill -9 <PID>

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# 或使用其他端口
python main.py serve --port 9000
```

**问题: API 请求返回 500 Internal Server Error**

```bash
# 查看详细错误日志
tail -f ./logs/agent.log

# 常见原因:
# 1. MinerU 模型未下载完成 -> 参见 9.2
# 2. 文件路径不存在 -> 检查 file_url 参数
# 3. 内存不足 -> 减小 batch_size 或 max_concurrent_tasks
```

### 9.6 文件处理相关

**问题: 上传文件返回 "Unsupported file type"**

检查文件扩展名是否在支持列表中: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.docx`, `.pptx`, `.html`, `.htm`

**问题: 解析结果为空**

```bash
# 可能原因:
# 1. PDF 为扫描件且 OCR 未启用 -> 确认 mineru.ocr.enabled: true
# 2. 图片质量过低 -> 启用图像增强: pipeline.preprocess.image_enhancement: true
# 3. 文件损坏 -> 尝试用其他工具打开文件确认完整性
```

**问题: 处理超时**

```yaml
# 调整超时时间:
server:
  timeout: 600    # 增加到 10 分钟
pipeline:
  max_retries: 1  # 减少重试次数
```

---

## 10. Directory Structure / 目录结构说明

```
MIDC26-Track2-Evo-Eval/
├── main.py                          # 主入口: CLI 命令分发 (serve/parse/batch/demo)
├── requirements.txt                 # Python 依赖清单
├── README.md                        # 项目概述
│
├── configs/                         # 配置文件目录
│   └── config.example.yaml          # 配置模板 (复制为 config.yaml 使用)
│
├── src/                             # 核心源码
│   ├── __init__.py
│   │
│   ├── agents/                      # Agent 核心模块
│   │   ├── __init__.py              # 导出 TaskPlanner, create_agent_graph 等
│   │   ├── planner.py               # 任务规划器: 分析/分解/工具选择/执行监控
│   │   └── graph.py                 # LangGraph 状态图: 6 节点编排引擎
│   │                                   #   analyze_task -> plan_execution ->
│   │                                   #   execute_step (loop) -> verify_result ->
│   │                                   #   format_output + error_handler
│   │
│   ├── tools/                       # 工具层: 可插拔的处理工具
│   │   ├── __init__.py              # create_tool_registry() 工具注册工厂
│   │   ├── mineru_parser.py         # MinerU 文档解析 (PDF/图片/Office/HTML)
│   │   │                              #   主路径: magic_pdf PymuDocDataset
│   │   │                              #   备选: PyMuPDF + PaddleOCR
│   │   ├── table_parser.py          # 表格解析: HTML解析/合并单元格/数值验证/
│   │   │                              #   财务分类/置信度评分
│   │   ├── chart_analyzer.py        # 图表分析: 分类/数据提取/描述生成
│   │   │                              #   LLM视觉 + OpenCV备选
│   │   ├── crosspage_merger.py      # 跨页合并: 表格续接/段落合并/指代消解
│   │   │                              #   实体注册 + 规则/LLM消解
│   │   └── image_enhancer.py        # 图像增强: CLAHE/降噪/倾斜校正/印章去除/
│   │                                  #   超分辨率/二值化/多轮OCR投票
│   │
│   ├── pipeline/                    # 数据处理管线 (预留)
│   │   └── __init__.py
│   │
│   ├── api/                         # API 服务层
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI 应用: REST端点/后台任务/CORS/
│   │   │                              #   请求日志/异常处理
│   │   └── task_store.py            # 线程安全内存任务存储
│   │
│   └── utils/                       # 通用工具
│       ├── __init__.py
│       ├── config.py                # 配置加载: YAML解析/环境变量展开/校验
│       └── logger.py                # 日志配置: loguru + 文件轮转
│
├── scripts/                         # 辅助脚本
│   └── run_demo.py                  # 独立演示脚本: 生成样本PDF并处理
│
├── tests/                           # 测试用例
│
├── data/                            # 数据目录
│   ├── output/                      # 处理结果输出
│   │   └── demo/                    # Demo 输出结果
│   ├── samples/                     # 样本文件 (运行 demo 时自动生成)
│   └── temp/                        # 临时文件 (上传缓存)
│
├── docs/                            # 文档
│   └── deployment.md                # 本部署文档
│
└── logs/                            # 运行日志
    └── agent.log                    # Agent 执行日志 (自动创建)
```

### 关键模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| **入口** | `main.py` | CLI 命令路由: serve / parse / batch / demo |
| **编排引擎** | `src/agents/graph.py` | LangGraph 状态图，6 个节点 + 条件路由，支持重试和重规划 |
| **任务规划** | `src/agents/planner.py` | 任务分析、子任务分解、工具选择 |
| **MinerU 解析** | `src/tools/mineru_parser.py` | 文档解析核心，支持 PDF/图片/DOCX/PPTX/HTML，含预处理和多重备选方案 |
| **表格处理** | `src/tools/table_parser.py` | HTML表格解析、合并单元格、财务分类、中文数字、数值一致性验证 |
| **图表分析** | `src/tools/chart_analyzer.py` | 图表分类(18种)、数据提取、描述生成 |
| **跨页合并** | `src/tools/crosspage_merger.py` | 跨页表格续接、段落拼接、指代消解(中英文) |
| **图像增强** | `src/tools/image_enhancer.py` | 质量评估、CLAHE/降噪/去倾斜/去印章/超分/二值化 |
| **API 服务** | `src/api/main.py` | FastAPI REST 接口，文件上传/任务追踪/日志流 |
| **配置管理** | `src/utils/config.py` | YAML 加载 + `${ENV_VAR}` 环境变量展开 |

---

## 附录 A: 快速复现清单 (Quick Start Checklist)

评测人员可按以下步骤快速复现:

```bash
# 1. 克隆代码
git clone https://github.com/CacinieP/MIDC26-Track2-Evo-Eval.git
cd MIDC26-Track2-Evo-Eval

# 2. 创建环境
conda create -n mineru-agent python=3.10 -y && conda activate mineru-agent

# 3. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 4. 配置
cp configs/config.example.yaml configs/config.yaml
# (可选) 编辑 configs/config.yaml 填入 ANTHROPIC_API_KEY

# 5. 配置 MinerU
# 创建 ~/.magic-pdf.json (参见 4.6 节)

# 6. 验证安装
python -c "import magic_pdf, paddleocr, langgraph, fastapi; print('OK')"

# 7. 运行 Demo
python main.py demo

# 8. 启动 API 服务
python main.py serve

# 9. 测试 API
curl http://localhost:8000/health
curl -X POST http://localhost:8000/tasks/upload -F "file=@./test.pdf"

# 10. CLI 单文件测试
python main.py parse ./test.pdf
```

---

*文档结束 -- MinerU DataAgent Competition Track 2 | Team CacinieP*
