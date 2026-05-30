# 输出结果目录

此目录存放 Data Agent 解析后的结构化结果。

## 运行生成

```bash
# 单文件解析
python main.py parse ../samples/finance_report.pdf --output-dir .

# 批量处理
python main.py batch ../samples/ --output-dir .

# API 上传
curl -X POST http://localhost:8000/tasks/upload \
  -F "file=@../samples/financial_report_sample.pdf" \
  -F "task_description=解析财务报告"
```

## 输出格式

每个文件生成一个 `{filename}_result.json`，结构如下：

```json
{
  "task_id": "parse_xxxxxxxx",
  "status": "completed",
  "file": "filename.pdf",
  "assessment": {
    "task_types": ["document_parse", "table_extract"],
    "difficulty": "medium"
  },
  "execution_plan": {
    "steps": [
      {"step_id": "step_000_extract", "tool_name": "mineru_parser", "status": "completed"},
      {"step_id": "step_001_table", "tool_name": "table_parser", "status": "completed"},
      {"step_id": "step_002_verify", "tool_name": "verifier", "status": "completed"}
    ]
  },
  "structured_content": {
    "pages": [{"page_num": 1, "blocks": [...]}],
    "tables": [{"headers": [...], "rows": [...], "confidence": 0.95}],
    "charts": [{"type": "bar", "description": "...", "data": [...]}]
  },
  "verification": {
    "quality_score": 0.85,
    "issues": [],
    "passed": true
  },
  "execution_summary": [
    {"step_id": "step_000_extract", "tool_name": "mineru_parser", "status": "completed", "duration_ms": 15870}
  ],
  "errors": [],
  "logs": ["..."]
}
```
