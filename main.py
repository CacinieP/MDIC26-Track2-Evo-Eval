#!/usr/bin/env python3
"""
MinerU DataAgent - Main Entry Point
====================================

Usage:
    python main.py serve [--host 0.0.0.0] [--port 8000]   # Start API server
    python main.py parse <file_path> [--output-dir ./output]  # Single file parse
    python main.py batch <dir_path> [--output-dir ./output]   # Batch process
    python main.py demo                                       # Run demo examples
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env into os.environ before anything else reads config
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def cmd_serve(args):
    """Start the FastAPI API server."""
    import uvicorn
    host = args.host or "127.0.0.1"
    port = args.port or 8000
    print(f"[START] Starting MinerU DataAgent API server at http://{host}:{port}")
    print(f"   API docs: http://{host}:{port}/docs")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


def cmd_parse(args):
    """Parse a single document file."""
    from src.utils.logger import setup_logging
    from src.utils.config import load_config
    from src.tools import create_tool_registry
    from src.agents.graph import create_agent_graph

    setup_logging({"level": "INFO"})
    config = load_config()
    tools = create_tool_registry(config)
    graph = create_agent_graph(tool_registry=tools)

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"[X] File not found: {file_path}")
        sys.exit(1)

    # Validate path is within expected directories
    resolved = file_path.resolve()
    if not str(resolved).startswith(str(Path.cwd().resolve())):
        # Allow if file is explicitly passed (CLI tool), but log a warning
        import logging
        logging.getLogger(__name__).warning(f"File path outside CWD: {resolved}")

    # Validate file extension
    ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.docx', '.pptx', '.html', '.htm', '.tiff', '.bmp'}
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        print(f"Error: Unsupported file type '{file_path.suffix}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        sys.exit(1)

    print(f"[FILE] Parsing: {file_path.name}")
    t0 = time.time()

    result = asyncio.run(graph.ainvoke({
        "task_id": f"parse_{int(time.time())}",
        "request": args.description or f"解析文档 {file_path.name}，提取结构化数据",
        "file_path": str(file_path),
        "file_info": {"name": file_path.name, "suffix": file_path.suffix.lower()},
        "options": {"output_format": "json"},
    }))

    elapsed = time.time() - t0
    print(f"\n[OK] Completed in {elapsed:.1f}s")

    # Save results
    output_dir = Path(args.output_dir or "./data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{file_path.stem}_result.json"
    output = result.get("final_output")
    if output is None:
        output = {"status": result.get("status", "unknown"), "error": "No final output produced"}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Results saved to: {output_file}")

    # Print summary
    final = result.get("final_output")
    if final is None:
        final = {}
    if final:
        exec_summary = final.get("execution_summary", [])
        print(f"\n[STATS] Execution Summary ({len(exec_summary)} steps):")
        for step in exec_summary:
            status_icon = "[OK]" if step.get("status") == "completed" else "[X]"
            print(f"  {status_icon} {step.get('step_id', '?')}: {step.get('tool_name', '?')} [{step.get('status', '?')}]")

        verification = final.get("verification", {})
        if verification:
            print(f"\n[VERIFY] Quality Score: {verification.get('quality_score', 'N/A')}")

        logs = final.get("logs", [])
        if logs:
            print(f"\n[LOG] Logs ({len(logs)} entries):")
            for log in logs[-10:]:
                print(f"  {log}")


def cmd_batch(args):
    """Batch process all supported files in a directory."""
    from src.utils.logger import setup_logging
    from src.utils.config import load_config
    from src.tools import create_tool_registry
    from src.agents.graph import create_agent_graph

    setup_logging({"level": "INFO"})
    config = load_config()
    tools = create_tool_registry(config)
    graph = create_agent_graph(tool_registry=tools)

    input_dir = Path(args.dir_path)
    if not input_dir.is_dir():
        print(f"[X] Directory not found: {input_dir}")
        sys.exit(1)

    # Validate path is within expected directories
    resolved_dir = input_dir.resolve()
    if not str(resolved_dir).startswith(str(Path.cwd().resolve())):
        import logging
        logging.getLogger(__name__).warning(f"Directory path outside CWD: {resolved_dir}")

    supported_exts = {".pdf", ".png", ".jpg", ".jpeg", ".docx", ".pptx", ".html", ".htm", ".tiff", ".bmp"}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_exts]

    if not files:
        print(f"[X] No supported files found in {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir or "./data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DIR] Found {len(files)} files to process in {input_dir}")

    results_summary = []
    for i, file_path in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(files)}] Processing: {file_path.name}")
        t0 = time.time()

        try:
            result = asyncio.run(graph.ainvoke({
                "task_id": f"batch_{i}_{int(time.time())}",
                "request": f"批量处理: 解析文档 {file_path.name}",
                "file_path": str(file_path),
                "file_info": {"name": file_path.name, "suffix": file_path.suffix.lower()},
                "options": {},
            }))

            elapsed = time.time() - t0
            final = result.get("final_output", {})
            status = final.get("status", "unknown")
            errors = final.get("errors", [])

            print(f"  Status: {status} | Time: {elapsed:.1f}s | Errors: {len(errors)}")

            # Save result
            out_file = output_dir / f"{file_path.stem}_result.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=2)

            results_summary.append({
                "file": file_path.name,
                "status": status,
                "time": round(elapsed, 1),
                "errors": len(errors),
                "output": str(out_file),
            })

        except Exception as e:
            print(f"  [X] Failed: {e}")
            results_summary.append({
                "file": file_path.name,
                "status": "error",
                "error": str(e),
            })

    # Print summary
    print(f"\n{'='*60}")
    print(f"[STATS] Batch Summary: {len(results_summary)} files processed")
    for r in results_summary:
        icon = "[OK]" if r["status"] in ("completed", "completed_with_errors") else "[X]"
        print(f"  {icon} {r['file']}: {r['status']} ({r.get('time', '?')}s)")

    # Save summary
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] Summary saved to: {summary_file}")

    # Exit with non-zero code if there were errors
    errors = [r for r in results_summary if r.get("status") == "error"]
    if errors:
        print(f"\nCompleted with {len(errors)} error(s) out of {len(results_summary)} files")
        sys.exit(1)


def cmd_demo(args):
    """Run demo examples with synthetic test data."""
    print("[DEMO] Running MinerU DataAgent Demo")
    print("=" * 60)

    from src.utils.logger import setup_logging
    from src.utils.config import load_config
    from src.tools import create_tool_registry
    from src.agents.graph import create_agent_graph

    setup_logging({"level": "INFO"})
    config = load_config()
    tools = create_tool_registry(config)
    graph = create_agent_graph(tool_registry=tools)

    demos = [
        {
            "name": "Demo 1: 财务报表结构化提取",
            "task_id": "demo_financial_001",
            "request": "解析这份财务报告，提取所有资产负债表和利润表数据，验证数值一致性",
            "file_info": {"name": "annual_report_2025.pdf", "suffix": ".pdf", "pages": 30},
        },
        {
            "name": "Demo 2: 跨页长表格合并",
            "task_id": "demo_crosspage_002",
            "request": "解析产品规格书，合并跨页表格，消解所有指代",
            "file_info": {"name": "product_spec.pdf", "suffix": ".pdf", "pages": 15},
        },
        {
            "name": "Demo 3: 低质量拍照件处理",
            "task_id": "demo_lowquality_003",
            "request": "增强并OCR这份模糊的拍照合同文件，提取关键条款",
            "file_info": {"name": "contract_photo.jpg", "suffix": ".jpg", "pages": 1},
        },
    ]

    output_dir = Path("./data/output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    for demo in demos:
        print(f"\n{'-'*60}")
        print(f"[*] {demo['name']}")
        print(f"   Request: {demo['request']}")

        t0 = time.time()
        # Note: demo mode uses dry-run — file_path is intentionally None;
        # downstream tools should handle a missing file_path gracefully.
        result = asyncio.run(graph.ainvoke({
            "task_id": demo["task_id"],
            "request": demo["request"],
            "file_path": None,
            "file_info": demo.get("file_info", {}),
            "options": {},
        }))
        elapsed = time.time() - t0

        final = result.get("final_output")
        if final is None:
            final = {"status": result.get("status", "unknown"), "error": "No final output produced"}
        status = final.get("status", "unknown")
        exec_steps = final.get("execution_summary", [])
        verification = final.get("verification", {})

        print(f"   Status: {status} | Time: {elapsed:.1f}s")
        print(f"   Steps: {len(exec_steps)}")
        for step in exec_steps:
            icon = "[OK]" if step.get("status") == "completed" else "[!]"
            print(f"     {icon} {step.get('step_id', '?')}: {step.get('description', '?')[:50]}")
        if verification:
            print(f"   Quality: {verification.get('quality_score', 'N/A')}")

        # Save demo result
        out_file = output_dir / f"{demo['task_id']}_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2, default=str)
        print(f"   Output: {out_file}")

    print(f"\n{'='*60}")
    print("[DEMO] Demo complete! Results saved to ./data/output/demo/")


def main():
    parser = argparse.ArgumentParser(
        description="MinerU DataAgent - Intelligent Document Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py serve --port 8000
  python main.py parse ./report.pdf --output-dir ./output
  python main.py batch ./documents/ --output-dir ./output
  python main.py demo
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # parse
    parse_parser = subparsers.add_parser("parse", help="Parse a single document")
    parse_parser.add_argument("file_path", help="Path to document file")
    parse_parser.add_argument("--output-dir", default="./data/output", help="Output directory")
    parse_parser.add_argument("--description", "-d", help="Task description")

    # batch
    batch_parser = subparsers.add_parser("batch", help="Batch process directory")
    batch_parser.add_argument("dir_path", help="Directory containing documents")
    batch_parser.add_argument("--output-dir", default="./data/output", help="Output directory")

    # demo
    subparsers.add_parser("demo", help="Run demo examples")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "parse":
        cmd_parse(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
