#!/usr/bin/env python3
"""
MinerU DataAgent - Standalone Demo Script
==========================================

Creates sample PDF files and processes them through the full DataAgent pipeline.
Can run independently without the API server.

Usage:
    python scripts/run_demo.py                  # Run all demo scenarios
    python scripts/run_demo.py --scenario 1     # Run specific scenario
    python scripts/run_demo.py --keep-files     # Keep generated sample files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ===================================================================
# Sample PDF generation
# ===================================================================

def _create_sample_pdf(output_path: Path, title: str, content_pages: list[str]) -> Path:
    """
    Create a minimal sample PDF file with text content.

    Uses reportlab if available, otherwise falls back to a minimal
    hand-crafted PDF byte stream.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(output_path), pagesize=A4)
        for i, page_text in enumerate(content_pages):
            if i > 0:
                c.showPage()
            c.setFont("Helvetica", 12)
            # Write title on first page
            y = 780
            if i == 0:
                c.setFont("Helvetica-Bold", 16)
                c.drawString(72, y, title)
                y -= 30
                c.setFont("Helvetica", 12)

            for line in page_text.split("\n"):
                if y < 72:
                    c.showPage()
                    y = 780
                c.drawString(72, y, line.strip())
                y -= 16

        c.save()
        return output_path

    except ImportError:
        # Fallback: create a minimal valid PDF by hand
        return _create_minimal_pdf(output_path, title, content_pages)


def _create_minimal_pdf(output_path: Path, title: str, content_pages: list[str]) -> Path:
    """Create a minimal valid PDF without any external libraries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    objects: list[str] = []

    # Object 1: Catalog
    objects.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")

    # Build page content strings and page references
    page_content_refs: list[str] = []
    page_refs: list[str] = []
    obj_num = 3

    full_text = f"{title}\n\n" + "\n\n".join(content_pages)

    for page_idx, page_text in enumerate(content_pages):
        # Content stream object
        escaped = page_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        content_stream = (
            f"BT\n/F1 12 Tf\n72 750 Td\n({escaped[:200]}) Tj\nET\n"
        )
        objects.append(f"{obj_num} 0 obj\n<< /Length {len(content_stream)} >>\nstream\n{content_stream}\nendstream\nendobj\n")
        page_content_refs.append(str(obj_num))
        obj_num += 1

    # Page objects
    for i, content_ref in enumerate(page_content_refs):
        page_obj = (
            f"{obj_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 792] "
            f"/Contents {content_ref} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f">>\nendobj\n"
        )
        objects.append(page_obj)
        page_refs.append(str(obj_num))
        obj_num += 1

    # Object 2: Pages
    kids = " ".join(f"{ref} 0 R" for ref in page_refs)
    objects.insert(1, f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {len(page_refs)} >>\nendobj\n")

    # Build PDF bytes
    pdf_parts: list[str] = []
    pdf_parts.append("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: list[int] = []
    for obj in objects:
        offsets.append(len("".join(pdf_parts).encode("latin-1")))
        pdf_parts.append(obj)

    # Cross-reference table
    xref_offset = len("".join(pdf_parts).encode("latin-1"))
    xref = f"xref\n0 {len(objects) + 1}\n"
    xref += "0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n"

    pdf_parts.append(xref)
    pdf_parts.append(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n")

    with open(output_path, "wb") as f:
        f.write("".join(pdf_parts).encode("latin-1"))

    return output_path


# ===================================================================
# Demo scenario definitions
# ===================================================================

DEMO_SCENARIOS = [
    {
        "id": 1,
        "name": "Financial Report - Table Extraction",
        "description": "Multi-page financial report with balance sheet and income statement tables",
        "request": "Extract all tables from this financial report, including the balance sheet and income statement. Verify numeric consistency across all tables.",
        "filename": "financial_report_2025.pdf",
        "pages": [
            "Annual Financial Report 2025\n\nCompany: TechCorp Inc.\nFiscal Year: January 2025 - December 2025\n\nTable 1: Balance Sheet (in millions USD)\n| Item | 2025 | 2024 |\n|------|------|------|\n| Total Assets | 1,250.3 | 1,100.8 |\n| Total Liabilities | 680.5 | 620.2 |\n| Shareholders Equity | 569.8 | 480.6 |",
            "Table 2: Income Statement (in millions USD)\n| Item | Q1 | Q2 | Q3 | Q4 | Total |\n|------|-----|-----|-----|-----|-------|\n| Revenue | 85.2 | 92.1 | 88.7 | 95.3 | 361.3 |\n| COGS | 42.6 | 46.0 | 44.3 | 47.6 | 180.5 |\n| Gross Profit | 42.6 | 46.1 | 44.4 | 47.7 | 180.8 |\n| Net Income | 28.4 | 30.8 | 29.6 | 31.8 | 120.6 |",
            "Table 3: Cash Flow Statement (in millions USD)\n| Category | Amount |\n|----------|--------|\n| Operating Activities | 145.2 |\n| Investing Activities | (52.8) |\n| Financing Activities | (30.1) |\n| Net Change | 62.3 |\n\nChart: Revenue by Quarter\n[Bar chart showing Q1=85.2, Q2=92.1, Q3=88.7, Q4=95.3]",
        ],
    },
    {
        "id": 2,
        "name": "Cross-Page Table Merge",
        "description": "Product specification document with a table spanning multiple pages",
        "request": "Parse this product specification document. Merge the cross-page table into a single structured table and resolve all references to previous sections.",
        "filename": "product_specification.pdf",
        "pages": [
            "Product Specification Sheet\nModel: XC-2000 Industrial Controller\n\nSection 1: Overview\nThe XC-2000 (hereafter referred to as 'the device') is a multi-function\nindustrial controller designed for factory automation.\n\nSection 2: Technical Specifications (continued on next page)\n| Parameter | Min | Typical | Max | Unit |\n|-----------|-----|---------|-----|------|\n| Supply Voltage | 10.8 | 12.0 | 36.0 | V |\n| Operating Temp | -40 | 25 | 85 | C |\n| Current Draw | - | 150 | 250 | mA |",
            "(Table continued from page 1)\n| Parameter | Min | Typical | Max | Unit |\n|-----------|-----|---------|-----|------|\n| Digital Inputs | - | 8 | - | channels |\n| Relay Outputs | - | 4 | - | channels |\n| Analog Inputs | - | 4 | - | channels |\n| Communication | - | - | - | RS485/CAN |\n| Response Time | - | 5 | 10 | ms |\n| MTBF | - | 50,000 | - | hours |\n\nSection 3: The device supports Modbus RTU and CANopen protocols.\nRefer to Section 2 for electrical specifications.",
        ],
    },
    {
        "id": 3,
        "name": "Low-Quality Scanned Contract",
        "description": "Simulated low-quality scanned contract with handwritten annotations",
        "request": "Enhance and OCR this scanned contract document. Handle blurry text and extract key terms including dates, amounts, and party names.",
        "filename": "contract_scan.pdf",
        "pages": [
            "SERVICE AGREEMENT\n\nThis Agreement ('Agreement') is entered into as of March 15, 2025\nbetween Alpha Services Ltd. ('Service Provider') and Beta Corp ('Client').\n\n1. TERM: This Agreement shall commence on April 1, 2025 and continue\nfor a period of 24 months unless terminated earlier.\n\n2. COMPENSATION: Client shall pay Service Provider a monthly fee of\nUSD 15,000.00 for services rendered.\n\n3. SCOPE: Service Provider shall deliver IT infrastructure management\nincluding network monitoring, security audits, and help desk support.\n\n[SIGNATURE: John Smith, CEO]\n[DATE: 03/15/2025]\n[STAMP: Alpha Services Ltd.]",
        ],
    },
]


# ===================================================================
# Demo execution
# ===================================================================

async def run_scenario(
    scenario: dict,
    sample_dir: Path,
    output_dir: Path,
    graph: object,
) -> dict:
    """Run a single demo scenario: generate sample PDF and process it."""
    name = scenario["name"]
    filename = scenario["filename"]
    file_path = sample_dir / filename

    print(f"\n{'='*70}")
    print(f"  Scenario {scenario['id']}: {name}")
    print(f"{'='*70}")
    print(f"  Description: {scenario['description']}")
    print(f"  Request:     {scenario['request']}")
    print()

    # Generate sample PDF
    print(f"  [1/3] Generating sample file: {filename}")
    _create_sample_pdf(file_path, scenario["pages"][0].split("\n")[0], scenario["pages"])
    file_size = file_path.stat().st_size
    print(f"         Created: {file_path} ({file_size:,} bytes)")

    # Run through agent pipeline
    print(f"  [2/3] Processing through DataAgent pipeline...")
    t0 = time.time()

    file_info = {
        "name": filename,
        "suffix": ".pdf",
        "size": file_size,
        "pages": len(scenario["pages"]),
    }

    result = await graph.ainvoke({
        "task_id": f"demo_{scenario['id']}_{int(time.time())}",
        "request": scenario["request"],
        "file_path": str(file_path),
        "file_info": file_info,
    })

    elapsed = time.time() - t0
    final = result.get("final_output", {})
    status = final.get("status", "unknown")

    print(f"         Status: {status}")
    print(f"         Time:   {elapsed:.2f}s")

    # Print execution details
    exec_summary = final.get("execution_summary", [])
    if exec_summary:
        print(f"         Steps ({len(exec_summary)}):")
        for step in exec_summary:
            icon = "[OK]" if step.get("status") == "completed" else "[!!]"
            desc = step.get("description", "?")[:60]
            print(f"           {icon} {step.get('step_id', '?')}: {desc}")

    verification = final.get("verification", {})
    if verification:
        score = verification.get("quality_score", "N/A")
        passed = verification.get("passed", "?")
        print(f"         Verification: score={score}, passed={passed}")

    # Save result
    print(f"  [3/3] Saving results...")
    out_file = output_dir / f"scenario_{scenario['id']}_{filename.replace('.pdf', '_result.json')}"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, default=str)
    print(f"         Saved: {out_file}")

    errors = final.get("errors", [])
    if errors:
        print(f"         Errors ({len(errors)}):")
        for err in errors[:3]:
            print(f"           - {err}")

    return {
        "scenario_id": scenario["id"],
        "name": name,
        "file": str(file_path),
        "status": status,
        "elapsed": round(elapsed, 2),
        "steps": len(exec_summary),
        "quality_score": verification.get("quality_score") if verification else None,
        "errors": len(errors),
        "output": str(out_file),
    }


async def run_all_demos(
    scenarios: list[dict],
    sample_dir: Path,
    output_dir: Path,
) -> list[dict]:
    """Initialize the pipeline once and run all scenarios."""
    from src.utils.logger import setup_logging
    from src.utils.config import load_config
    from src.tools import create_tool_registry
    from src.agents.graph import create_agent_graph

    print("MinerU DataAgent - Standalone Demo")
    print("=" * 70)
    print()
    print("Initializing pipeline...")

    setup_logging({"level": "INFO"})
    config = load_config()
    tools = create_tool_registry(config)
    graph = create_agent_graph(tool_registry=tools, config=config.get("pipeline"))

    print(f"  Tools loaded: {list(tools.keys())}")
    print(f"  Sample dir:   {sample_dir}")
    print(f"  Output dir:   {output_dir}")
    print()

    sample_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for scenario in scenarios:
        try:
            r = await run_scenario(scenario, sample_dir, output_dir, graph)
            results.append(r)
        except Exception as e:
            print(f"\n  [ERROR] Scenario {scenario['id']} failed: {e}")
            results.append({
                "scenario_id": scenario["id"],
                "name": scenario["name"],
                "status": "error",
                "error": str(e),
            })

    return results


def print_summary(results: list[dict], output_dir: Path) -> None:
    """Print a final summary table."""
    print(f"\n{'='*70}")
    print("  DEMO SUMMARY")
    print(f"{'='*70}")
    print(f"  {'#':<3} {'Scenario':<40} {'Status':<20} {'Time':>6}")
    print(f"  {'-'*3} {'-'*40} {'-'*20} {'-'*6}")

    for r in results:
        sid = r.get("scenario_id", "?")
        name = r.get("name", "?")[:40]
        status = r.get("status", "?")
        elapsed = f"{r.get('elapsed', '?')}s" if isinstance(r.get('elapsed'), (int, float)) else "?"
        icon = "[OK]" if status in ("completed", "completed_with_errors") else "[!!]"
        print(f"  {sid:<3} {name:<40} {icon} {status:<17} {elapsed:>6}")

    total_time = sum(r.get("elapsed", 0) for r in results if isinstance(r.get("elapsed"), (int, float)))
    total_errors = sum(r.get("errors", 0) for r in results)
    print(f"\n  Total time: {total_time:.2f}s | Total errors: {total_errors}")
    print(f"  Results saved to: {output_dir}")

    # Save summary JSON
    summary_file = output_dir / "demo_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scenarios": results,
            "total_time": total_time,
            "total_errors": total_errors,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Summary: {summary_file}")

    print(f"\n{'='*70}")
    print("  Next steps:")
    print("    python main.py serve              # Start the API server")
    print("    python main.py parse <file>       # Process a real document")
    print("    python main.py batch <directory>  # Batch process files")
    print(f"{'='*70}")


# ===================================================================
# CLI
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MinerU DataAgent - Standalone Demo Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_demo.py                    # Run all scenarios
  python scripts/run_demo.py --scenario 1       # Run scenario 1 only
  python scripts/run_demo.py --keep-files       # Keep generated sample PDFs
  python scripts/run_demo.py --sample-dir ./samples --output-dir ./demo_output
        """,
    )
    parser.add_argument(
        "--scenario", "-s",
        type=int,
        default=None,
        help="Run a specific scenario by ID (1, 2, or 3)",
    )
    parser.add_argument(
        "--sample-dir",
        default="./data/samples",
        help="Directory for generated sample files (default: ./data/samples)",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/output/demo",
        help="Directory for demo output files (default: ./data/output/demo)",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated sample PDFs after demo (default: clean up)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    output_dir = Path(args.output_dir)

    # Select scenarios
    if args.scenario is not None:
        scenarios = [s for s in DEMO_SCENARIOS if s["id"] == args.scenario]
        if not scenarios:
            print(f"Error: scenario {args.scenario} not found. Available: 1, 2, 3")
            sys.exit(1)
    else:
        scenarios = DEMO_SCENARIOS

    # Run demos
    results = asyncio.run(run_all_demos(scenarios, sample_dir, output_dir))

    # Print summary
    print_summary(results, output_dir)

    # Clean up sample files if requested
    if not args.keep_files and sample_dir.exists():
        import shutil
        try:
            shutil.rmtree(sample_dir)
            print(f"\n  Cleaned up sample files: {sample_dir}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
