"""
Hexagon Bridge — Coverage Report Formatter
===========================================
Generates plain-English, human-readable reports from the coverage analysis.

This is the artifact judges see. It answers:
  "91% of nodes NPU-eligible. These 4 ops fall back to CPU. Here's why."

Output formats:
  - Rich terminal output (colored, formatted)
  - JSON (for dashboard consumption)
  - Plain text (for logs / CI)
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hexagon-bridge.report")


def print_coverage_report(report, use_rich: bool = True) -> None:
    """
    Print a formatted coverage report to the terminal.

    Args:
        report:   CoverageReport from graph_analyzer
        use_rich: If True, use rich library for colored output; falls back to plain text
    """
    if use_rich:
        try:
            _print_rich_report(report)
            return
        except ImportError:
            pass

    _print_plain_report(report)


def _print_rich_report(report) -> None:
    """Rich-formatted terminal report with colors and tables."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    # ── Header ──
    # Format errors override the node-level grade entirely: a graph can look
    # 85% NPU-eligible and still run 100% on CPU because QNN EP cannot build
    # from its quantization format.
    fmt_err = getattr(report, "format_error", None)
    if fmt_err:
        console.print()
        console.print(Panel(
            Text.from_markup(
                f"[bold red]QUANTIZATION FORMAT ERROR — "
                f"{fmt_err['error'].replace('_', ' ').upper()}[/bold red]\n\n"
                f"[bold]{fmt_err['verdict']}[/bold]\n\n"
                f"Offending ops: [yellow]{', '.join(fmt_err['offending_ops'])}[/yellow]\n\n"
                f"{fmt_err['explanation']}\n\n"
                f"[bold]Node-level coverage reads {report.coverage_percent:.1f}%, "
                f"but effective NPU coverage is 0%.[/bold]\n"
                f"This is the silent failure: the model loads, returns correct "
                f"outputs, and never touches the NPU.\n\n"
                f"[bold green]FIX[/bold green]\n{fmt_err['fix']}"
            ),
            title=f"🔴 {report.model_name}",
            border_style="red",
            box=box.HEAVY,
        ))
        console.print()

    # A format error forces the headline to 0%. Showing "80% — GOOD" underneath
    # a panel that says nothing runs on the NPU is exactly the mixed signal this
    # tool exists to eliminate.
    coverage = 0.0 if fmt_err else report.coverage_percent
    fallback_nodes = sum(fb["count"] for fb in report.fallback_ops)
    if fmt_err:
        color = "red"
        emoji = "🔴"
        verdict = "WILL NOT RUN ON NPU — incompatible quantization format"
    elif fallback_nodes and coverage >= 70:
        # A high percentage hides the damage: each CPU node inside the graph
        # splits the NPU graph and forces a round trip. 9 rejected LayerNorms
        # read as 93.9% here — and split whisper-tiny into 9 NPU graphs, which
        # then failed to compile on a real Snapdragon X Elite.
        color = "yellow"
        emoji = "🟡"
        verdict = (
            f"SPLIT — {fallback_nodes} compute node(s) fall back to CPU, breaking the "
            "NPU graph into pieces with a CPU round trip at each"
        )
    elif coverage >= 90:
        color = "green"
        emoji = "🟢"
        verdict = "EXCELLENT — Highly NPU-eligible"
    elif coverage >= 70:
        color = "yellow"
        emoji = "🟡"
        verdict = "GOOD — Mostly NPU-eligible with some CPU fallback"
    elif coverage >= 50:
        color = "orange3"
        emoji = "🟠"
        verdict = "MODERATE — Significant CPU fallback expected"
    else:
        color = "red"
        emoji = "🔴"
        verdict = "POOR — Majority of ops will fall back to CPU"

    # The real compiler outranks static analysis. If it split the graph or
    # failed, say so in the headline rather than under a green "EXCELLENT".
    htp = getattr(report, "htp_compile", None)
    if htp and htp.get("available") and not htp.get("ok") and not fmt_err:
        color, emoji = "red", "🔴"
        verdict = "THE HTP COMPILER DID NOT BUILD ONE NPU GRAPH — see compile check below"

    console.print()
    console.print(
        Panel(
            f"[bold]{report.model_name}[/bold]\n"
            f"[dim]{report.model_path}[/dim]",
            title="🔍 Hexagon Bridge — QNN Coverage Report",
            border_style="blue",
        )
    )

    # ── Coverage Summary ──
    console.print()
    console.print(
        f"  {emoji} [bold {color}]{coverage:.1f}% NPU-Eligible[/bold {color}]  "
        f"({report.supported_nodes}/{report.total_nodes} compute nodes)"
    )
    console.print(f"     {verdict}")
    console.print()

    if htp:
        console.print(Panel(
            Text.from_markup(describe_htp_compile(htp, markup=True)),
            title="🔧 Local HTP compile check — Qualcomm's real compiler",
            border_style=_htp_style(htp),
        ))
        console.print()

    # ── Support Level Breakdown ──
    level_table = Table(
        title="Support Level Breakdown",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    level_table.add_column("Level", style="bold")
    level_table.add_column("Count", justify="right")
    level_table.add_column("", justify="left", width=30)

    total = report.total_nodes
    if report.full_support_count > 0:
        bar = _bar(report.full_support_count, total, "green")
        level_table.add_row(
            "✅ Full Support",
            str(report.full_support_count),
            bar,
        )
    if report.partial_support_count > 0:
        bar = _bar(report.partial_support_count, total, "yellow")
        level_table.add_row(
            "⚠️  Partial Support",
            str(report.partial_support_count),
            bar,
        )
    if report.unsupported_count > 0:
        bar = _bar(report.unsupported_count, total, "red")
        level_table.add_row(
            "❌ Unsupported",
            str(report.unsupported_count),
            bar,
        )
    if report.unknown_count > 0:
        bar = _bar(report.unknown_count, total, "dim")
        level_table.add_row(
            "❓ Unknown",
            str(report.unknown_count),
            bar,
        )

    console.print(level_table)

    # ── Category Coverage ──
    if report.category_coverage:
        console.print()
        cat_table = Table(
            title="Coverage by Category",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("NPU", justify="right", style="green")
        cat_table.add_column("CPU", justify="right", style="red")
        cat_table.add_column("Coverage", justify="right")

        from scanner.op_registry import CATEGORY_DESCRIPTIONS

        for cat, stats in sorted(
            report.category_coverage.items(),
            key=lambda x: x[1]["coverage_percent"],
            reverse=True,
        ):
            cat_name = CATEGORY_DESCRIPTIONS.get(cat, cat.title())
            pct = stats["coverage_percent"]
            pct_style = "green" if pct >= 90 else "yellow" if pct >= 70 else "red"
            cat_table.add_row(
                cat_name,
                str(stats["supported"]),
                str(stats["unsupported"]),
                f"[{pct_style}]{pct:.0f}%[/{pct_style}]",
            )

        console.print(cat_table)

    # ── Fallback Ops (THE critical info) ──
    if report.fallback_ops:
        console.print()
        console.print(
            Panel(
                "[bold red]These operators will fall back to CPU at runtime:[/bold red]",
                title="⚠️  CPU Fallback Details",
                border_style="red",
            )
        )

        fb_table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            padding=(0, 1),
        )
        fb_table.add_column("Operator", style="bold red")
        fb_table.add_column("Count", justify="right")
        fb_table.add_column("Category")
        fb_table.add_column("Reason", style="dim")

        for fb in report.fallback_ops:
            fb_table.add_row(
                fb["op_type"],
                str(fb["count"]),
                fb["category"],
                fb["reason"][:80] + ("..." if len(fb["reason"]) > 80 else ""),
            )

        console.print(fb_table)
    else:
        console.print()
        console.print(
            "  [bold green]✅ No CPU fallback required — "
            "all compute ops are NPU-eligible![/bold green]"
        )

    # ── Op Type Frequency ──
    console.print()
    op_table = Table(
        title="Top 15 Operators by Frequency",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    op_table.add_column("Operator", style="bold")
    op_table.add_column("Count", justify="right")
    op_table.add_column("Status")

    from scanner.op_registry import is_supported

    # Per-node verdicts outrank the registry: an op the registry calls "partial"
    # can be rejected outright for specific quantization encodings.
    fell_back = {fb["op_type"] for fb in report.fallback_ops}

    for op_type, count in sorted(
        report.op_type_counts.items(), key=lambda x: x[1], reverse=True
    )[:15]:
        supported, level, _ = is_supported(op_type)
        if op_type in fell_back:
            status = "[red]❌ CPU[/red]"
        elif level == "full":
            status = "[green]✅ NPU[/green]"
        elif level == "partial":
            status = "[yellow]⚠️  Partial[/yellow]"
        else:
            status = "[red]❌ CPU[/red]"
        op_table.add_row(op_type, str(count), status)

    console.print(op_table)

    # ── Next Steps ──
    console.print()
    if coverage >= 90:
        console.print(
            Panel(
                "[green]This model is a strong candidate for NPU acceleration.[/green]\n"
                "Deploy with QNN EP and expect significant battery/performance gains.\n\n"
                "[dim]Next: python -m scripts.benchmark --model <path> --compare cpu,qnn[/dim]",
                title="✅ Recommendation",
                border_style="green",
            )
        )
    elif coverage >= 70:
        console.print(
            Panel(
                "[yellow]This model will benefit from NPU acceleration, "
                "but some ops will fall back to CPU.[/yellow]\n"
                "The NPU-eligible portion will still reduce battery draw and improve throughput.\n\n"
                "[dim]Consider: Can the fallback ops be replaced with NPU-friendly alternatives?[/dim]",
                title="⚠️  Recommendation",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                "[red]This model has limited NPU eligibility.[/red]\n"
                "Consider using a different model architecture or checking if newer QNN SDK "
                "versions support the missing operators.\n\n"
                "[dim]Tip: Models with decomposed attention (separate Q/K/V MatMul) "
                "tend to have better QNN coverage than fused-attention variants.[/dim]",
                title="⚠️  Recommendation",
                border_style="red",
            )
        )

    console.print()


def _bar(count: int, total: int, color: str) -> str:
    """Create a simple progress bar string."""
    if total == 0:
        return ""
    pct = count / total
    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{color}]{bar}[/{color}] {pct:.0%}"


def _htp_style(htp: dict) -> str:
    if not htp.get("available"):
        return "dim"
    return "green" if htp.get("ok") else "red"


def describe_htp_compile(htp: dict, markup: bool = False) -> str:
    """One-paragraph description of a compile_check() result."""
    def b(text, color=None):
        if not markup:
            return text
        return f"[bold {color}]{text}[/bold {color}]" if color else f"[bold]{text}[/bold]"

    if not htp.get("available"):
        return f"Skipped: {htp.get('error')}"

    version = f"QNN {htp.get('qnn_version') or '?'}"
    if htp.get("error"):
        body = f"{b('Compile FAILED', 'red')} ({version}): {htp['error']}"
    elif htp.get("ok"):
        body = (f"{b('Compiled into ONE NPU graph, no compute ops left on CPU.', 'green')} "
                f"{htp['compile_s']} s, {version}.")
    else:
        cpu = ", ".join(f"{op} ×{n}" for op, n in htp.get("cpu_ops", {}).items()) or "none"
        graphs = htp.get("npu_graphs", 0)
        body = (f"{b(f'Split into {graphs} NPU graphs', 'red')}; compiler left on CPU: {cpu}. "
                f"{htp.get('boundary_ops', 0)} quantize/dequantize nodes shuttle data between "
                f"the NPU fragments and the CPU. Static analysis cannot see this; the compiler can.")
    return body + (
        "\nPassing locally is necessary, not sufficient — the device's QNN version may differ. "
        "Confirm with: python -m scripts.aihub_validate"
    )


def _print_plain_report(report) -> None:
    """Plain text report (no colors, no rich dependency)."""
    print()
    print("=" * 70)
    print("  HEXAGON BRIDGE — QNN COVERAGE REPORT")
    print("=" * 70)
    print(f"  Model: {report.model_name}")
    print(f"  Path:  {report.model_path}")
    print("-" * 70)
    fmt_err = getattr(report, "format_error", None)
    coverage = 0.0 if fmt_err else report.coverage_percent
    print(
        f"  NPU Coverage: {coverage:.1f}% "
        f"({report.supported_nodes}/{report.total_nodes} compute nodes)"
    )
    if fmt_err:
        print(f"  !! QUANTIZATION FORMAT ERROR ({fmt_err['error']}): {fmt_err['verdict']}")
        print(f"     Offending ops: {', '.join(fmt_err['offending_ops'])}")
    htp = getattr(report, "htp_compile", None)
    if htp:
        print(f"  HTP compile check: {describe_htp_compile(htp)}")
    print()
    print(f"  Full support:    {report.full_support_count}")
    print(f"  Partial support: {report.partial_support_count}")
    print(f"  Unsupported:     {report.unsupported_count}")
    print(f"  Unknown:         {report.unknown_count}")

    if report.fallback_ops:
        print()
        print("-" * 70)
        print("  CPU FALLBACK OPERATORS:")
        print("-" * 70)
        for fb in report.fallback_ops:
            print(
                f"  ❌ {fb['op_type']} (×{fb['count']}) — {fb['reason']}"
            )

    print()
    print("-" * 70)
    print("  TOP OPERATORS BY FREQUENCY:")
    print("-" * 70)
    from scanner.op_registry import is_supported
    fell_back = {fb["op_type"] for fb in report.fallback_ops}
    for op_type, count in sorted(
        report.op_type_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]:
        supported, level, _ = is_supported(op_type)
        if op_type in fell_back:
            status = "CPU"
        else:
            status = "NPU" if level == "full" else "PARTIAL" if level == "partial" else "CPU"
        print(f"  [{status:>7}] {op_type}: {count}")

    print("=" * 70)
    print()


def save_json_report(
    reports: dict,
    output_path: str | Path,
) -> None:
    """
    Save coverage reports as JSON for dashboard consumption.

    Args:
        reports:     dict mapping filename to CoverageReport
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "tool": "Hexagon Bridge",
        "version": "1.0.0",
        "models": {},
    }

    for filename, report in reports.items():
        data["models"][filename] = report.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"JSON report saved to: {output_path}")


def generate_summary_text(reports: dict) -> str:
    """
    Generate a one-paragraph plain-English summary suitable for a pitch.

    Example output:
      "Whisper-Medium encoder is 94.2% NPU-eligible (247/262 compute nodes).
       3 operators fall back to CPU: Softmax (×8, attention layers),
       Erf (×4, GELU decomposition), Range (×2, positional encoding).
       Expected battery improvement: significant."
    """
    lines = []
    for filename, report in reports.items():
        # A format error is a whole-model verdict — it makes the node-level
        # percentage misleading, so it replaces the summary rather than joining it.
        fmt_err = getattr(report, "format_error", None)
        if fmt_err:
            lines.append(
                f"{report.model_name} will NOT run on the NPU. It was quantized as "
                f"{fmt_err['error'].replace('_', ' ')} "
                f"({', '.join(fmt_err['offending_ops'])}), which QNN EP cannot consume. "
                f"Node-level analysis reads {report.coverage_percent:.1f}% NPU-eligible, "
                f"but effective coverage is 0% — every node falls back to CPU, silently. "
                f"{fmt_err['fix'].splitlines()[0]}"
            )
            continue

        parts = [
            f"{report.model_name} is {report.coverage_percent:.1f}% NPU-eligible "
            f"({report.supported_nodes}/{report.total_nodes} compute nodes)."
        ]

        if report.fallback_ops:
            fb_descriptions = []
            for fb in report.fallback_ops[:5]:
                fb_descriptions.append(
                    f"{fb['op_type']} (×{fb['count']}, {fb['category']})"
                )
            parts.append(
                f"{len(report.fallback_ops)} operator type(s) fall back to CPU: "
                + ", ".join(fb_descriptions)
                + "."
            )
        else:
            parts.append("All compute operators are NPU-eligible — no CPU fallback needed.")

        htp = getattr(report, "htp_compile", None)
        if htp and htp.get("available"):
            parts.append("Local HTP compile: " + describe_htp_compile(htp).splitlines()[0])

        lines.append(" ".join(parts))

    return "\n\n".join(lines)
