import json
import lzma
import math
import os
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from model import ByteDataloader, make_splits
from models import MODEL_REGISTRY, BytePredictor, create_model, list_models
from train import evaluate_bpp

app = typer.Typer(
    name="boa-benchmark",
    help="Benchmark different byte-predictor backbones for BOA compression.",
    rich_markup_mode="rich",
)
console = Console()


@dataclass
class ModelMetrics:
    architecture: str = ""
    d_model: int = 0
    num_layers: int = 0
    param_count: int = 0
    memory_mb: float = 0.0


@dataclass
class TrainingMetrics:
    train_bpb: float = 0.0
    val_bpb: float = 0.0
    test_bpb: float = 0.0
    train_time_s: float = 0.0
    steps_per_sec: float = 0.0
    epochs: int = 0


@dataclass
class CompressionMetrics:
    original_bytes: int = 0
    compressed_bytes: int = 0
    ratio: float = 0.0
    bpb: float = 0.0
    compress_time_s: float = 0.0
    decompress_time_s: float = 0.0
    compress_throughput_mbs: float = 0.0
    decompress_throughput_mbs: float = 0.0
    lossless_verified: bool = False


@dataclass
class LatencyMetrics:
    init_stream_us: float = 0.0
    step_mean_us: float = 0.0
    step_p50_us: float = 0.0
    step_p99_us: float = 0.0


@dataclass
class BaselineMetrics:
    name: str = ""
    compressed_bytes: int = 0
    ratio: float = 0.0
    time_s: float = 0.0


@dataclass
class BenchmarkResult:
    model: ModelMetrics = field(default_factory=ModelMetrics)
    training: TrainingMetrics = field(default_factory=TrainingMetrics)
    compression: CompressionMetrics = field(default_factory=CompressionMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)


@dataclass
class BenchmarkReport:
    data_file: str = ""
    data_size_bytes: int = 0
    device: str = ""
    timestamp: str = ""
    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    baselines: list[BaselineMetrics] = field(default_factory=list)


def _measure_step_latency(model: BytePredictor, device: str,
                          batch_size: int = 64, n_steps: int = 256
                          ) -> LatencyMetrics:
    model.eval().to(device)
    cache = model.init_stream(max_len=n_steps + 1, batch_size=batch_size,
                              device=device, dtype=torch.float32)
    prev = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(min(16, n_steps)):
        model.step(prev, cache)
    if device == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.step(prev, cache)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1e6)

    arr = np.array(latencies)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.init_stream(max_len=n_steps + 1, batch_size=batch_size,
                      device=device, dtype=torch.float32)
    if device == "cuda":
        torch.cuda.synchronize()
    init_us = (time.perf_counter() - t0) * 1e6

    return LatencyMetrics(
        init_stream_us=init_us,
        step_mean_us=float(arr.mean()),
        step_p50_us=float(np.percentile(arr, 50)),
        step_p99_us=float(np.percentile(arr, 99)),
    )


def _compute_baselines(data: bytes) -> list[BaselineMetrics]:
    baselines = []
    orig = len(data)

    t0 = time.perf_counter()
    try:
        comp = lzma.compress(data, preset=9 | getattr(lzma, "PRESET_EXTREME", 0))
    except Exception:
        comp = lzma.compress(data, preset=9)
    t_lz = time.perf_counter() - t0
    baselines.append(BaselineMetrics(
        name="LZMA", compressed_bytes=len(comp),
        ratio=orig / len(comp) if comp else 0, time_s=t_lz,
    ))

    t0 = time.perf_counter()
    comp = zlib.compress(data, level=9)
    t_zl = time.perf_counter() - t0
    baselines.append(BaselineMetrics(
        name="ZLIB", compressed_bytes=len(comp),
        ratio=orig / len(comp) if comp else 0, time_s=t_zl,
    ))

    return baselines


def _run_single_benchmark(
    arch: str, data_bytes: bytes, device: str,
    d_model: int, num_layers: int, vocab_size: int,
    epochs: int, seq_len: int, batch_size: int, lr: float,
    chunks_count: int, progress_cb=None,
) -> BenchmarkResult:
    result = BenchmarkResult()

    model = create_model(arch, d_model=d_model, num_layers=num_layers,
                         vocab_size=vocab_size, device=device)
    result.model = ModelMetrics(
        architecture=arch, d_model=d_model, num_layers=num_layers,
        param_count=model.count_parameters(),
        memory_mb=model.memory_footprint_mb(),
    )

    train_b, val_b, test_b = make_splits(data_bytes, seq_len, batch_size)
    train_loader = ByteDataloader(train_b, seq_len=seq_len,
                                  batch_size=batch_size, device=device)
    val_loader = ByteDataloader(val_b, seq_len=seq_len,
                                batch_size=batch_size, device=device)
    test_loader = ByteDataloader(test_b, seq_len=seq_len,
                                 batch_size=batch_size, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train().to(device)
    total_steps = 0
    t_train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for batch in train_loader:
            x = batch[:, :-1].to(device, non_blocking=True)
            y = batch[:, 1:].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_steps += 1

    train_time = time.perf_counter() - t_train_start

    model.eval()
    val_bpb = evaluate_bpp(model, val_loader, criterion, device=device,
                           vocab_size=vocab_size)
    test_bpb = evaluate_bpp(model, test_loader, criterion, device=device,
                            vocab_size=vocab_size)

    result.training = TrainingMetrics(
        train_bpb=loss.item() / np.log(2),
        val_bpb=val_bpb, test_bpb=test_bpb,
        train_time_s=train_time,
        steps_per_sec=total_steps / max(train_time, 1e-9),
        epochs=epochs,
    )
    if progress_cb is not None:
        progress_cb("latency")

    result.latency = _measure_step_latency(model, device, batch_size=64,
                                           n_steps=256)
    if progress_cb is not None:
        progress_cb("compression")

    from boa import BOA

    bench_dir = Path("benchmark_workspace")
    bench_dir.mkdir(exist_ok=True)
    data_path = bench_dir / f"bench_input_{arch}.bin"
    data_path.write_bytes(data_bytes)

    boa_path = bench_dir / f"bench_{arch}.boa"
    boa = BOA(device, str(boa_path), model)

    t0 = time.perf_counter()
    boa.compress(data_path=str(data_path), chunks_count=chunks_count,
                 progress=False)
    compress_time = time.perf_counter() - t0

    boa_size = boa_path.stat().st_size
    orig_size = len(data_bytes)
    ratio = orig_size / boa_size if boa_size > 0 else float("inf")
    bpb = 8.0 / ratio if ratio > 0 else float("inf")

    boa2 = BOA(device, str(boa_path), model)
    t0 = time.perf_counter()
    decompressed = boa2.decompress(progress=False)
    decompress_time = time.perf_counter() - t0
    if progress_cb is not None:
        progress_cb("decompression")

    lossless = decompressed == data_bytes

    result.compression = CompressionMetrics(
        original_bytes=orig_size,
        compressed_bytes=boa_size,
        ratio=ratio,
        bpb=bpb,
        compress_time_s=compress_time,
        decompress_time_s=decompress_time,
        compress_throughput_mbs=orig_size / (1024 * 1024) / max(compress_time, 1e-9),
        decompress_throughput_mbs=orig_size / (1024 * 1024) / max(decompress_time, 1e-9),
        lossless_verified=lossless,
    )
    if progress_cb is not None:
        progress_cb("done")

    return result


def _build_overview_panel(report: BenchmarkReport) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("Data file", report.data_file)
    grid.add_row("Data size", f"{report.data_size_bytes:,} bytes "
                 f"({report.data_size_bytes / 1024 / 1024:.2f} MiB)")
    grid.add_row("Device", report.device)
    grid.add_row("Architectures", ", ".join(report.results.keys()))
    grid.add_row("Timestamp", report.timestamp)
    return Panel(grid, title="Benchmark Overview", border_style="blue")


def _build_model_table(report: BenchmarkReport) -> Table:
    table = Table(title="Model Architecture Comparison", box=box.ROUNDED,
                  show_lines=True)
    table.add_column("Metric", style="bold")
    for arch in report.results:
        table.add_column(arch.upper(), justify="right")

    rows = [
        ("Parameters", lambda r: f"{r.model.param_count:,}"),
        ("Memory (MiB)", lambda r: f"{r.model.memory_mb:.2f}"),
        ("d_model", lambda r: str(r.model.d_model)),
        ("Layers", lambda r: str(r.model.num_layers)),
    ]
    for label, fn in rows:
        table.add_row(label, *[fn(r) for r in report.results.values()])
    return table


def _build_training_table(report: BenchmarkReport) -> Table:
    table = Table(title="Training Metrics", box=box.ROUNDED, show_lines=True)
    table.add_column("Metric", style="bold")
    for arch in report.results:
        table.add_column(arch.upper(), justify="right")

    rows = [
        ("Epochs", lambda r: str(r.training.epochs)),
        ("Train BPB", lambda r: f"{r.training.train_bpb:.4f}"),
        ("Val BPB", lambda r: f"{r.training.val_bpb:.4f}"),
        ("Test BPB", lambda r: f"{r.training.test_bpb:.4f}"),
        ("Train time (s)", lambda r: f"{r.training.train_time_s:.1f}"),
        ("Steps/sec", lambda r: f"{r.training.steps_per_sec:.1f}"),
    ]
    for label, fn in rows:
        table.add_row(label, *[fn(r) for r in report.results.values()])
    return table


def _build_compression_table(report: BenchmarkReport) -> Table:
    table = Table(title="Compression Metrics", box=box.ROUNDED, show_lines=True)
    table.add_column("Metric", style="bold")
    for arch in report.results:
        table.add_column(arch.upper(), justify="right")
    for bl in report.baselines:
        table.add_column(bl.name, justify="right", style="dim")

    def _bl_val(bl, fn):
        try:
            return fn(bl)
        except Exception:
            return "-"

    rows = [
        ("Compressed (bytes)",
         lambda r: f"{r.compression.compressed_bytes:,}",
         lambda b: f"{b.compressed_bytes:,}"),
        ("Ratio",
         lambda r: f"{r.compression.ratio:.2f}x",
         lambda b: f"{b.ratio:.2f}x"),
        ("BPB",
         lambda r: f"{r.compression.bpb:.4f}",
         lambda b: f"{8.0 / b.ratio:.4f}" if b.ratio else "-"),
        ("Compress (s)",
         lambda r: f"{r.compression.compress_time_s:.2f}",
         lambda b: f"{b.time_s:.2f}"),
        ("Decompress (s)",
         lambda r: f"{r.compression.decompress_time_s:.2f}",
         lambda b: "-"),
        ("Compress MB/s",
         lambda r: f"{r.compression.compress_throughput_mbs:.2f}",
         lambda b: "-"),
        ("Decompress MB/s",
         lambda r: f"{r.compression.decompress_throughput_mbs:.2f}",
         lambda b: "-"),
        ("Lossless",
         lambda r: "YES" if r.compression.lossless_verified else "NO",
         lambda b: "-"),
    ]
    for label, fn_r, fn_b in rows:
        vals = [fn_r(r) for r in report.results.values()]
        vals += [_bl_val(b, fn_b) for b in report.baselines]
        table.add_row(label, *vals)
    return table


def _build_latency_table(report: BenchmarkReport) -> Table:
    table = Table(title="Step Latency (microseconds)", box=box.ROUNDED,
                  show_lines=True)
    table.add_column("Metric", style="bold")
    for arch in report.results:
        table.add_column(arch.upper(), justify="right")

    rows = [
        ("init_stream", lambda r: f"{r.latency.init_stream_us:.0f}"),
        ("step mean", lambda r: f"{r.latency.step_mean_us:.1f}"),
        ("step p50", lambda r: f"{r.latency.step_p50_us:.1f}"),
        ("step p99", lambda r: f"{r.latency.step_p99_us:.1f}"),
    ]
    for label, fn in rows:
        table.add_row(label, *[fn(r) for r in report.results.values()])
    return table


def _build_winners_panel(report: BenchmarkReport) -> Panel:
    if not report.results:
        return Panel(
            "[yellow]No successful architecture runs.[/yellow]\n"
            "Only baselines are available. Check the errors above and rerun.",
            title="Winners",
            border_style="yellow",
        )
    categories = {
        "Best compression ratio": lambda r: r.compression.ratio,
        "Lowest BPB": lambda r: -r.compression.bpb,
        "Fastest compress": lambda r: r.compression.compress_throughput_mbs,
        "Fastest decompress": lambda r: r.compression.decompress_throughput_mbs,
        "Lowest step latency": lambda r: -r.latency.step_mean_us,
        "Fewest parameters": lambda r: -r.model.param_count,
        "Best val BPB": lambda r: -r.training.val_bpb,
    }
    tree = Tree("[bold]Category Winners[/bold]")
    for cat, key_fn in categories.items():
        winner = max(report.results.items(), key=lambda kv: key_fn(kv[1]))
        tree.add(f"[bold]{cat}[/bold]: [green]{winner[0].upper()}[/green]")
    return Panel(tree, title="Winners", border_style="green")


def _render_dashboard(report: BenchmarkReport):
    console.print()
    console.rule("[bold blue]BOA Backbone Benchmark Results[/bold blue]")
    console.print()
    console.print(_build_overview_panel(report))
    console.print()
    if report.results:
        console.print(_build_model_table(report))
        console.print()
        console.print(_build_training_table(report))
        console.print()
    console.print(_build_compression_table(report))
    console.print()
    if report.results:
        console.print(_build_latency_table(report))
        console.print()
    console.print(_build_winners_panel(report))
    console.print()
    console.rule("[bold blue]End of Report[/bold blue]")
    console.print()


def _report_to_dict(report: BenchmarkReport) -> dict:
    d = {
        "data_file": report.data_file,
        "data_size_bytes": report.data_size_bytes,
        "device": report.device,
        "timestamp": report.timestamp,
        "baselines": [asdict(b) for b in report.baselines],
        "results": {k: asdict(v) for k, v in report.results.items()},
    }
    return d


def _dict_to_report(d: dict) -> BenchmarkReport:
    report = BenchmarkReport(
        data_file=d["data_file"],
        data_size_bytes=d["data_size_bytes"],
        device=d["device"],
        timestamp=d["timestamp"],
    )
    for bl in d.get("baselines", []):
        report.baselines.append(BaselineMetrics(**bl))
    for arch, rd in d.get("results", {}).items():
        br = BenchmarkResult(
            model=ModelMetrics(**rd["model"]),
            training=TrainingMetrics(**rd["training"]),
            compression=CompressionMetrics(**rd["compression"]),
            latency=LatencyMetrics(**rd["latency"]),
        )
        report.results[arch] = br
    return report


@app.command()
def run(
    data: Path = typer.Option(..., help="Path to binary input data file"),
    archs: str = typer.Option("mamba,transformer,gru,conv",
                              help="Comma-separated list of architectures"),
    d_model: int = typer.Option(256, help="Model hidden dimension"),
    num_layers: int = typer.Option(2, help="Number of layers"),
    epochs: int = typer.Option(5, help="Training epochs per architecture"),
    seq_len: int = typer.Option(4096, help="Sequence length for training"),
    batch_size: int = typer.Option(3, help="Batch size for training"),
    lr: float = typer.Option(5e-4, help="Learning rate"),
    chunks_count: int = typer.Option(500, help="Number of chunks for compression"),
    vocab_size: int = typer.Option(256, help="Vocabulary size"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu",
                               help="Device (cuda|cpu)"),
    output: Path = typer.Option("benchmark_results.json",
                                help="Output JSON file for results"),
):
    if not data.exists():
        console.print(f"[red]Error: data file '{data}' not found[/red]")
        raise typer.Exit(1)

    arch_list = [a.strip() for a in archs.split(",") if a.strip()]
    for a in arch_list:
        if a not in MODEL_REGISTRY:
            console.print(f"[red]Unknown architecture '{a}'. "
                          f"Available: {', '.join(list_models())}[/red]")
            raise typer.Exit(1)

    data_bytes = data.read_bytes()
    console.print()
    console.rule("[bold blue]BOA Backbone Benchmark[/bold blue]")
    console.print(f"  Data: {data}  ({len(data_bytes):,} bytes)")
    console.print(f"  Architectures: {', '.join(arch_list)}")
    console.print(f"  Device: {device}")
    console.print(f"  Epochs: {epochs}  |  d_model: {d_model}  |  layers: {num_layers}")
    console.print()

    report = BenchmarkReport(
        data_file=str(data),
        data_size_bytes=len(data_bytes),
        device=device,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    with console.status("[bold]Computing LZMA / ZLIB baselines...[/bold]"):
        report.baselines = _compute_baselines(data_bytes)
    for bl in report.baselines:
        console.print(f"  {bl.name}: ratio={bl.ratio:.2f}x  time={bl.time_s:.2f}s")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        phase_count = 4
        overall = progress.add_task("Overall", total=len(arch_list) * phase_count)
        for arch in arch_list:
            progress.update(overall, description=f"[bold]{arch}[/bold]: training")
            phases_done = {"count": 0}

            def _tick(next_label: str):
                phases_done["count"] += 1
                progress.advance(overall, 1)
                progress.update(
                    overall,
                    description=f"[bold]{arch}[/bold]: {next_label}",
                )

            try:
                result = _run_single_benchmark(
                    arch=arch, data_bytes=data_bytes, device=device,
                    d_model=d_model, num_layers=num_layers,
                    vocab_size=vocab_size, epochs=epochs, seq_len=seq_len,
                    batch_size=batch_size, lr=lr, chunks_count=chunks_count,
                    progress_cb=_tick,
                )
                report.results[arch] = result
                console.print(f"  [green]{arch}[/green]: ratio={result.compression.ratio:.2f}x  "
                              f"BPB={result.compression.bpb:.4f}  "
                              f"step={result.latency.step_mean_us:.0f}us  "
                              f"lossless={result.compression.lossless_verified}")
            except Exception as e:
                console.print(f"  [red]{arch} FAILED: {e}[/red]")
                import traceback
                traceback.print_exc()
                remaining = max(0, phase_count - phases_done["count"])
                if remaining:
                    progress.advance(overall, remaining)
                if device == "cuda" and "device-side assert" in str(e).lower():
                    console.print(
                        "[yellow]CUDA device-side assert detected; stopping remaining runs. "
                        "Restart the process before retrying.[/yellow]"
                    )
                    break

    with open(output, "w") as f:
        json.dump(_report_to_dict(report), f, indent=2)
    console.print(f"\nResults saved to [bold]{output}[/bold]")

    _render_dashboard(report)


@app.command()
def compare(
    results: Path = typer.Option("benchmark_results.json",
                                 help="Path to benchmark results JSON"),
):
    if not results.exists():
        console.print(f"[red]Results file '{results}' not found[/red]")
        raise typer.Exit(1)

    with open(results) as f:
        data = json.load(f)
    report = _dict_to_report(data)
    _render_dashboard(report)


@app.command()
def info():
    table = Table(title="Available Architectures", box=box.ROUNDED)
    table.add_column("Name", style="bold green")
    table.add_column("Class")
    table.add_column("Description")

    descriptions = {
        "mamba": "Mamba SSM (state-space model) -- the original BOA backbone. "
                 "O(1) per step via selective scan cache.",
        "transformer": "Causal Transformer with KV-cache. Multi-head self-attention. "
                       "O(L) per step due to growing KV-cache.",
        "gru": "Stacked GRU with LayerNorm + FFN. O(1) per step via hidden state. "
               "Simple recurrent baseline.",
        "conv": "Dilated causal Conv1D (WaveNet/TCN-style). O(1) per step via "
                "sliding window buffer. No recurrence.",
    }

    for name, cls in sorted(MODEL_REGISTRY.items()):
        table.add_row(name, cls.__name__, descriptions.get(name, ""))

    console.print()
    console.print(table)
    console.print()

    ptable = Table(title="Parameter Count (d_model=256, 4 layers)", box=box.SIMPLE)
    ptable.add_column("Architecture", style="bold")
    ptable.add_column("Parameters", justify="right")
    ptable.add_column("Memory (MiB)", justify="right")

    for name in sorted(MODEL_REGISTRY):
        try:
            m = create_model(name, d_model=256, num_layers=4, device="cpu")
            ptable.add_row(name, f"{m.count_parameters():,}",
                           f"{m.memory_footprint_mb():.2f}")
        except Exception as e:
            ptable.add_row(name, f"ERROR: {e}", "-")

    console.print(ptable)
    console.print()


if __name__ == "__main__":
    app()
