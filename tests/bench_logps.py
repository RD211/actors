# ruff: noqa
import argparse
import csv
import gc
import statistics
import sys
import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from actors.utils.get_logps import chunked_logp

# ---------------------------------------------------------------------------#
#  Log‑prob routines                                                          #
# ---------------------------------------------------------------------------#


@torch.no_grad()
def fast_logps(
    model,
    tokenizer,
    batch_ids: Sequence[Sequence[int]],
    temperature: float,
    max_fused: int,
    device,
) -> list[torch.Tensor]:
    lengths = [len(seq) for seq in batch_ids]

    enc = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt").to(
        device
    )

    hidden = model.model(  # type: ignore[attr-defined]
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        use_cache=False,
    ).last_hidden_state  # (B, L, H)

    hidden = hidden[:, :-1]
    target = enc.input_ids[:, 1:]

    L = enc.input_ids.size(1)
    non_pad = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(l - 1, dtype=torch.bool),
                    torch.zeros(L - l, dtype=torch.bool),
                ]
            )
            for l in lengths
        ],
        dim=0,
    ).to(device)

    h_flat = hidden.reshape(-1, hidden.size(-1))[non_pad.reshape(-1)]
    tgt_flat = target.reshape(-1)[non_pad.reshape(-1)]

    out_flat = chunked_logp(
        h_flat, model.lm_head, tgt_flat, max_fused=max_fused, temperature=temperature
    ).cpu()

    out, pos = [], 0
    for l in lengths:
        n = l - 1
        out.append(out_flat[pos : pos + n])
        pos += n

    return out


@torch.no_grad()
def baseline_logps(
    model,
    tokenizer,
    batch_ids: Sequence[Sequence[int]],
    temperature: float,
    device,
) -> list[torch.Tensor]:
    enc = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt").to(
        device
    )

    logits = model(**enc, use_cache=False).logits[:, :-1].float()
    logp = F.log_softmax(logits / temperature, dim=-1)
    tgt = enc.input_ids[:, 1:].unsqueeze(-1)
    gathered = torch.gather(logp, -1, tgt).squeeze(-1)

    lengths = [len(seq) for seq in batch_ids]
    return [gathered[i, : l - 1].detach().cpu() for i, l in enumerate(lengths)]


# ---------------------------------------------------------------------------#
#  Synthetic data                                                            #
# ---------------------------------------------------------------------------#
def synth_batch(
    tokenizer,
    batch_size: int,
    seq_len: int,
    include_bos: bool,
    seed: int,
) -> list[list[int]]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    bos_id = (
        tokenizer.bos_token_id
        if include_bos and tokenizer.bos_token_id is not None
        else None
    )

    vocab = tokenizer.vocab_size or len(tokenizer.get_vocab())
    body_len = seq_len - (1 if bos_id is not None else 0)

    rand_ids = torch.randint(0, vocab, (batch_size, body_len), generator=g)

    if bos_id is not None:
        ids = torch.cat(
            [torch.full((batch_size, 1), bos_id, dtype=torch.long), rand_ids],
            dim=1,
        )
    else:
        ids = rand_ids

    return ids.tolist()


# ---------------------------------------------------------------------------#
#  Benchmark                                                                 #
# ---------------------------------------------------------------------------#
def run_benchmark(
    model,
    tokenizer,
    seq_len: int,
    batch_size: int,
    temperature: float,
    max_fused: int,
    runs: int,
    warmup: int,
    device,
) -> tuple[dict, dict]:
    base_batch = synth_batch(tokenizer, batch_size, seq_len, True, 1234)

    def _measure(fn_fast: bool):
        walls, tps, mems = [], [], []

        for r in range(warmup + runs):
            gc.collect()
            torch.cuda.empty_cache()

            start_mem = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

            batch_ids = [list(x) for x in base_batch]

            start = time.perf_counter()
            if fn_fast:
                out = fast_logps(
                    model,
                    tokenizer,
                    batch_ids,
                    temperature,
                    max_fused,
                    device,
                )
            else:
                out = baseline_logps(model, tokenizer, batch_ids, temperature, device)
            torch.cuda.synchronize(device)
            end = time.perf_counter()

            walls.append(end - start)
            tps.append(sum(len(t) for t in out) / (end - start))
            mems.append(torch.cuda.max_memory_allocated(device) - start_mem)

        med = lambda v: statistics.median(v)
        return {
            "wall_s": med(walls),
            "tok_per_s": med(tps),
            "delta_mem": med(mems),
        }

    return _measure(True), _measure(False)


# ---------------------------------------------------------------------------#
#  Main                                                                       #
# ---------------------------------------------------------------------------#
def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--seqlens",
        type=int,
        nargs="*",
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_fused", type=int, default=1 << 16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--atol", type=float, default=1e-4)

    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": device}
    )
    model.eval()

    rows = []

    for L in args.seqlens:
        print(f"\n--- seq_len={L}  batch={args.batch_size} ---")

        fast_m, base_m = run_benchmark(
            model,
            tokenizer,
            L,
            args.batch_size,
            args.temperature,
            args.max_fused,
            args.runs,
            args.warmup,
            device,
        )

        fmt_mem = lambda x: f"{x / 1e6:7.0f} MB"

        print(f"{'method':>9s} | {'tok/s':>10s} | {'wall':>8s} | {'Δ-mem':>9s}")
        print("-" * 46)
        print(
            f"{'fast':>9s} | {fast_m['tok_per_s']:10.0f} | "
            f"{fast_m['wall_s']:8.2f}s | {fmt_mem(fast_m['delta_mem'])}"
        )
        print(
            f"{'baseline':>9s} | {base_m['tok_per_s']:10.0f} | "
            f"{base_m['wall_s']:8.2f}s | {fmt_mem(base_m['delta_mem'])}"
        )

        rows.extend(
            [
                {
                    "model": args.model,
                    "batch": args.batch_size,
                    "seq_len": L,
                    "method": "fast",
                    **fast_m,
                },
                {
                    "model": args.model,
                    "batch": args.batch_size,
                    "seq_len": L,
                    "method": "baseline",
                    **base_m,
                },
            ]
        )

        test_batch = synth_batch(tokenizer, args.batch_size, L, True, 4321)

        a = fast_logps(
            model,
            tokenizer,
            test_batch,
            args.temperature,
            args.max_fused,
            device,
        )
        b = baseline_logps(model, tokenizer, test_batch, args.temperature, device)

        max_abs = max(
            abs(float(x) - float(y))
            for u, v in zip(a, b, strict=False)
            for x, y in zip(u, v, strict=False)
        )

        if max_abs > args.atol:
            print(f"WARNING  max|Δ| = {max_abs:.2e}")
        else:
            print(f"outputs match  (max|Δ| = {max_abs:.1e})")

    if args.csv:
        fieldnames = [
            "model",
            "batch",
            "seq_len",
            "method",
            "wall_s",
            "tok_per_s",
            "delta_mem",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
