"""LoRA fine-tuning of the student model on offloading decisions (plan 6.2, step 3).

Plain PyTorch loop (robust to transformers API changes), CPU or GPU, loss on answer tokens only.
    python -m liteoff_llm.train_lora --base models/hf/Qwen2.5-0.5B-Instruct \
        --train data/teacher/distilled.jsonl --val data/sft/val.jsonl --out models/adapters/student
Writes the PEFT adapter, tokenizer, and train_log.json (loss curve, val loss, val exact-match).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoder import parse_answer


def load_jsonl(paths, limit=0):
    out = []
    for p in paths:
        with open(p) as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out[:limit] if limit else out


def build(tok, ex, max_len):
    msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    prompt_ids = list(prompt_ids)
    ans_ids = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + ans_ids)[-max_len:]
    labels = ([-100] * len(prompt_ids) + ans_ids)[-max_len:]
    return ids, labels


def batches(items, bs, pad_id, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [items[j] for j in idx[i:i + bs]]
        L = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, (a, b) in enumerate(chunk):
            ids[r, :len(a)] = torch.tensor(a)
            lab[r, :len(b)] = torch.tensor(b)
            att[r, :len(a)] = 1
        yield ids, lab, att


@torch.no_grad()
def eval_loss(model, data, bs, pad_id, device):
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, att in batches(data, bs, pad_id, False, None):
        out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        tot += float(out.loss) * ids.shape[0]
        n += ids.shape[0]
    model.train()
    return tot / max(1, n)


@torch.no_grad()
def eval_accuracy(model, tok, exs, device, n=40):
    model.eval()
    hit = parsed = 0
    for ex in exs[:n]:
        msgs = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(device)
        out = model.generate(input_ids=ids, attention_mask=enc["attention_mask"].to(device), max_new_tokens=12,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        ok, slot, _ = parse_answer(txt, k=ex["meta"]["k"])
        parsed += ok
        hit += ok and slot == ex["meta"]["slot"]
    model.train()
    m = max(1, min(n, len(exs)))
    return hit / m, parsed / m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--eval-gen", type=int, default=40)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dtype).to(device)
    model.config.use_cache = False
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    tr_ex = load_jsonl(a.train, a.limit)
    va_ex = load_jsonl(a.val, a.val_limit) if a.val else []
    tr = [build(tok, e, a.max_len) for e in tr_ex]
    va = [build(tok, e, a.max_len) for e in va_ex]
    steps_per_epoch = math.ceil(len(tr) / (a.batch * a.grad_accum))
    total_steps = a.max_steps or max(1, int(steps_per_epoch * a.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total_steps // 10)) *
                                              0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"device={device} dtype={dtype} train={len(tr)} val={len(va)} trainable={trainable / 1e6:.2f}M/{total / 1e6:.0f}M "
          f"steps={total_steps} mean_len={sum(len(x[0]) for x in tr) / max(1, len(tr)):.0f}", flush=True)
    log = {"args": vars(a), "device": device, "trainable_params": trainable, "steps": [], "evals": []}
    if va:
        vl = eval_loss(model, va, a.batch, tok.pad_token_id, device)
        acc, pr = eval_accuracy(model, tok, va_ex, device, a.eval_gen)
        log["evals"].append({"step": 0, "val_loss": vl, "val_acc": acc, "val_parse": pr})
        print(f"step 0 val_loss={vl:.4f} val_acc={acc:.3f} parse={pr:.3f}", flush=True)
    model.train()
    step, t0 = 0, time.time()
    done = False
    while not done:
        accum, run_loss = 0, 0.0
        for ids, lab, att in batches(tr, a.batch, tok.pad_token_id, True, rng):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
            (out.loss / a.grad_accum).backward()
            run_loss += float(out.loss)
            accum += 1
            if accum < a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            loss = run_loss / accum
            accum, run_loss = 0, 0.0
            log["steps"].append({"step": step, "loss": loss, "lr": sched.get_last_lr()[0], "t": time.time() - t0})
            if step % 5 == 0 or step == 1:
                print(f"step {step}/{total_steps} loss={loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if va and (step % a.eval_every == 0 or step == total_steps):
                vl = eval_loss(model, va, a.batch, tok.pad_token_id, device)
                acc, pr = eval_accuracy(model, tok, va_ex, device, a.eval_gen)
                log["evals"].append({"step": step, "val_loss": vl, "val_acc": acc, "val_parse": pr})
                print(f"step {step} val_loss={vl:.4f} val_acc={acc:.3f} parse={pr:.3f}", flush=True)
            if step >= total_steps:
                done = True
                break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    size = sum(f.stat().st_size for f in out.glob("adapter_model*"))
    log["adapter_bytes"] = size
    log["train_time_s"] = time.time() - t0
    first = [s["loss"] for s in log["steps"][:5]]
    last = [s["loss"] for s in log["steps"][-5:]]
    log["loss_first5"] = sum(first) / max(1, len(first))
    log["loss_last5"] = sum(last) / max(1, len(last))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved adapter to {out} ({size / 1e6:.1f} MB); loss {log['loss_first5']:.3f} -> {log['loss_last5']:.3f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
