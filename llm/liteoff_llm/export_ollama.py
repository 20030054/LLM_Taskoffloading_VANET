"""Merge a LoRA adapter into the base model and register it in Ollama (plan 6.2, step 5).

    python -m liteoff_llm.export_ollama --base models/hf/Qwen2.5-0.5B-Instruct \
        --adapter models/adapters/student --name liteoff-student --quant q4_K_M q8_0 f16

Creates Ollama models  <name>-q4 (q4_K_M), <name>-q8 (q8_0), <name>-f16 (no quantisation) from the
merged safetensors directory using `ollama create --quantize`. Writes export.json with sizes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

SHORT = {"q4_K_M": "q4", "q4_0": "q4_0", "q8_0": "q8", "f16": "f16"}


def ollama_bin():
    return os.environ.get("OLLAMA_BIN") or shutil.which("ollama") or str(Path.home() / "sim/ollama/bin/ollama")


def merge(base, adapter, out):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model = model.to(torch.float16)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    return out


def create(name, src_dir, quant):
    mf = Path(src_dir) / "Modelfile"
    mf.write_text(f"FROM {Path(src_dir).resolve()}\nPARAMETER temperature 0\nPARAMETER num_predict 12\n")
    cmd = [ollama_bin(), "create", name, "-f", str(mf)]
    if quant != "f16":
        cmd += ["--quantize", quant]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ollama create failed for {name}:\n{r.stdout}\n{r.stderr}")
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="", help="LoRA adapter dir ('' = export the base model as is)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--quant", nargs="+", default=["q4_K_M", "f16"])
    ap.add_argument("--merged-dir", default="")
    a = ap.parse_args(argv)
    src = Path(a.base)
    if a.adapter:
        src = Path(a.merged_dir or f"models/merged/{a.name}")
        merge(a.base, a.adapter, src)
    info = {"name": a.name, "source": str(src), "models": {}}
    for q in a.quant:
        tag = f"{a.name}-{SHORT.get(q, q)}"
        create(tag, src, q)
        info["models"][tag] = q
        print(f"created ollama model {tag} ({q})", flush=True)
    if a.adapter:
        info["adapter_bytes"] = sum(f.stat().st_size for f in Path(a.adapter).glob("adapter_model*"))
    (Path(src) / "export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
