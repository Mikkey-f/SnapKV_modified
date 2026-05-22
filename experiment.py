"""
ThinkKV Experiment v3
===========================================================
功能：
1. 自动捕获 DeepSeek-R1 类 reasoning model 的 <think> block
2. 可视化：
   - answer -> think token attention heatmap
   - attention curve
3. 支持：
   - Head/Tail Hard Truncation
   - Attention-guided Eviction
4. MATH-500 pass@1 精度评测
5. 稳定支持 HF DynamicCache
6. 支持 flash_attention/eager 自动切换

===========================================================
使用方法：

# 可视化
python experiment_v3.py \
    --mode visualize \
    --model_path /root/autodl-tmp/models/DeepSeek-R1-8B

# 精度评测
python experiment_v3.py \
    --mode eval \
    --model_path /root/autodl-tmp/models/DeepSeek-R1-8B \
    --num_samples 50

# 两个都跑
python experiment.py \
    --mode both \
    --model_path /root/autodl-tmp/models/DeepSeek-R1-8B

===========================================================
"""

import os
import re
import json
import copy
import argparse

import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache


# ============================================================
# Config
# ============================================================

SYSTEM_PROMPT = (
    "Please reason step by step, "
    "and put your final answer within \\boxed{}."
)

VIS_PROBLEMS = [
    {
        "id": "vis_001",
        "question": "What is the sum of all positive integers less than 100 that are divisible by 3 or 5?",
        "answer": "2318"
    },
    {
        "id": "vis_002",
        "question": "If x + y = 10 and x^2 + y^2 = 60, what is the value of xy?",
        "answer": "20"
    },
    {
        "id": "vis_003",
        "question": "A train travels 120 km in 2 hours. How far does it travel in 5 hours at the same speed?",
        "answer": "300"
    },
]


# ============================================================
# Utils
# ============================================================

def build_prompt(question, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\n{question}"
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_boxed_answer(text):

    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if len(matches) == 0:
        return None

    return matches[-1].strip()


def normalize_answer(ans):

    if ans is None:
        return None

    ans = ans.replace(",", "")
    ans = ans.replace(" ", "")
    ans = ans.strip()

    return ans


def find_subsequence(lst, sub):

    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i+len(sub)] == sub:
            return i

    return None


def get_special_token_ids(tokenizer):

    think_start_ids = tokenizer.encode(
        "<think>",
        add_special_tokens=False
    )

    think_end_ids = tokenizer.encode(
        "</think>",
        add_special_tokens=False
    )

    return think_start_ids, think_end_ids


def kv_to_list(kv_cache):
    """
    统一把 KV cache 转成:
        List[(k, v)]

    兼容:
        - 老版 tuple cache
        - 新版 DynamicCache
        - HF 4.45+
    """

    # ---------------------------------------------------
    # DynamicCache
    # ---------------------------------------------------
    if isinstance(kv_cache, DynamicCache):

        kv_list = []

        # 新版 transformers
        try:

            for layer_idx in range(len(kv_cache)):

                layer = kv_cache.layers[layer_idx]

                k = layer.keys
                v = layer.values

                kv_list.append((k, v))

            return kv_list

        except Exception:
            pass

        # 中间版本 transformers
        try:

            for layer_idx in range(len(kv_cache)):

                k, v = kv_cache[layer_idx]

                kv_list.append((k, v))

            return kv_list

        except Exception:
            pass

        # 老版本 transformers
        try:

            for layer_idx in range(len(kv_cache)):

                k = kv_cache.key_cache[layer_idx]
                v = kv_cache.value_cache[layer_idx]

                kv_list.append((k, v))

            return kv_list

        except Exception:
            pass

    # ---------------------------------------------------
    # tuple/list cache
    # ---------------------------------------------------
    if isinstance(kv_cache, (tuple, list)):

        if len(kv_cache) > 0:

            first = kv_cache[0]

            if isinstance(first, (tuple, list)) and len(first) == 2:
                return [(x[0], x[1]) for x in kv_cache]

    raise ValueError(
        f"Unknown KV type: {type(kv_cache)}"
    )


def list_to_dynamic_cache(kv_list):

    cache = DynamicCache()

    for k, v in kv_list:
        cache.append(k, v)

    return cache


def get_cache_length(kv_cache):

    kv_list = kv_to_list(kv_cache)

    return kv_list[0][0].shape[2]


# ============================================================
# Generation
# ============================================================

def generate_full_reasoning(
    model,
    tokenizer,
    prompt,
    max_new_tokens=4096,
    device="cuda",
):

    think_start_ids, think_end_ids = get_special_token_ids(tokenizer)

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    prompt_len = input_ids.shape[1]

    generated = []

    past_key_values = None

    think_end_pos = None
    kv_at_think_end = None

    print(f"Prompt length: {prompt_len}")

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for step in range(max_new_tokens):

            next_token = next_logits.argmax(
                dim=-1,
                keepdim=True
            )

            token_id = next_token.item()

            generated.append(token_id)

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            current_ids = generated

            if (
                len(current_ids) >= len(think_end_ids)
                and current_ids[-len(think_end_ids):] == think_end_ids
                and kv_at_think_end is None
            ):

                think_end_pos = prompt_len + len(generated)

                kv_at_think_end = list_to_dynamic_cache([
                    (
                        k.clone(),
                        v.clone()
                    )
                    for k, v in kv_to_list(past_key_values)
                ])

                print(f"Detected </think> at {think_end_pos}")

            if token_id == tokenizer.eos_token_id:
                break

    full_ids = torch.cat([
        input_ids[0].cpu(),
        torch.tensor(generated)
    ]).tolist()

    think_start = find_subsequence(
        full_ids,
        think_start_ids
    )

    return {
        "generated_ids": generated,
        "full_ids": full_ids,
        "think_start": think_start,
        "think_end": think_end_pos,
        "kv_at_think_end": kv_at_think_end,
    }


# ============================================================
# Attention Analysis
# ============================================================

def collect_probe_tokens(
    model,
    tokenizer,
    kv_cache,
    max_tokens=16,
    device="cuda",
):

    _, think_end_ids = get_special_token_ids(tokenizer)

    next_input = torch.tensor(
        [[think_end_ids[-1]]],
        device=device
    )

    cur_kv = kv_cache

    tokens = []

    with torch.no_grad():

        for _ in range(max_tokens):

            outputs = model(
                input_ids=next_input,
                past_key_values=cur_kv,
                use_cache=True,
                return_dict=True,
            )

            cur_kv = outputs.past_key_values

            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True
            )

            token_id = next_token.item()

            tokens.append(token_id)

            next_input = next_token

            if token_id == tokenizer.eos_token_id:
                break

    return tokens


def compute_think_attention_scores(
    model,
    kv_cache,
    think_start,
    think_end,
    probe_tokens,
    device="cuda",
):

    cur_kv = kv_cache

    all_scores = []

    with torch.no_grad():

        for token_id in probe_tokens:

            input_ids = torch.tensor(
                [[token_id]],
                device=device
            )

            outputs = model(
                input_ids=input_ids,
                past_key_values=cur_kv,
                use_cache=True,
                return_dict=True,
                output_attentions=True,
            )

            cur_kv = outputs.past_key_values

            layer_scores = []

            for layer_attn in outputs.attentions:

                # (heads, kv_len)
                attn = layer_attn[0, :, 0, :].float()

                # max-head pooling
                attn = attn.max(dim=0).values

                think_attn = attn[
                    think_start:think_end
                ]

                layer_scores.append(
                    think_attn.cpu()
                )

            layer_scores = torch.stack(
                layer_scores
            ).mean(dim=0)

            all_scores.append(layer_scores)

    scores = torch.stack(all_scores).mean(dim=0)

    return scores.numpy()


# ============================================================
# KV Compression
# ============================================================

def sparse_select_kv(
    kv_cache,
    keep_indices
):

    kv_list = kv_to_list(kv_cache)

    keep_indices = sorted(
        list(set(keep_indices))
    )

    keep_indices = torch.tensor(
        keep_indices,
        dtype=torch.long,
    )

    new_kv = []

    for k, v in kv_list:

        idx = keep_indices.to(k.device)

        k_new = k[:, :, idx, :]
        v_new = v[:, :, idx, :]

        new_kv.append((k_new, v_new))

    return list_to_dynamic_cache(new_kv)


def hard_truncate_head_tail(
    kv_cache,
    think_start,
    think_end,
    keep_ratio=0.1,
):

    think_len = think_end - think_start

    keep_n = max(
        1,
        int(think_len * keep_ratio)
    )

    keep_indices = []

    # before think
    keep_indices.extend(
        range(think_start)
    )

    # head
    keep_indices.extend(
        range(
            think_start,
            think_start + keep_n
        )
    )

    # tail
    keep_indices.extend(
        range(
            think_end - keep_n,
            think_end
        )
    )

    total_len = get_cache_length(kv_cache)

    # after think
    keep_indices.extend(
        range(
            think_end,
            total_len
        )
    )

    print(
        f"[Hard Truncation] "
        f"think_len={think_len}, "
        f"keep_head_tail={keep_n}, "
        f"compression={(1 - 2*keep_ratio)*100:.1f}%"
    )

    return sparse_select_kv(
        kv_cache,
        keep_indices
    )


def attention_guided_eviction(
    kv_cache,
    think_start,
    think_end,
    scores,
    keep_ratio=0.1,
):

    think_len = think_end - think_start

    keep_n = max(
        1,
        int(think_len * keep_ratio)
    )

    topk_local = np.argsort(scores)[-keep_n:]

    topk_global = [
        think_start + int(i)
        for i in topk_local
    ]

    keep_indices = []

    keep_indices.extend(
        range(think_start)
    )

    keep_indices.extend(
        topk_global
    )

    total_len = get_cache_length(kv_cache)

    keep_indices.extend(
        range(
            think_end,
            total_len
        )
    )

    print(
        f"[Attention Eviction] "
        f"think_len={think_len}, "
        f"keep={keep_n}, "
        f"evict={think_len - keep_n}"
    )

    return sparse_select_kv(
        kv_cache,
        keep_indices
    )


# ============================================================
# Continue Generation
# ============================================================

def continue_generation(
    model,
    tokenizer,
    past_key_values,
    start_token_id,
    max_new_tokens=512,
    collect_attention=False,
    attention_window=32,
    device="cuda",
):

    generated = []

    next_input = torch.tensor(
        [[start_token_id]],
        device=device
    )

    attn_rows = []
    kv_len_start = None

    with torch.no_grad():

        for step in range(max_new_tokens):

            need_attn = (
                collect_attention
                and step < attention_window
            )

            outputs = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=need_attn,
            )

            past_key_values = outputs.past_key_values

            next_token = outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True
            )

            token_id = next_token.item()

            generated.append(token_id)

            if need_attn:

                layer_rows = []

                for layer_attn in outputs.attentions:

                    row = layer_attn[
                        0, :, 0, :
                    ].float().mean(dim=0)

                    layer_rows.append(
                        row.cpu()
                    )

                row = torch.stack(
                    layer_rows
                ).mean(dim=0).numpy()

                if kv_len_start is None:
                    kv_len_start = len(row)

                attn_rows.append(
                    row[:kv_len_start]
                )

            next_input = next_token

            if token_id == tokenizer.eos_token_id:
                break

    if len(attn_rows) > 0:
        attn_matrix = np.stack(attn_rows)
    else:
        attn_matrix = None

    return generated, attn_matrix


# ============================================================
# Visualization
# ============================================================

def plot_heatmap(
    attn_matrix,
    think_start,
    think_end,
    save_path,
):

    fig, ax = plt.subplots(
        figsize=(16, 5)
    )

    im = ax.imshow(
        attn_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )

    plt.colorbar(im, ax=ax)

    ax.axvline(
        x=think_start,
        color="red",
        linestyle="--",
        linewidth=1.5
    )

    ax.axvline(
        x=think_end,
        color="darkred",
        linestyle="--",
        linewidth=1.5
    )

    ax.set_title(
        "Answer -> Context Attention"
    )

    ax.set_xlabel(
        "Context Position"
    )

    ax.set_ylabel(
        "Generation Step"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved heatmap: {save_path}")


def plot_curve(
    attn_matrix,
    think_start,
    think_end,
    save_path,
):

    mean_attn = attn_matrix.mean(axis=0)

    x = np.arange(len(mean_attn))

    fig, ax = plt.subplots(
        figsize=(16, 4)
    )

    ax.plot(
        x,
        mean_attn,
        linewidth=0.8
    )

    ax.fill_between(
        x,
        mean_attn,
        alpha=0.3
    )

    ax.axvspan(
        think_start,
        think_end,
        alpha=0.15,
        color="red"
    )

    think_len = think_end - think_start

    keep_n = int(think_len * 0.1)

    ax.axvspan(
        think_start + keep_n,
        think_end - keep_n,
        alpha=0.2,
        color="orange"
    )

    ax.set_title(
        "Mean Attention by Position"
    )

    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved curve: {save_path}")


# ============================================================
# Dataset
# ============================================================

def load_math500(
    data_path=None,
    num_samples=50
):

    if data_path and os.path.exists(data_path):

        with open(data_path) as f:
            data = json.load(f)

        return data[:num_samples]

    try:

        from modelscope.msdatasets import MsDataset

        ds = MsDataset.load(
            "modelscope/MATH-500",
            split="test"
        )

        data = []

        for i, x in enumerate(ds):

            data.append({
                "id": f"math500_{i}",
                "question": x["problem"],
                "answer": x["answer"],
            })

        return data[:num_samples]

    except Exception as e:

        print("ModelScope load failed:", e)

    try:

        from datasets import load_dataset

        ds = load_dataset(
            "hendrycks/competition_math",
            split="test"
        )

        data = []

        for i, x in enumerate(ds):

            data.append({
                "id": f"math_{i}",
                "question": x["problem"],
                "answer": x["solution"],
            })

        return data[:num_samples]

    except Exception as e:

        print("HF datasets load failed:", e)

    return VIS_PROBLEMS


# ============================================================
# Visualization Experiment
# ============================================================

def run_visualization(
    model,
    tokenizer,
    args,
    device,
):

    os.makedirs(
        "heatmaps",
        exist_ok=True
    )

    _, think_end_ids = get_special_token_ids(
        tokenizer
    )

    think_end_token_id = think_end_ids[-1]

    for prob in VIS_PROBLEMS:

        print("\n" + "=" * 60)
        print(prob["id"])
        print("=" * 60)

        prompt = build_prompt(
            prob["question"],
            tokenizer
        )

        out = generate_full_reasoning(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        if out["kv_at_think_end"] is None:
            print("No </think> found")
            continue

        think_start = out["think_start"]
        think_end = out["think_end"]

        probe_tokens = collect_probe_tokens(
            model,
            tokenizer,
            out["kv_at_think_end"],
            max_tokens=16,
            device=device,
        )

        scores = compute_think_attention_scores(
            model,
            out["kv_at_think_end"],
            think_start,
            think_end,
            probe_tokens[:8],
            device=device,
        )

        if args.method == "hard":

            compressed_kv = hard_truncate_head_tail(
                out["kv_at_think_end"],
                think_start,
                think_end,
                keep_ratio=args.keep_ratio,
            )

        else:

            compressed_kv = attention_guided_eviction(
                out["kv_at_think_end"],
                think_start,
                think_end,
                scores,
                keep_ratio=args.keep_ratio,
            )

        _, attn_matrix = continue_generation(
            model,
            tokenizer,
            compressed_kv,
            think_end_token_id,
            max_new_tokens=256,
            collect_attention=True,
            attention_window=32,
            device=device,
        )

        plot_heatmap(
            attn_matrix,
            think_start,
            think_end,
            f"heatmaps/{prob['id']}_heatmap.png"
        )

        plot_curve(
            attn_matrix,
            think_start,
            think_end,
            f"heatmaps/{prob['id']}_curve.png"
        )


# ============================================================
# Eval
# ============================================================

def run_eval(
    model,
    tokenizer,
    args,
    device,
):

    os.makedirs(
        "results",
        exist_ok=True
    )

    problems = load_math500(
        args.data_path,
        args.num_samples
    )

    _, think_end_ids = get_special_token_ids(
        tokenizer
    )

    think_end_token_id = think_end_ids[-1]

    total = 0
    correct_full = 0
    correct_trunc = 0

    details = []

    for i, prob in enumerate(problems):

        print("\n" + "=" * 60)
        print(f"[{i+1}/{len(problems)}]")
        print(prob["question"][:80])
        print("=" * 60)

        prompt = build_prompt(
            prob["question"],
            tokenizer
        )

        out = generate_full_reasoning(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        if out["kv_at_think_end"] is None:
            print("No </think>")
            continue

        full_text = tokenizer.decode(
            out["generated_ids"],
            skip_special_tokens=False
        )

        full_answer = extract_boxed_answer(
            full_text
        )

        think_start = out["think_start"]
        think_end = out["think_end"]

        probe_tokens = collect_probe_tokens(
            model,
            tokenizer,
            out["kv_at_think_end"],
            max_tokens=16,
            device=device,
        )

        scores = compute_think_attention_scores(
            model,
            out["kv_at_think_end"],
            think_start,
            think_end,
            probe_tokens[:8],
            device=device,
        )

        if args.method == "hard":

            compressed_kv = hard_truncate_head_tail(
                out["kv_at_think_end"],
                think_start,
                think_end,
                keep_ratio=args.keep_ratio,
            )

        else:

            compressed_kv = attention_guided_eviction(
                out["kv_at_think_end"],
                think_start,
                think_end,
                scores,
                keep_ratio=args.keep_ratio,
            )

        trunc_ids, _ = continue_generation(
            model,
            tokenizer,
            compressed_kv,
            think_end_token_id,
            max_new_tokens=512,
            collect_attention=False,
            device=device,
        )

        trunc_text = tokenizer.decode(
            trunc_ids,
            skip_special_tokens=False
        )

        trunc_answer = extract_boxed_answer(
            trunc_text
        )

        gt = normalize_answer(
            prob["answer"]
        )

        pred_full = normalize_answer(
            full_answer
        )

        pred_trunc = normalize_answer(
            trunc_answer
        )

        ok_full = pred_full == gt
        ok_trunc = pred_trunc == gt

        if ok_full:
            correct_full += 1

        if ok_trunc:
            correct_trunc += 1

        total += 1

        print(
            f"GT={gt} | "
            f"Full={'✓' if ok_full else '✗'} | "
            f"Trunc={'✓' if ok_trunc else '✗'}"
        )

        details.append({
            "id": prob["id"],
            "gt": gt,
            "full_answer": full_answer,
            "trunc_answer": trunc_answer,
            "ok_full": ok_full,
            "ok_trunc": ok_trunc,
        })

    summary = {
        "total": total,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "accuracy_full": correct_full / total,
        "accuracy_trunc": correct_trunc / total,
        "accuracy_drop":
            (correct_full - correct_trunc) / total,
        "details": details,
    }

    save_path = (
        f"results/"
        f"{args.method}_keep"
        f"{int(args.keep_ratio*100)}.json"
    )

    with open(save_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Full KV: "
        f"{summary['accuracy_full']*100:.2f}%"
    )

    print(
        f"Compressed KV: "
        f"{summary['accuracy_trunc']*100:.2f}%"
    )

    print(
        f"Accuracy Drop: "
        f"{summary['accuracy_drop']*100:.2f}%"
    )

    print(f"Saved: {save_path}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=[
            "visualize",
            "eval",
            "both"
        ]
    )

    parser.add_argument(
        "--method",
        type=str,
        default="attention",
        choices=[
            "hard",
            "attention"
        ]
    )

    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=50
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    print("\nLoading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    model.eval()

    print("Model loaded.")

    if args.mode in ["visualize", "both"]:

        print("\n===== Visualization =====")

        run_visualization(
            model,
            tokenizer,
            args,
            device,
        )

    if args.mode in ["eval", "both"]:

        print("\n===== Evaluation =====")

        run_eval(
            model,
            tokenizer,
            args,
            device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()