"""
ThinkKV Experiment v3 - Fixed
===========================================================
核心修复：

【可视化】
  完整 KV → 生成 answer 前 window_size 步 → 收集 attention
  → 用原始坐标画热力图（红线位置正确）

【评测】
  完整 KV → 生成 answer 前 window_size 步 → 用这些步的
  attention scores 对 think block 打分 → 压缩 KV →
  继续生成剩余 answer → 和标答比对

===========================================================
使用方法：

# 可视化（默认 attention-guided）
python experiment_v3.py \
    --mode visualize \
    --model_path /root/autodl-tmp/models/DeepSeek-R1-8B

# 硬截断可视化
python experiment_v3.py \
    --mode visualize \
    --method hard \
    --model_path /root/autodl-tmp/models/DeepSeek-R1-8B

# 精度评测
python experiment_v3.py \
    --mode eval \
    --model_path /root/autodl-tmp/models/DeepSeek-R1-8B \
    --num_samples 50

# 两个都跑
python experiment_v3.py \
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

from transformers import AutoTokenizer, AutoModelForCausalLM
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
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_boxed_answer(text):
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans):
    if ans is None:
        return None
    return ans.replace(",", "").replace(" ", "").strip()


def find_subsequence(lst, sub):
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i+len(sub)] == sub:
            return i
    return None


def get_special_token_ids(tokenizer):
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids   = tokenizer.encode("</think>", add_special_tokens=False)
    return think_start_ids, think_end_ids


def kv_to_list(kv_cache):
    """统一转成 List[(k, v)]，兼容各版本 transformers"""
    if isinstance(kv_cache, DynamicCache):
        # 新版：key_cache / value_cache
        try:
            return list(zip(kv_cache.key_cache, kv_cache.value_cache))
        except Exception:
            pass
        # 老版：layers
        try:
            result = []
            for layer in kv_cache.layers:
                result.append((layer.keys, layer.values))
            return result
        except Exception:
            pass
        # 中间版：下标访问
        try:
            return [kv_cache[i] for i in range(len(kv_cache))]
        except Exception:
            pass

    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) > 0 and isinstance(kv_cache[0], (tuple, list)):
            return [(x[0], x[1]) for x in kv_cache]

    raise ValueError(f"Unknown KV type: {type(kv_cache)}")


def list_to_dynamic_cache(kv_list):
    """List[(k,v)] → DynamicCache"""
    cache = DynamicCache()
    try:
        for k, v in kv_list:
            cache.key_cache.append(k)
            cache.value_cache.append(v)
        return cache
    except Exception:
        pass
    try:
        cache2 = DynamicCache()
        for i, (k, v) in enumerate(kv_list):
            cache2.update(k, v, i)
        return cache2
    except Exception:
        pass
    raise RuntimeError("list_to_dynamic_cache failed")


def get_cache_length(kv_cache):
    kv_list = kv_to_list(kv_cache)
    return kv_list[0][0].shape[2]


def clone_kv(kv_cache):
    """深拷贝 KV cache，避免被后续生成污染"""
    kv_list = kv_to_list(kv_cache)
    return list_to_dynamic_cache([(k.clone(), v.clone()) for k, v in kv_list])


# ============================================================
# 完整推理生成
# ============================================================

def generate_full_reasoning(model, tokenizer, prompt, max_new_tokens=4096, device="cuda"):
    """
    完整生成 think block + answer，
    在检测到 </think> 时保存那一刻的 KV cache 快照。
    """
    think_start_ids, think_end_ids = get_special_token_ids(tokenizer)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated = []
    past_key_values = None
    think_end_pos = None
    kv_at_think_end = None

    print(f"  Prompt length: {prompt_len}")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for step in range(max_new_tokens):
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated.append(token_id)

            # 检测 </think>
            if (kv_at_think_end is None
                    and len(generated) >= len(think_end_ids)
                    and generated[-len(think_end_ids):] == think_end_ids):
                think_end_pos = prompt_len + len(generated)
                kv_at_think_end = clone_kv(past_key_values)
                print(f"  Detected </think> at pos {think_end_pos}, "
                      f"think block ≈ {len(generated)} tokens")

            if token_id == tokenizer.eos_token_id:
                print(f"  EOS at step {step+1}")
                break

            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

    full_ids = torch.cat([input_ids[0].cpu(), torch.tensor(generated)]).tolist()
    think_start = find_subsequence(full_ids, think_start_ids)

    return {
        "generated_ids": generated,
        "full_ids": full_ids,
        "think_start": think_start,
        "think_end": think_end_pos,
        "kv_at_think_end": kv_at_think_end,
        "final_kv": past_key_values,
    }


# ============================================================
# 核心：用完整 KV 生成 answer 前 N 步，收集 attention scores
# ============================================================

def generate_window_and_collect_attention(
    model,
    tokenizer,
    kv_at_think_end,   # 完整的 KV cache（think block 结束时）
    think_start,
    think_end,
    window_size=32,    # answer 前多少步用来打分
    device="cuda",
):
    """
    关键函数：
    1. 从完整 KV cache 出发，生成 answer 前 window_size 个 token
    2. 每一步都开启 output_attentions，收集对历史 token 的 attention
    3. 对 think block 内各 token 的被关注程度求和，作为重要性分数
    4. 返回：
       - window_generated_ids : 前 window_size 步生成的 token ids
       - attn_matrix          : np.array (window_size, full_kv_len)
                                用于画热力图，横轴是完整序列，红线坐标正确
       - think_scores         : np.array (think_len,)
                                think block 内每个 token 的重要性分数
       - window_kv            : 生成完 window 之后的 KV cache（用于后续截断）
    """
    _, think_end_ids = get_special_token_ids(tokenizer)
    start_token_id = think_end_ids[-1]

    next_input = torch.tensor([[start_token_id]], device=device)
    past_kv = kv_at_think_end   # 用完整 KV，不截断

    window_generated_ids = []
    attn_rows = []
    kv_len_start = None

    with torch.no_grad():
        for step in range(window_size):
            outputs = model(
                input_ids=next_input,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
                output_attentions=True,   # ← 关键：开启真实 attention
            )
            past_kv = outputs.past_key_values

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            window_generated_ids.append(next_token.item())

            # ── 收集 attention ──────────────────────────────
            # outputs.attentions: tuple(num_layers)
            # 每层: (batch=1, num_heads, q_len=1, kv_len)
            layer_rows = []
            for layer_attn in outputs.attentions:
                # 取所有 head 的平均，得到 (kv_len,)
                row = layer_attn[0, :, 0, :].float().mean(dim=0).cpu()
                layer_rows.append(row)

            # 所有层再平均，得到 (kv_len,)
            row = torch.stack(layer_rows).mean(dim=0)
            row = row / (row.sum() + 1e-8)   # 归一化
            row_np = row.numpy()

            if kv_len_start is None:
                kv_len_start = len(row_np)

            # 截断到初始长度（kv_len 会随生成增长）
            attn_rows.append(row_np[:kv_len_start])

            next_input = next_token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # attn_matrix: (steps, kv_len_at_start)  ← 对应完整序列的坐标
    attn_matrix = np.stack(attn_rows, axis=0) if attn_rows else np.zeros((1, 1))

    # think block 内各 token 的重要性：对 window 步取平均
    think_len = think_end - think_start
    if think_start is not None and kv_len_start and think_end <= kv_len_start:
        think_scores = attn_matrix[:, think_start:think_end].mean(axis=0)
    else:
        think_scores = np.ones(think_len) / think_len   # fallback 均匀

    return window_generated_ids, attn_matrix, think_scores, past_kv


# ============================================================
# KV 压缩
# ============================================================

def compress_kv(kv_cache, keep_indices):
    """按 keep_indices 选取 KV cache 的 token 维度"""
    kv_list = kv_to_list(kv_cache)
    keep_tensor = torch.tensor(sorted(set(keep_indices)), dtype=torch.long)
    new_kv = []
    for k, v in kv_list:
        idx = keep_tensor.to(k.device)
        new_kv.append((k[:, :, idx, :], v[:, :, idx, :]))
    return list_to_dynamic_cache(new_kv)


def hard_truncate(kv_cache, think_start, think_end, keep_ratio=0.1):
    """保留 think block 头尾各 keep_ratio，丢弃中间"""
    think_len = think_end - think_start
    keep_n = max(1, int(think_len * keep_ratio))
    total_len = get_cache_length(kv_cache)

    keep_indices = (
        list(range(think_start)) +                              # before
        list(range(think_start, think_start + keep_n)) +       # head
        list(range(think_end - keep_n, think_end)) +           # tail
        list(range(think_end, total_len))                      # after
    )
    removed = think_len - 2 * keep_n
    print(f"  [Hard] think_len={think_len}, keep={2*keep_n}, remove={removed} "
          f"({removed/think_len*100:.1f}%)")
    return compress_kv(kv_cache, keep_indices)


def attention_guided_eviction(kv_cache, think_start, think_end, scores, keep_ratio=0.1):
    """根据 attention scores 保留 think block 里最重要的 top-k token"""
    think_len = think_end - think_start
    keep_n = max(1, int(think_len * keep_ratio))
    total_len = get_cache_length(kv_cache)

    # top-k 局部下标 → 全局下标
    topk_local = np.argsort(scores)[-keep_n:]
    topk_global = [think_start + int(i) for i in topk_local]

    keep_indices = (
        list(range(think_start)) +   # before
        topk_global +                # top-k from think block
        list(range(think_end, total_len))   # after
    )
    evicted = think_len - keep_n
    print(f"  [Attn-guided] think_len={think_len}, keep={keep_n}, evict={evicted} "
          f"({evicted/think_len*100:.1f}%)")
    return compress_kv(kv_cache, keep_indices)


# ============================================================
# 压缩后继续生成
# ============================================================

def continue_generation(model, tokenizer, compressed_kv, window_last_token_id,
                         max_new_tokens=512, device="cuda"):
    """
    从压缩后的 KV cache 继续生成剩余 answer。
    start_token_id 是 window 最后生成的那个 token。
    """
    generated = []
    next_input = torch.tensor([[window_last_token_id]], device=device)
    past_kv = compressed_kv

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=next_input,
                past_key_values=past_kv,
                use_cache=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            token_id = next_token.item()
            generated.append(token_id)
            next_input = next_token
            if token_id == tokenizer.eos_token_id:
                break

    return generated


# ============================================================
# 可视化
# ============================================================

def plot_heatmap(attn_matrix, think_start, think_end, save_path, title="Answer -> Context Attention"):
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(attn_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, ax=ax)

    if think_start is not None:
        ax.axvline(x=think_start, color="red", linestyle="--", linewidth=1.5, label="<think>")
    if think_end is not None:
        ax.axvline(x=think_end, color="darkred", linestyle="--", linewidth=1.5, label="</think>")

    ax.set_title(title)
    ax.set_xlabel("Context Token Position (被关注的历史位置)")
    ax.set_ylabel("Answer Generation Step (答案生成步数)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {save_path}")


def plot_curve(attn_matrix, think_start, think_end, save_path, title="Mean Attention by Position"):
    mean_attn = attn_matrix.mean(axis=0)
    x = np.arange(len(mean_attn))

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(x, mean_attn, linewidth=0.8, color="steelblue")
    ax.fill_between(x, mean_attn, alpha=0.3, color="steelblue")

    if think_start is not None and think_end is not None and think_end <= len(mean_attn):
        ax.axvspan(think_start, think_end, alpha=0.15, color="red", label="think block")
        ax.axvline(x=think_start, color="red", linewidth=1.5, linestyle="--")
        ax.axvline(x=think_end, color="darkred", linewidth=1.5, linestyle="--", label="</think>")

        think_len = think_end - think_start
        keep_n = int(think_len * 0.1)
        if think_len > 20:
            ax.axvspan(think_start + keep_n, think_end - keep_n,
                       alpha=0.25, color="orange", label="中间80%（低attention区）")

    ax.set_title(title)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Mean Attention Score")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved curve: {save_path}")


def plot_think_scores(scores, save_path, title="Think Token Importance Scores"):
    """画 think block 内各 token 的重要性分数，用于直观验证 attention-guided 打分"""
    fig, ax = plt.subplots(figsize=(14, 3))
    x = np.arange(len(scores))
    ax.bar(x, scores, color="steelblue", width=1.0)
    ax.set_title(title)
    ax.set_xlabel("Position within Think Block")
    ax.set_ylabel("Importance Score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved scores: {save_path}")


# ============================================================
# 数据集
# ============================================================

def load_math500(data_path=None, num_samples=50):
    if data_path and os.path.exists(data_path):
        with open(data_path) as f:
            data = json.load(f)
        return data[:num_samples]

    try:
        from modelscope.msdatasets import MsDataset
        ds = MsDataset.load("modelscope/MATH-500", split="test")
        data = [{"id": f"math500_{i}", "question": x["problem"], "answer": x["answer"]}
                for i, x in enumerate(ds)]
        print(f"Loaded MATH-500: {len(data)} problems")
        return data[:num_samples]
    except Exception as e:
        print(f"ModelScope failed: {e}")

    try:
        from datasets import load_dataset
        ds = load_dataset("hendrycks/competition_math", split="test")
        data = [{"id": f"math_{i}", "question": x["problem"], "answer": x["solution"]}
                for i, x in enumerate(ds)]
        return data[:num_samples]
    except Exception as e:
        print(f"HF datasets failed: {e}")

    print("Using built-in problems")
    return VIS_PROBLEMS


# ============================================================
# 可视化实验
# ============================================================

def run_visualization(model, tokenizer, args, device):
    os.makedirs("heatmaps", exist_ok=True)

    for prob in VIS_PROBLEMS:
        print(f"\n{'='*55}")
        print(f"[VIS] {prob['id']}: {prob['question'][:60]}")
        print(f"{'='*55}")

        prompt = build_prompt(prob["question"], tokenizer)

        # Step 1: 完整生成，拿到 kv_at_think_end
        out = generate_full_reasoning(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens, device=device
        )

        if out["kv_at_think_end"] is None:
            print("  No </think> found, skip")
            continue

        think_start = out["think_start"]
        think_end   = out["think_end"]
        print(f"  think block: [{think_start}, {think_end}], "
              f"len={think_end - think_start}")

        # Step 2: 用完整 KV 生成 answer 前 window_size 步，收集 attention
        # ← 这里用 kv_at_think_end（完整），坐标是原始序列坐标
        print(f"  Collecting attention (window={args.window_size})...")
        window_ids, attn_matrix, think_scores, _ = generate_window_and_collect_attention(
            model, tokenizer,
            kv_at_think_end=out["kv_at_think_end"],
            think_start=think_start,
            think_end=think_end,
            window_size=args.window_size,
            device=device,
        )
        print(f"  attn_matrix shape: {attn_matrix.shape}")
        print(f"  think_scores shape: {think_scores.shape}, "
              f"max={think_scores.max():.4f}, min={think_scores.min():.4f}")

        # Step 3: 画热力图（用原始坐标，红线位置正确）
        plot_heatmap(
            attn_matrix, think_start, think_end,
            save_path=f"heatmaps/{prob['id']}_heatmap.png",
            title=f"Answer→Context Attention ({prob['id']}), "
                  f"window={args.window_size}"
        )
        plot_curve(
            attn_matrix, think_start, think_end,
            save_path=f"heatmaps/{prob['id']}_curve.png",
            title=f"Mean Attention by Position ({prob['id']})"
        )
        plot_think_scores(
            think_scores,
            save_path=f"heatmaps/{prob['id']}_scores.png",
            title=f"Think Token Importance ({prob['id']})"
        )


# ============================================================
# 精度评测
# ============================================================

def run_eval(model, tokenizer, args, device):
    os.makedirs("results", exist_ok=True)

    problems = load_math500(args.data_path, args.num_samples)
    _, think_end_ids = get_special_token_ids(tokenizer)

    total = 0
    correct_full = 0
    correct_trunc = 0
    details = []

    for i, prob in enumerate(problems):
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(problems)}] {prob['id']}: {prob['question'][:55]}...")
        print(f"{'='*55}")

        prompt = build_prompt(prob["question"], tokenizer)

        # Step 1: 完整生成
        out = generate_full_reasoning(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens, device=device
        )

        if out["kv_at_think_end"] is None:
            print("  No </think>, skip")
            continue

        think_start = out["think_start"]
        think_end   = out["think_end"]
        think_len   = think_end - think_start
        print(f"  think block len: {think_len}")

        # 完整答案
        full_text   = tokenizer.decode(out["generated_ids"], skip_special_tokens=False)
        full_answer = extract_boxed_answer(full_text)
        print(f"  Full answer: {full_answer}")

        # Step 2: 用完整 KV 生成 answer 前 window_size 步，收集 attention scores
        print(f"  Collecting attention window ({args.window_size} steps)...")
        window_ids, attn_matrix, think_scores, window_kv = generate_window_and_collect_attention(
            model, tokenizer,
            kv_at_think_end=out["kv_at_think_end"],
            think_start=think_start,
            think_end=think_end,
            window_size=args.window_size,
            device=device,
        )

        # Step 3: 压缩 window_kv（window 生成后的 KV，已包含 window tokens）
        # 注意：此时 KV 长度 = think_end + window_size
        # think block 的坐标没变，仍是 [think_start, think_end]
        if args.method == "hard":
            compressed_kv = hard_truncate(
                window_kv, think_start, think_end,
                keep_ratio=args.keep_ratio
            )
        else:
            compressed_kv = attention_guided_eviction(
                window_kv, think_start, think_end,
                think_scores, keep_ratio=args.keep_ratio
            )

        # Step 4: 从压缩后的 KV 继续生成剩余 answer
        last_window_token = window_ids[-1] if window_ids else think_end_ids[-1]
        trunc_ids = continue_generation(
            model, tokenizer, compressed_kv,
            window_last_token_id=last_window_token,
            max_new_tokens=512, device=device
        )

        # 把 window 生成的 token + 后续生成拼在一起
        all_trunc_ids = window_ids + trunc_ids
        trunc_text   = tokenizer.decode(all_trunc_ids, skip_special_tokens=False)
        trunc_answer = extract_boxed_answer(trunc_text)
        print(f"  Trunc answer: {trunc_answer}")

        # 判断对错
        gt         = normalize_answer(prob["answer"])
        pred_full  = normalize_answer(full_answer)
        pred_trunc = normalize_answer(trunc_answer)

        ok_full  = (pred_full  == gt)
        ok_trunc = (pred_trunc == gt)

        print(f"  GT={gt} | Full={'✓' if ok_full else '✗'} | "
              f"Trunc={'✓' if ok_trunc else '✗'}")

        if ok_full:  correct_full  += 1
        if ok_trunc: correct_trunc += 1
        total += 1

        details.append({
            "id": prob["id"],
            "gt": gt,
            "think_len": think_len,
            "full_answer": full_answer,
            "trunc_answer": trunc_answer,
            "ok_full": ok_full,
            "ok_trunc": ok_trunc,
        })

        if (i + 1) % 10 == 0:
            print(f"\n  [Progress] Full={correct_full}/{total} "
                  f"({correct_full/total*100:.1f}%) | "
                  f"Trunc={correct_trunc}/{total} "
                  f"({correct_trunc/total*100:.1f}%)\n")

    # 汇总
    acc_full  = correct_full  / total if total else 0
    acc_trunc = correct_trunc / total if total else 0

    print(f"\n{'='*55}")
    print(f"FINAL RESULTS  (n={total}, method={args.method}, "
          f"keep_ratio={args.keep_ratio}, window={args.window_size})")
    print(f"  Full KV  accuracy: {acc_full*100:.2f}%")
    print(f"  Trunc KV accuracy: {acc_trunc*100:.2f}%")
    print(f"  Accuracy drop:     {(acc_full-acc_trunc)*100:.2f}%")
    print(f"  Compression rate:  {(1-args.keep_ratio)*100:.0f}% of think block evicted")
    print(f"{'='*55}")

    summary = {
        "total": total,
        "method": args.method,
        "keep_ratio": args.keep_ratio,
        "window_size": args.window_size,
        "accuracy_full": acc_full,
        "accuracy_trunc": acc_trunc,
        "accuracy_drop": acc_full - acc_trunc,
        "details": details,
    }

    out_path = f"results/{args.method}_keep{int(args.keep_ratio*100)}_w{args.window_size}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

    # 柱状图
    _plot_bar(summary, out_path.replace(".json", "_bar.png"))


def _plot_bar(summary, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Full KV", f"Compressed KV\n({summary['method']}, "
                          f"keep={summary['keep_ratio']*100:.0f}%)"]
    values = [summary["accuracy_full"] * 100, summary["accuracy_trunc"] * 100]
    bars = ax.bar(labels, values, color=["steelblue", "coral"], width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Full vs Compressed KV (n={summary['total']})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["visualize", "eval", "both"])
    parser.add_argument("--method", type=str, default="attention",
                        choices=["hard", "attention"],
                        help="hard=头尾硬截断, attention=attention-guided eviction")
    parser.add_argument("--keep_ratio", type=float, default=0.1,
                        help="think block 保留比例（attention方法），默认10%")
    parser.add_argument("--window_size", type=int, default=32,
                        help="用 answer 前多少步来收集 attention scores")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",   # eager 支持 output_attentions
    )
    model.eval()
    print("Model loaded.\n")

    if args.mode in ["visualize", "both"]:
        print("===== Visualization =====")
        run_visualization(model, tokenizer, args, device)

    if args.mode in ["eval", "both"]:
        print("===== Evaluation =====")
        run_eval(model, tokenizer, args, device)

    print("\nAll done.")
    print("  Heatmaps: ./heatmaps/")
    print("  Results:  ./results/")


if __name__ == "__main__":
    main()