"""
实验脚本：Think Block KV 截断 + Attention 热力图可视化

实验目标：
    1. 可视化：生成答案时对 think block 各位置的 attention score 分布
    2. 截断对比：截断中间80% KV vs 不截断，答案质量是否有差异

使用方法：
    # 只跑可视化（不截断，先看现象）
    python experiment.py --mode visualize --model_path /path/to/DeepSeek-R1-Distill-Llama-8B

    # 跑截断对比
    python experiment.py --mode compare --model_path /path/to/DeepSeek-R1-Distill-Llama-8B

    # 两个都跑
    python experiment.py --mode both --model_path /path/to/DeepSeek-R1-Distill-Llama-8B

环境要求：
    pip install transformers==4.37.0 torch flash-attn matplotlib numpy
    （flash-attn 装不上的话见下面的 attn_implementation 注释）
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM

# experiment.py 在 SnapKV 目录下，直接把当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ================================================================
# 测试用的数学题（MATH 类型，会触发较长的 think block）
# ================================================================
TEST_PROBLEMS = [
    {
        "id": "prob_001",
        "question": "What is the sum of all positive integers less than 100 that are divisible by 3 or 5?",
        "answer": "2318"
    },
    {
        "id": "prob_002",
        "question": "Find the number of ways to arrange the letters in the word 'MISSISSIPPI'.",
        "answer": "34650"
    },
    {
        "id": "prob_003",
        "question": "If x + y = 10 and x^2 + y^2 = 60, what is the value of xy?",
        "answer": "20"
    },
]

SYSTEM_PROMPT = "You are a helpful math assistant. Think step by step."


def find_think_boundaries(token_ids, tokenizer):
    """
    在 token id 序列里找 <think> 和 </think> 的位置。
    返回 (think_start, think_end)，找不到返回 (None, None)
    """
    # 把整个序列 decode 成文本，找字符位置，再映射回 token 位置
    # 更简单的方法：直接用 tokenizer encode 特殊 token
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_end_ids   = tokenizer.encode("</think>", add_special_tokens=False)

    def find_sublist(lst, sublst):
        """在 lst 里找 sublst 第一次出现的起始位置"""
        n, m = len(lst), len(sublst)
        for i in range(n - m + 1):
            if lst[i:i+m] == sublst:
                return i
        return None

    ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)

    start_pos = find_sublist(ids, think_start_ids)
    end_pos   = find_sublist(ids, think_end_ids)

    if start_pos is not None and end_pos is not None:
        return start_pos, end_pos + len(think_end_ids)
    return None, None


def build_prompt(question, tokenizer):
    """构建符合 DeepSeek-R1 格式的 prompt"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    # apply_chat_template 会自动加上 <think> 开始标记
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def generate_with_monitoring(model, tokenizer, prompt, max_new_tokens=2048, device="cuda"):
    """
    手动 token-by-token 生成，监测 </think> 出现的位置。
    返回：
        generated_ids     : 完整生成的 token id 列表
        think_start_pos   : <think> 在完整序列（prompt+生成）里的位置
        think_end_pos     : </think> 在完整序列里的位置
        past_key_values   : 生成完 think block 之后那一刻的 KV cache
    """
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_end_id  = think_end_ids[-1]  # 通常是单个 token

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    generated_ids = []
    past_key_values = None
    think_end_pos = None
    kv_at_think_end = None  # </think> 那一刻的 KV cache 快照

    print(f"  [生成中] prompt 长度: {prompt_len} tokens, 最大生成: {max_new_tokens} tokens")

    with torch.no_grad():
        # 先跑一次 prefill 拿到初始 KV cache
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

        for step in range(max_new_tokens):
            # 采样下一个 token（greedy）
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token_id.item())

            # 检测是否是 </think>
            if next_token_id.item() == think_end_id and think_end_pos is None:
                think_end_pos = prompt_len + step + 1

                # 必须 deepcopy，否则 DynamicCache 会继续被后续 generation 污染
                kv_at_think_end = copy.deepcopy(past_key_values)

                print(f"  [检测到 </think>] 位置: {think_end_pos}, think block 长度约 {step} tokens")
                print(f"  KV cache 类型: {type(kv_at_think_end)}")

                # 调试输出
                if hasattr(kv_at_think_end, "layers"):
                    print(f"  layers 数量: {len(kv_at_think_end.layers)}")

            # EOS 停止
            if next_token_id.item() == tokenizer.eos_token_id:
                print(f"  [生成结束] 共生成 {step+1} tokens")
                break

            # 继续生成
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

    full_ids = torch.cat([input_ids[0].cpu(), torch.tensor(generated_ids)], dim=0)

    # 找 think block 边界
    think_start_pos, _ = find_think_boundaries(full_ids, tokenizer)

    return generated_ids, think_start_pos, think_end_pos, kv_at_think_end, past_key_values


def normalize_kv_cache(kv_cache):
    """
    强制兼容 transformers 所有 KV cache 格式
    最终统一返回:

        [(k,v), (k,v), ...]

    """

    import torch

    # ==========================================================
    # 情况1: 新版 DynamicCache
    # ==========================================================
    if hasattr(kv_cache, "layers"):

        out = []

        for idx, layer in enumerate(kv_cache.layers):

            # 调试
            print(f"[Layer {idx}] type = {type(layer)}")

            # keys / values
            if hasattr(layer, "keys") and hasattr(layer, "values"):

                k = layer.keys
                v = layer.values

                print("  using keys/values")

                out.append((k, v))
                continue

            # key_states / value_states
            if hasattr(layer, "key_states") and hasattr(layer, "value_states"):

                k = layer.key_states
                v = layer.value_states

                print("  using key_states/value_states")

                out.append((k, v))
                continue

            # key_cache / value_cache
            if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):

                k = layer.key_cache
                v = layer.value_cache

                print("  using key_cache/value_cache")

                out.append((k, v))
                continue

            # 暴力反射 tensor
            tensor_attrs = []

            for attr in dir(layer):

                if attr.startswith("_"):
                    continue

                try:
                    val = getattr(layer, attr)

                    if isinstance(val, torch.Tensor):
                        tensor_attrs.append((attr, val))

                except:
                    pass

            print("  tensor attrs =", [x[0] for x in tensor_attrs])

            if len(tensor_attrs) >= 2:

                k = tensor_attrs[0][1]
                v = tensor_attrs[1][1]

                print(f"  fallback tensor attrs: {tensor_attrs[0][0]}, {tensor_attrs[1][0]}")

                out.append((k, v))
                continue

            raise ValueError(f"无法解析 layer: {type(layer)}")

        return out

    # ==========================================================
    # 情况2: 老 tuple
    # ==========================================================
    if isinstance(kv_cache, tuple):

        # ((k,v), ...)
        if len(kv_cache) > 0 and isinstance(kv_cache[0], tuple):

            return [(x[0], x[1]) for x in kv_cache]

    # ==========================================================
    # 情况3: key_cache/value_cache
    # ==========================================================
    if hasattr(kv_cache, "key_cache"):

        return list(zip(
            kv_cache.key_cache,
            kv_cache.value_cache
        ))

    # ==========================================================
    # 实在不行
    # ==========================================================
    print("kv_cache dir =", dir(kv_cache))

    raise ValueError(
        f"彻底无法解析 KV cache: {type(kv_cache)}"
    )


def visualize_kv_attention(kv_cache, think_start, think_end, save_dir="heatmaps", problem_id=""):
    os.makedirs(save_dir, exist_ok=True)

    # 【这里直接使用终极清洗函数】
    legacy_kv = normalize_kv_cache(kv_cache)

    num_layers = len(legacy_kv)
    layers_to_plot = [0, num_layers // 4, num_layers // 2, num_layers - 1]

    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(16, 4 * len(layers_to_plot)))
    if len(layers_to_plot) == 1:
        axes = [axes]

    for plot_idx, layer_idx in enumerate(layers_to_plot):
        k, v = legacy_kv[layer_idx]

        key_norm = k[0].float().norm(dim=-1).mean(dim=0).cpu().numpy()  # (seq_len,)
        seq_len = len(key_norm)
        positions = np.arange(seq_len)

        ax = axes[plot_idx]
        ax.plot(positions, key_norm, color='steelblue', linewidth=0.8, alpha=0.8)
        ax.fill_between(positions, key_norm, alpha=0.3, color='steelblue')

        if think_start is not None and think_end is not None and think_end <= seq_len:
            think_start_c = min(think_start, seq_len - 1)
            think_end_c = min(think_end, seq_len - 1)

            ax.axvspan(think_start_c, think_end_c, alpha=0.15, color='red', label='think block')
            ax.axvline(x=think_start_c, color='red', linewidth=1.5, linestyle='--')
            ax.axvline(x=think_end_c, color='darkred', linewidth=1.5, linestyle='--', label='</think>')

            think_len = think_end_c - think_start_c
            keep_n = int(think_len * 0.1)
            mid_start = think_start_c + keep_n
            mid_end = think_end_c - keep_n
            if mid_end > mid_start:
                ax.axvspan(mid_start, mid_end, alpha=0.2, color='orange', label='中间80%(待截断)')

        ax.set_title(f"Layer {layer_idx} - Key Norm Distribution")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Key L2 Norm (avg over heads)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"KV Cache 重要性分布 - {problem_id}", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{problem_id}_kv_norm.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [可视化] 热力图保存至: {save_path}")
    return save_path

def visualize_attention_scores(
    all_attn_scores,
    think_start,
    think_end,
    save_path="attention_heatmap.png",
):
    """
    可视化：
    answer generation 时
    对历史 token 的 attention

    横轴:
        context token position

    纵轴:
        answer generation step
    """

    import numpy as np
    import matplotlib.pyplot as plt

    num_steps = len(all_attn_scores)

    if num_steps == 0:
        print("[警告] 没有 attention 数据")
        return

    num_layers = len(all_attn_scores[0])

    layers_to_plot = [
        0,
        num_layers // 4,
        num_layers // 2,
        num_layers - 1
    ]

    fig, axes = plt.subplots(
        len(layers_to_plot),
        1,
        figsize=(16, 4 * len(layers_to_plot))
    )

    if len(layers_to_plot) == 1:
        axes = [axes]

    for idx, layer_idx in enumerate(layers_to_plot):

        mat = []

        for step_scores in all_attn_scores:

            scores = step_scores[layer_idx].numpy()

            mat.append(scores)

        max_len = max(len(x) for x in mat)

        padded = []

        for x in mat:

            if len(x) < max_len:
                pad_width = max_len - len(x)

                x = np.pad(
                    x,
                    (0, pad_width),
                    mode='constant',
                    constant_values=0,
                )

            padded.append(x)

        mat = np.stack(padded, axis=0)

        ax = axes[idx]

        im = ax.imshow(
            mat,
            aspect='auto',
            interpolation='nearest',
        )

        # think block 边界
        ax.axvline(
            x=think_start,
            color='red',
            linestyle='--',
            linewidth=1.5
        )

        ax.axvline(
            x=think_end,
            color='darkred',
            linestyle='--',
            linewidth=1.5
        )

        ax.set_title(f"Layer {layer_idx} Attention Heatmap")

        ax.set_xlabel("Context Token Position")

        ax.set_ylabel("Answer Generation Step")

        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.close()

    print(f"[保存] Attention Heatmap -> {save_path}")

def truncate_think_kv(past_key_values, think_start, think_end, keep_ratio=0.1):
    if think_start is None or think_end is None:
        print("  [截断] 未找到 think block 边界，跳过截断")
        return past_key_values

    think_len = think_end - think_start
    if think_len < 50:
        print(f"  [截断] think block 太短 ({think_len} tokens)，跳过截断")
        return past_key_values

    keep_n = max(1, int(think_len * keep_ratio))
    removed = think_len - 2 * keep_n
    print(
        f"  [截断] think block {think_len} tokens → 保留头尾各 {keep_n}，截断中间 {removed} tokens ({removed / think_len * 100:.1f}%)")

    legacy_kv = normalize_kv_cache(past_key_values)
    new_kv = []

    for k, v in legacy_kv:
        # k/v shape: (batch, num_heads, seq_len, head_dim)
        before = k[:, :, :think_start, :]
        head_ = k[:, :, think_start:think_start + keep_n, :]
        tail_ = k[:, :, think_end - keep_n:think_end, :]
        after = k[:, :, think_end:, :]
        new_k = torch.cat([before, head_, tail_, after], dim=2)

        before_v = v[:, :, :think_start, :]
        head_v = v[:, :, think_start:think_start + keep_n, :]
        tail_v = v[:, :, think_end - keep_n:think_end, :]
        after_v = v[:, :, think_end:, :]
        new_v = torch.cat([before_v, head_v, tail_v, after_v], dim=2)

        new_kv.append((new_k, new_v))

    legacy_tuple = tuple(new_kv)

    # =========================================================
    # 强制转回 DynamicCache
    # transformers >= 4.44 必须这样
    # =========================================================
    try:
        from transformers.cache_utils import DynamicCache

        new_cache = DynamicCache()

        for layer_idx, (k, v) in enumerate(legacy_tuple):
            new_cache.update(
                key_states=k,
                value_states=v,
                layer_idx=layer_idx,
            )

        return new_cache

    except Exception as e:
        print(f"[警告] DynamicCache 重建失败: {e}")

        # fallback
        if hasattr(past_key_values.__class__, "from_legacy_cache"):
            return past_key_values.__class__.from_legacy_cache(legacy_tuple)

        return legacy_tuple


def continue_generation(
    model,
    tokenizer,
    last_token_id,
    past_key_values,
    think_start,
    think_end,
    max_new_tokens=512,
    device="cuda",
):
    """
    从截断后的 KV cache 继续生成答案
    并记录 answer -> context attention
    """

    generated_ids = []

    next_input = torch.tensor([[last_token_id]], device=device)

    # =========================================================
    # 保存 attention
    #
    # all_attn_scores[step][layer]
    # -> tensor(seq_len,)
    # =========================================================
    all_attn_scores = []

    with torch.no_grad():

        for step in range(max_new_tokens):

            outputs = model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=True,   # <<< 新增
            )

            past_key_values = outputs.past_key_values

            # =====================================================
            # attentions:
            # tuple(num_layers)
            #
            # each:
            # (batch, num_heads, q_len=1, kv_len)
            # =====================================================

            attentions = outputs.attentions

            if attentions is None:
                print("[错误] attentions=None，当前 attention backend 不支持 output_attentions")
                break

            layer_scores = []

            for layer_attn in attentions:

                # shape:
                # (num_heads, kv_len)
                attn = layer_attn[0, :, 0, :]

                # 平均所有 head
                attn_mean = attn.mean(dim=0).float().cpu()

                layer_scores.append(attn_mean)

            all_attn_scores.append(layer_scores)

            # greedy decode
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            generated_ids.append(next_token_id.item())

            next_input = next_token_id

            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated_ids, all_attn_scores


def run_experiment(args):
    print("=" * 60)
    print("加载模型...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print("模型加载完成\n")

    os.makedirs("results", exist_ok=True)
    os.makedirs("heatmaps", exist_ok=True)

    # for prob in TEST_PROBLEMS:
    for prob in gsm8k dataset:
        print(f"\n{'='*60}")
        print(f"问题: {prob['id']} - {prob['question'][:50]}...")
        print(f"标准答案: {prob['answer']}")
        print(f"{'='*60}")

        prompt = build_prompt(prob['question'], tokenizer)

        # --------------------------------------------------------
        # 步骤一：完整生成（不截断），拿到 think block 位置和 KV cache
        # --------------------------------------------------------
        print("\n[步骤1] 完整生成（baseline）...")
        gen_ids, think_start, think_end, kv_at_think_end, final_kv = generate_with_monitoring(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        print(f"\n--- 完整输出 ---\n{full_text[:500]}...\n")

        # --------------------------------------------------------
        # 步骤二：可视化
        # --------------------------------------------------------
        if args.mode in ["visualize", "both"]:
            print("\n[步骤2] 生成 attention 热力图...")
            if kv_at_think_end is not None:
                visualize_kv_attention(
                    kv_at_think_end,
                    think_start=think_start,
                    think_end=think_end,
                    save_dir="heatmaps",
                    problem_id=prob['id'],
                )
            else:
                print("  未检测到 </think>，跳过可视化（模型可能未使用 think 格式）")

        # --------------------------------------------------------
        # 步骤三：截断对比
        # --------------------------------------------------------
        if args.mode in ["compare", "both"] and kv_at_think_end is not None:
            print("\n[步骤3] 截断 KV cache，重新生成答案...")

            # 用 </think> 时刻的 KV cache 做截断
            truncated_kv = truncate_think_kv(
                kv_at_think_end,
                think_start=think_start,
                think_end=think_end,
                keep_ratio=args.keep_ratio,
            )

            # 从 </think> token 继续生成
            think_end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
            # truncated_gen_ids  = continue_generation(
            #     model, tokenizer,
            #     last_token_id=think_end_token_id,
            #     past_key_values=truncated_kv,
            #     max_new_tokens=512,
            #     device=device,
            # )
            truncated_gen_ids, attn_scores = continue_generation(
                model,
                tokenizer,
                last_token_id=think_end_token_id,
                past_key_values=truncated_kv,
                think_start=think_start,
                think_end=think_end,
                max_new_tokens=512,
                device=device,
            )
            truncated_text = tokenizer.decode(truncated_gen_ids, skip_special_tokens=False)
            # =====================================================
            # Attention Heatmap
            # =====================================================
            visualize_attention_scores(
                attn_scores,
                think_start,
                think_end,
                save_path=f"heatmaps/{prob['id']}_attention.png"
            )
            print(f"\n--- 截断后输出 ---\n{truncated_text[:500]}\n")

            # 简单评估：看答案里有没有标准答案的数字
            correct_baseline  = prob['answer'] in full_text
            correct_truncated = prob['answer'] in truncated_text

            print(f"标准答案: {prob['answer']}")
            print(f"完整生成 包含正确答案: {'✓' if correct_baseline  else '✗'}")
            print(f"截断生成 包含正确答案: {'✓' if correct_truncated else '✗'}")

            # 保存结果
            result_path = f"results/{prob['id']}_comparison.txt"
            with open(result_path, "w") as f:
                f.write(f"问题: {prob['question']}\n")
                f.write(f"标准答案: {prob['answer']}\n")
                f.write(f"Think block: [{think_start}, {think_end}]\n\n")
                f.write(f"=== 完整生成 ===\n{full_text}\n\n")
                f.write(f"=== 截断生成 ===\n{truncated_text}\n\n")
                f.write(f"正确性 - 完整: {correct_baseline}, 截断: {correct_truncated}\n")
            print(f"结果保存至: {result_path}")

    print("\n\n实验完成！")
    print("  热力图: ./heatmaps/")
    print("  对比结果: ./results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="模型路径或 HuggingFace model id"
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["visualize", "compare", "both"],
        default="both",
        help="visualize=只看热力图, compare=只跑截断对比, both=都跑"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="最大生成 token 数（think block + 答案）"
    )
    parser.add_argument(
        "--keep_ratio", type=float, default=0.1,
        help="think block 头尾各保留的比例（默认0.1即各10%）"
    )
    args = parser.parse_args()
    run_experiment(args)