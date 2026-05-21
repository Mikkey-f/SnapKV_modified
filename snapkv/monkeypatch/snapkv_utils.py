#
# import torch
# import time
# import torch.nn.functional as F
# import torch.nn as nn
# import math
#
# # perform qk calculation and get indices
# # this version will not update in inference mode
#
# # Copied from transformers.models.llama.modeling_llama.repeat_kv
# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
#
# class SnapKVCluster():
#     def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
#         self.window_size = window_size
#         self.max_capacity_prompt = max_capacity_prompt
#         assert self.max_capacity_prompt - self.window_size > 0
#         self.kernel_size = kernel_size
#         self.pooling = pooling
#
#     def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
#         self.window_size = window_size
#         self.max_capacity_prompt = max_capacity_prompt
#         assert self.max_capacity_prompt - self.window_size > 0
#         self.kernel_size = kernel_size
#         self.pooling = pooling
#
#     def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
#         # check if prefix phase
#         assert key_states.shape[-2] == query_states.shape[-2]
#         bsz, num_heads, q_len, head_dim = query_states.shape
#         if q_len < self.max_capacity_prompt:
#             return key_states, value_states
#         else:
#             attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
#             mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
#             mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
#             mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#             mask = mask.to(attn_weights.device)
#             attention_mask = mask[None, None, :, :]
#
#             attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
#
#             attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#             attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
#             if self.pooling == 'avgpool':
#                 attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
#             elif self.pooling == 'maxpool':
#                 attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
#             else:
#                 raise ValueError('Pooling method not supported')
#             indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
#             indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
#             k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
#             v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
#             k_cur = key_states[:, :, -self.window_size:, :]
#             v_cur = value_states[:, :, -self.window_size:, :]
#             key_states = torch.cat([k_past_compress, k_cur], dim = 2)
#             value_states = torch.cat([v_past_compress, v_cur], dim = 2)
#             return key_states, value_states
#
# def init_snapkv(self):
#     if not hasattr(self, "kv_cluster"):
#         if not hasattr(self.config, 'window_size'):
#             self.config.window_size = 32
#         if not hasattr(self.config, 'max_capacity_prompt'):
#             self.config.max_capacity_prompt = 2048
#         if not hasattr(self.config, 'kernel_size'):
#             self.config.kernel_size = 5
#         if not hasattr(self.config, 'pooling'):
#             self.config.pooling = 'avgpool'
#     self.kv_cluster = SnapKVCluster(
#         window_size = self.config.window_size,
#         max_capacity_prompt = self.config.max_capacity_prompt,
#         kernel_size = self.config.kernel_size,
#         pooling = self.config.pooling
#         )
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# ============================================================
# 新增：可视化所需的库
# ============================================================
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 服务器无显示器时用这个后端
import numpy as np
import os


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# ============================================================
# 改动一：新增可视化函数
# 在 update_kv 调用之前或之后调用这个函数，传入 attn_weights 即可
# ============================================================
def visualize_attention_heatmap(
        attn_weights,  # shape: (batch, num_heads, q_len, kv_len)  softmax之后的
        save_path="attn_heatmap.png",
        title="Attention Heatmap",
        layer_idx=None,
        think_end_pos=None,  # </think> token 在序列中的位置，传入后会画一条红线标注
        max_heads_to_show=4,  # 最多展示几个 head，太多了图会很乱
):
    """
    把 attention score 矩阵可视化成热力图并保存。

    典型用法：
        在 update_kv 里算完 attn_weights 之后，调用：
        visualize_attention_heatmap(attn_weights, save_path="layer0_attn.png", layer_idx=0)
    """
    # 取第一个 batch，取前 max_heads_to_show 个 head
    attn_np = attn_weights[0].float().cpu().detach().numpy()  # (num_heads, q_len, kv_len)
    num_heads = min(attn_np.shape[0], max_heads_to_show)

    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]

    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attn_np[i], aspect='auto', cmap='viridis', vmin=0)
        ax.set_title(f"Head {i}")
        ax.set_xlabel("Key Position (被关注的历史token)")
        ax.set_ylabel("Query Position (当前生成的token)")
        plt.colorbar(im, ax=ax)

        # ======================================================
        # 如果传入了 think_end_pos，画一条红色竖线标注 </think> 的位置
        # 这样就能直观看到：生成答案时对 think block 内各位置的 attention 分布
        # ======================================================
        if think_end_pos is not None:
            ax.axvline(x=think_end_pos, color='red', linewidth=2, linestyle='--', label='</think>')
            ax.legend(fontsize=8)

    layer_str = f" - Layer {layer_idx}" if layer_idx is not None else ""
    fig.suptitle(f"{title}{layer_str}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SnapKV] Attention heatmap saved to: {save_path}")


class SnapKVCluster():
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool',
                 # ============================================================
                 # 改动二：新增两个参数控制 think block 截断行为
                 # ============================================================
                 enable_think_truncation=False,  # 是否开启中间80%截断
                 think_keep_ratio=0.1,  # 头尾各保留多少比例（默认各10%）
                 ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.enable_think_truncation = enable_think_truncation
        self.think_keep_ratio = think_keep_ratio

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool',
              enable_think_truncation=False, think_keep_ratio=0.1):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.enable_think_truncation = enable_think_truncation
        self.think_keep_ratio = think_keep_ratio

    # ============================================================
    # 改动三：新增 truncate_think_kv 方法
    # 这是 Gemini idea 的核心实现：保留 think block 的头尾，丢弃中间
    #
    # 用法：
    #   在检测到 </think> token 生成之后，在外部调用：
    #   key_states, value_states = kv_cluster.truncate_think_kv(
    #       key_states, value_states,
    #       think_start=0,       # <think> 在 kv cache 中的起始位置
    #       think_end=8000,      # </think> 在 kv cache 中的结束位置
    #   )
    # ============================================================
    def truncate_think_kv(self, key_states, value_states, think_start, think_end):
        """
        对 think block 范围内的 KV cache 做头尾保留、中间丢弃。

        key_states shape:   (batch, num_heads, seq_len, head_dim)
        think_start:        <think> 起始 token 在 kv cache 中的 index
        think_end:          </think> token 在 kv cache 中的 index

        操作逻辑：
            think block 长度 = think_end - think_start
            keep_n = int(长度 * think_keep_ratio)   # 头尾各保留 keep_n 个 token
            丢弃中间 (think_start + keep_n) 到 (think_end - keep_n) 的部分
        """
        think_len = think_end - think_start

        # think block 太短就不截断，避免把有效信息截没了
        if think_len < 50:
            print(f"[ThinkTruncate] think block too short ({think_len} tokens), skipping.")
            return key_states, value_states

        keep_n = max(1, int(think_len * self.think_keep_ratio))

        # 三段拼接：think之前 + think头部 + think尾部 + think之后（answer部分）
        before_think = key_states[:, :, :think_start, :]
        think_head = key_states[:, :, think_start: think_start + keep_n, :]
        think_tail = key_states[:, :, think_end - keep_n: think_end, :]
        after_think = key_states[:, :, think_end:, :]

        new_key_states = torch.cat([before_think, think_head, think_tail, after_think], dim=2)

        # value_states 做同样的操作
        before_think_v = value_states[:, :, :think_start, :]
        think_head_v = value_states[:, :, think_start: think_start + keep_n, :]
        think_tail_v = value_states[:, :, think_end - keep_n: think_end, :]
        after_think_v = value_states[:, :, think_end:, :]

        new_value_states = torch.cat([before_think_v, think_head_v, think_tail_v, after_think_v], dim=2)

        removed = think_len - 2 * keep_n
        print(
            f"[ThinkTruncate] think block: {think_len} tokens → kept {2 * keep_n} (head+tail), removed {removed} ({removed / think_len * 100:.1f}%)")

        return new_key_states, new_value_states

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups,
                  # ============================================================
                  # 改动四：update_kv 新增可视化参数，默认关闭不影响原有逻辑
                  # ============================================================
                  visualize=False,
                  vis_save_path="attn_heatmap.png",
                  vis_layer_idx=None,
                  think_end_pos=None,  # 用于在热力图上标注 </think> 位置
                  ):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :],
                                        key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min,
                              device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # ============================================================
            # 改动五：在 softmax 之后、topk 之前，插入可视化调用
            # 此时 attn_weights shape: (batch, heads, window_size, q_len)
            # 这是最完整的 attention 矩阵，最适合画热力图
            # ============================================================
            if visualize:
                visualize_attention_heatmap(
                    attn_weights,
                    save_path=vis_save_path,
                    title="SnapKV Attention Score",
                    layer_idx=vis_layer_idx,
                    think_end_pos=think_end_pos,
                )

            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2,
                                          stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2,
                                          stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling
    )