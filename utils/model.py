import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_normed * self.scale


class Head(nn.Module):
    def __init__(self, d_embd: int, d_k: int, block_size: int, drop_rate: float):
        super().__init__()

        self.query = nn.Linear(d_embd, d_k, bias=False)
        self.key = nn.Linear(d_embd, d_k, bias=False)
        self.value = nn.Linear(d_embd, d_k, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(drop_rate)

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        att = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5
        att = att.masked_fill(self.mask == 0, float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))

        return att @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.query(x), self.key(x), self.value(x))


class MultiHeadAttention(nn.Module):
    def __init__(
        self, block_size: int, d_embd: int, d_head: int, n_heads: int, drop_rate: float
    ) -> None:
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(d_embd, d_head, block_size, drop_rate) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(d_head * n_heads, d_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, d: int, d_hidden: int):
        super().__init__()

        self.fc1 = nn.Linear(d, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


class NoisyTopKGating(nn.Module):
    def __init__(self, d: int, n_experts: int, k_top: int) -> None:
        super().__init__()

        self.k_top = k_top

        self.gate = nn.Linear(d, n_experts)
        self.var = nn.Sequential(nn.Linear(d, n_experts), nn.Softplus())

    @staticmethod
    def _keep_topk(
        v: torch.Tensor, k: int, dim: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk = torch.topk(v, k=k, dim=dim, sorted=False)

        topk_mask = (topk.indices.unsqueeze(dim=-1) == torch.arange(v.shape[dim])).any(
            dim=dim
        )

        return v.masked_fill(~topk_mask, float("-inf")), topk_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        topk_logits, topk_mask = NoisyTopKGating._keep_topk(
            self.gate(x) + torch.randn_like(x) * self.var(x), self.k_top
        )
        return F.softmax(topk_logits), topk_mask


class SparselyGatedMoE(nn.Module):
    def __init__(self, d_embd: int, n_experts: int, k_top: int) -> None:
        super().__init__()

        self.topk_gate = NoisyTopKGating(d_embd, n_experts, k_top)

        self.experts = nn.ModuleList(
            FeedForward(d_embd, 4 * d_embd) for _ in range(n_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(x)

        topk_coef, topk_mask = self.topk_gate(x)

        for expert_idx, expert in enumerate(self.experts):
            output[expert_mask] = (
                output[(expert_mask := topk_mask[..., expert_idx])]
                + expert(x[expert_mask]) * topk_coef[..., expert_idx]
            )

        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        block_size: int,
        d_embd: int,
        n_heads: int,
        n_experts: int,
        k_top: int,
        drop_rate: float,
    ):
        super().__init__()

        self.sa = MultiHeadAttention(
            block_size, d_embd, d_embd // n_heads, n_heads, drop_rate
        )

        self.moe = SparselyGatedMoE(d_embd, n_experts, k_top)

        self.norm1 = RMSNorm(d_embd)
        self.norm2 = RMSNorm(d_embd)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.sa(self.norm1(x)))
        x = x + self.dropout(self.moe(self.norm2(x)))

        return x


class LanguageModel(nn.Module):
    def __init__(
        self,
        block_size: int,
        vocab_size: int,
        d_embd: int,
        n_heads: int,
        n_experts: int,
        k_top: int,
        n_blocks: int,
        drop_rate: float,
    ):
        super().__init__()

        self.tok_embd = nn.Embedding(vocab_size, d_embd)
        self.pos_embd = nn.Embedding(block_size, d_embd)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    block_size, d_embd, n_heads, n_experts, k_top, drop_rate
                )
                for _ in range(n_blocks)
            ]
        )

        self.norm = RMSNorm(d_embd)

        self.unembd = nn.Linear(d_embd, vocab_size)
        self.unembd.weight = self.tok_embd.weight

        self.loss = nn.CrossEntropyLoss()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        embd = self.tok_embd(x)
        embd += self.pos_embd(torch.arange(x.shape[-1]).to(x.device))

        return self.unembd(self.norm(self.blocks(embd)))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._forward(x)
        logits = logits.view(-1, logits.shape[-1])
        y = y.flatten()

        return self.loss(logits, y)

    def generate(
        self,
        x: torch.Tensor,
        max_tokens: int = 100,
    ) -> torch.Tensor:
        T = x.shape[-1]
        for _ in range(max_tokens):
            logits = self._forward(x[..., -T:])

            p = F.softmax(logits[..., -1, :], dim=-1)

            next_token = torch.multinomial(p, num_samples=1)

            x = torch.cat([x, next_token], dim=-1)

        return x
