"""The LoRA mapper: compiler prefix hidden states -> per-layer LoRA A/B.

Mirrors the authors' `LoraMapper` (mean_pool mode, trunk_depth=1), whose
state dict is published as `lora_mapper.pt`:

    trunk.0            Linear(2560, 2560)        (followed by GELU)
    coeff_head         Linear(2560, L*M*N*2)     L=28, M=7, N=64
    A_bases_{module}   (N, rank, d_in)
    B_bases_{module}   (N, d_out, rank)

Coefficients reshape to (L, M, N, 2) with modules in *sorted* name order
and the last dim indexing (A, B). Final LoRA per (layer, module):
    A = sum_n alpha_A[n] * A_bases[n]   (rank, d_in)
    B = sum_n alpha_B[n] * B_bases[n]   (d_out, rank)
applied at runtime as  out += x @ A^T @ B^T * (lora_alpha / rank).
"""

import torch


class LoraMapper(torch.nn.Module):
    def __init__(
        self,
        teacher_hidden_size: int,
        student_num_layers: int,
        module_dims: dict[str, tuple[int, int]],
        lora_rank: int,
        lora_alpha: float,
        num_bases: int,
    ) -> None:
        super().__init__()
        self.student_num_layers = student_num_layers
        self.lora_rank = lora_rank
        self.lora_scaling = lora_alpha / lora_rank
        self.module_names = sorted(module_dims.keys())
        self.num_bases = num_bases

        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(teacher_hidden_size, teacher_hidden_size),
            torch.nn.GELU(),
        )
        for m in self.module_names:
            d_in, d_out = module_dims[m]
            self.register_parameter(
                f"A_bases_{m}",
                torch.nn.Parameter(torch.zeros(num_bases, lora_rank, d_in)),
            )
            self.register_parameter(
                f"B_bases_{m}",
                torch.nn.Parameter(torch.zeros(num_bases, d_out, lora_rank)),
            )
        num_coeff = student_num_layers * len(self.module_names) * num_bases * 2
        self.coeff_head = torch.nn.Linear(teacher_hidden_size, num_coeff)

    @torch.no_grad()
    def forward(
        self, teacher_hidden: list[torch.Tensor]
    ) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
        """teacher_hidden: one (T, H) tensor per interpreter layer."""
        h = torch.stack(teacher_hidden, dim=0).mean(dim=0)  # (T, H)
        h = h.mean(dim=0)  # (H,)
        z = self.trunk(h)
        coeffs = self.coeff_head(z)

        L = self.student_num_layers
        M = len(self.module_names)
        N = self.num_bases
        coeffs = coeffs.view(L, M, N, 2)

        lora_params: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for mi, m in enumerate(self.module_names):
            a_bases = getattr(self, f"A_bases_{m}")  # (N, rank, d_in)
            b_bases = getattr(self, f"B_bases_{m}")  # (N, d_out, rank)
            for layer in range(L):
                alpha_a = coeffs[layer, mi, :, 0]  # (N,)
                alpha_b = coeffs[layer, mi, :, 1]  # (N,)
                a = torch.einsum("n,nrd->rd", alpha_a, a_bases)
                b = torch.einsum("n,ndr->dr", alpha_b, b_bases)
                lora_params[(layer, m)] = (a, b)
        return lora_params


def depth_ratio_layers(num_teacher_layers: int, num_student_layers: int) -> list[int]:
    """Teacher layer index (0-based) aligned to each student layer.

    Matches the authors' depth_ratio rule; indexes into
    `hidden_states[idx + 1]` (position 0 is the embedding output).
    """
    ratio = num_teacher_layers / num_student_layers
    indices = []
    for layer_idx in range(num_student_layers):
        t = int((layer_idx + 1) * ratio) - 1
        indices.append(max(0, min(num_teacher_layers - 1, t)))
    return indices


def load_mapper(
    mapper_path: str,
    teacher_hidden_size: int,
    student_num_layers: int,
    module_dims: dict[str, tuple[int, int]],
    lora_rank: int,
    lora_alpha: float,
    num_bases: int,
) -> LoraMapper:
    state = torch.load(mapper_path, map_location="cpu", weights_only=True)
    mapper = LoraMapper(
        teacher_hidden_size=teacher_hidden_size,
        student_num_layers=student_num_layers,
        module_dims=module_dims,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_bases=num_bases,
    )
    mapper.load_state_dict(state, strict=True)
    # The mapper is kept in fp32 for numerical stability (paper Appendix G).
    mapper.float().eval()
    return mapper
