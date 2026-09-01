class FlyInspiredThoughtletCore(nn.Module):
    """Fly-inspired thoughtlet architecture with 4 functional modules.

    Each module: 8 thoughtlet slots (32 total).
    Modules: Sensory Encoding, Integration, Memory, Action Selection.
    Each module has its own input projection with unique seed for
    differentiation (fixes cross-module correlation = 1.0 issue).
    """

    def __init__(self, input_dim: int = 64, thoughtlet_width: int = 120,
                 n_thoughtlets: int = 32, n_modules: int = 4,
                 sparsity: float = 0.03, seed: int = 42):
        super().__init__()
        self.n_modules = n_modules
        self.thoughtlet_width = thoughtlet_width
        self.n_thoughtlets = n_thoughtlets

        self.sparse_encoder = SparseEncoder(
            input_dim, n_kc=2000, sparsity=sparsity, seed=seed)

        # Module-specific input projections with different seeds
        self.module_projections = nn.ModuleList()
        for m in range(n_modules):
            proj_seed = seed + m * 137
            rng = np.random.RandomState(proj_seed)
            w = rng.randn(input_dim, thoughtlet_width).astype(np.float32)
            proj = nn.Linear(input_dim, thoughtlet_width)
            proj.weight.data = torch.from_numpy(w)
            proj.bias.data = torch.zeros(thoughtlet_width)
            self.module_projections.append(proj)

        self.ring_attractor = RingAttractorModule(n_neurons=36)
        self.dan_gate = DopaminergicGate()

        # ~6% of inter-module connections get modulation (fly ratio)
        self.inter_gate_weights = nn.Parameter(
            torch.ones(n_modules, n_modules) * 0.06, requires_grad=True)

        self.output_projection = nn.Linear(thoughtlet_width, input_dim)

    def forward(self, x: torch.Tensor,
                reward_signal: torch.Tensor = None) -> torch.Tensor:
        z = self.sparse_encoder(x)
        module_outputs = []
        for i, proj in enumerate(self.module_projections):
            h = proj(z)
            h = F.relu(h)
            module_outputs.append(h)

        if reward_signal is not None:
            gate_values = self.dan_gate(reward_signal)
            for i in range(self.n_modules):
                module_outputs[i] = module_outputs[i] * gate_values[i]

        stacked = torch.stack(module_outputs, dim=1)
        gated = stacked * torch.sigmoid(
            self.inter_gate_weights).unsqueeze(0)
        combined = gated.mean(dim=1)
        output = self.output_projection(combined)
        return output


def create_fly_inspired_model(input_dim: int = 64,
                              thoughtlet_width: int = 120,
                              n_thoughtlets: int = 32):
    """Factory function for fly-inspired model."""
    return FlyInspiredThoughtletCore(
        input_dim=input_dim,
        thoughtlet_width=thoughtlet_width,
        n_thoughtlets=n_thoughtlets,
        sparsity=0.03,
        seed=42,
    )
