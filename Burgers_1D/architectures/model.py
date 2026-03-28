import torch
import torch.nn as nn


RESIDUAL_MODES = {"constrained", "residual", "residual_constrained"}


def _compute_logits(alpha_net, phi_net, macro_state, basis, logit_clip):
    flat_macro_state = macro_state.movedim(1, -1).reshape(-1, macro_state.shape[1])
    alpha_coeffs = alpha_net(flat_macro_state)
    basis_coeffs = phi_net(basis)
    logits = torch.einsum("bi,ni->bn", alpha_coeffs, basis_coeffs)
    if logit_clip is not None:
        logits = logits.clamp(min=-logit_clip, max=logit_clip)
    return logits, flat_macro_state


def _flatten_population(population):
    if population.ndim == 3:
        return population.permute(0, 2, 1).reshape(-1, population.shape[1])
    if population.ndim == 2:
        return population
    raise ValueError(f"Unsupported Burgers population tensor rank: {population.ndim}")


def _build_burgers_moment_matrix(basis):
    ex = basis[:, 0] if basis.shape[1] > 1 else basis[:, 0]
    ones = torch.ones_like(ex)
    return torch.stack([ones, ex], dim=0)


def _burgers_targets(flat_macro_state):
    u = flat_macro_state[:, 0]
    flux = 0.5 * u.square()
    return torch.stack([u, flux], dim=-1)


def _build_nullspace_projector(moment_matrix):
    gram = moment_matrix @ moment_matrix.transpose(0, 1)
    return torch.eye(
        moment_matrix.shape[1],
        device=moment_matrix.device,
        dtype=moment_matrix.dtype,
    ) - moment_matrix.transpose(0, 1) @ torch.linalg.inv(gram) @ moment_matrix


def _project_to_nonconserved(logits, moment_matrix):
    projector = _build_nullspace_projector(moment_matrix)
    return logits @ projector.transpose(0, 1)


def _zero_last_linear(dense_net):
    for layer in reversed(dense_net.layers):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return
    raise RuntimeError("DenseNet does not contain a linear layer to initialize.")


def _project_population_to_moments(population, target_moments, moment_matrix):
    gram_inv = torch.linalg.inv(moment_matrix @ moment_matrix.transpose(0, 1))
    predicted_moments = population @ moment_matrix.transpose(0, 1)
    correction = (target_moments - predicted_moments) @ gram_inv @ moment_matrix
    return population + correction


def project_scalar_conservative_mass(population, flat_macro_state):
    target_moments = flat_macro_state[:, :1]
    moment_matrix = torch.ones((1, population.shape[-1]), device=population.device, dtype=population.dtype)
    return _project_population_to_moments(population, target_moments, moment_matrix)


def project_burgers_conservative_moments(population, flat_macro_state, basis):
    target_moments = _burgers_targets(flat_macro_state)
    moment_matrix = _build_burgers_moment_matrix(basis)
    return _project_population_to_moments(population, target_moments, moment_matrix)


class EquilibriumHead(nn.Module):
    def __init__(
        self,
        alpha_layer,
        trunk_layer,
        activation,
        mode="positive",
        logit_clip=15.0,
    ):
        super().__init__()
        self.alpha = DenseNet(alpha_layer, activation)
        self.phi = DenseNet(trunk_layer, activation)
        self.mode = str(mode).lower()
        if self.mode in {"moment_projected_positive", "projected_burgers_moments"}:
            self.mode = "moment_constrained_free"
        self.logit_clip = logit_clip

        if self.mode not in {
            "positive",
            "projected_positive",
            "moment_constrained_free",
            "constrained",
            "residual",
            "residual_constrained",
        }:
            raise ValueError(f"Unsupported Burgers equilibrium head mode: {mode}")
        if self.mode not in {"positive", "projected_positive", "moment_constrained_free"}:
            _zero_last_linear(self.alpha)

    def forward(self, macro_state, basis, base_population=None):
        logits, flat_macro_state = _compute_logits(
            self.alpha,
            self.phi,
            macro_state,
            basis,
            self.logit_clip,
        )

        if self.mode == "positive":
            return torch.exp(logits)

        if self.mode == "projected_positive":
            population = torch.exp(logits)
            return project_scalar_conservative_mass(population, flat_macro_state)

        if self.mode == "moment_constrained_free":
            population = torch.exp(logits)
            return project_burgers_conservative_moments(population, flat_macro_state, basis)

        if base_population is None:
            raise ValueError(f"Burgers head in mode '{self.mode}' requires a baseline population.")

        base_flat = _flatten_population(base_population).to(device=logits.device, dtype=logits.dtype)
        if base_flat.shape != logits.shape:
            raise ValueError(
                f"Baseline population shape {base_flat.shape} does not match logits shape {logits.shape}."
            )

        moment_matrix = _build_burgers_moment_matrix(basis)
        target_moments = _burgers_targets(flat_macro_state)
        base_moments = base_flat @ moment_matrix.transpose(0, 1)
        if not torch.allclose(base_moments, target_moments, rtol=1e-4, atol=1e-6):
            raise ValueError("Baseline Burgers equilibrium does not satisfy the required moments.")

        nonconserved_residual = _project_to_nonconserved(logits, moment_matrix)
        return base_flat + nonconserved_residual


class NeurDE(nn.Module):
    def __init__(
        self,
        alpha_layer,
        phi_layer=None,
        activation="relu",
        branch_layer=None,
        learn_feq=True,
        learn_geq=False,
        feq_mode="positive",
        geq_mode="positive",
        logit_clip=15.0,
        conservative_output=None,
    ):
        super().__init__()
        trunk_layer = phi_layer if phi_layer is not None else branch_layer
        if trunk_layer is None:
            raise ValueError("Either phi_layer or branch_layer must be provided.")

        resolved_feq_mode = str(feq_mode).lower() if feq_mode is not None else None
        if resolved_feq_mode is None:
            resolved_feq_mode = "projected_positive" if bool(conservative_output) else "positive"

        self.feq_head = (
            EquilibriumHead(
                alpha_layer,
                trunk_layer,
                activation,
                mode=resolved_feq_mode,
                logit_clip=logit_clip,
            )
            if learn_feq
            else None
        )
        self.geq_head = (
            EquilibriumHead(
                alpha_layer,
                trunk_layer,
                activation,
                mode=str(geq_mode).lower(),
                logit_clip=logit_clip,
            )
            if learn_geq
            else None
        )

    def forward(self, macro_state, basis, feq_base=None, geq_base=None):
        feq = (
            self.feq_head(macro_state, basis, base_population=feq_base)
            if self.feq_head is not None
            else None
        )
        geq = (
            self.geq_head(macro_state, basis, base_population=geq_base)
            if self.geq_head is not None
            else None
        )
        if feq is not None and geq is not None:
            return feq, geq
        if feq is not None:
            return feq
        if geq is not None:
            return geq
        raise RuntimeError("NeurDE was created without any equilibrium heads.")


class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        if isinstance(nonlinearity, str):
            if nonlinearity.lower() == "tanh":
                self.nonlinearity = nn.Tanh()
            elif nonlinearity.lower() == "relu":
                self.nonlinearity = nn.ReLU()
            else:
                raise ValueError(f"{nonlinearity} is not supported")

        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            self.layers.append(nn.Linear(layers[idx], layers[idx + 1]))
            if idx != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[idx + 1]))
                self.layers.append(self.nonlinearity)

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
