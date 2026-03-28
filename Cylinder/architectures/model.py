import torch
import torch.nn as nn


def _compute_logits(alpha_net, phi_net, macro_state, basis, logit_clip):
    flat_macro_state = macro_state.movedim(1, -1).reshape(-1, macro_state.shape[1])
    alpha_coeffs = alpha_net(flat_macro_state)
    basis_coeffs = phi_net(basis)
    logits = torch.einsum("bi,ni->bn", alpha_coeffs, basis_coeffs)
    if logit_clip is not None:
        logits = logits.clamp(min=-logit_clip, max=logit_clip)
    return logits, flat_macro_state


def _build_moment_matrix(basis):
    ones = torch.ones_like(basis[:, 0])
    return torch.stack([ones, basis[:, 0], basis[:, 1]], dim=0)


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


def _feq_targets(flat_macro_state):
    rho = flat_macro_state[:, 0]
    ux = flat_macro_state[:, 1]
    uy = flat_macro_state[:, 2]
    return torch.stack([rho, rho * ux, rho * uy], dim=-1)


def _geq_targets(flat_macro_state, cv):
    rho = flat_macro_state[:, 0]
    ux = flat_macro_state[:, 1]
    uy = flat_macro_state[:, 2]
    temperature = flat_macro_state[:, 3]
    kinetic = ux.square() + uy.square()
    energy = cv * temperature + 0.5 * kinetic
    enthalpy = energy + temperature
    return torch.stack(
        [
            2.0 * rho * energy,
            2.0 * rho * ux * enthalpy,
            2.0 * rho * uy * enthalpy,
        ],
        dim=-1,
    )


def _base_measure(q_count, device, dtype, mode):
    mode = str(mode).lower()
    if mode != "d2q9":
        raise ValueError(f"Unsupported base measure mode: {mode}")
    if q_count != 9:
        raise ValueError(f"d2q9 base measure only supports 9 populations, got {q_count}")
    return torch.tensor(
        [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4 + [4.0 / 9.0],
        device=device,
        dtype=dtype,
    )


def _solve_conserved_multipliers(
    log_base,
    moment_matrix,
    target_moments,
    max_iters=20,
    tolerance=1e-6,
):
    multipliers = torch.zeros(
        (target_moments.shape[0], target_moments.shape[1]),
        device=log_base.device,
        dtype=log_base.dtype,
    )
    moment_matrix = moment_matrix.to(device=log_base.device, dtype=log_base.dtype)
    identity = torch.eye(
        moment_matrix.shape[0], device=log_base.device, dtype=log_base.dtype
    ).unsqueeze(0)

    def evaluate(candidate):
        exponent = (log_base + candidate @ moment_matrix).clamp(min=-60.0, max=60.0)
        population = torch.exp(exponent)
        moments = population @ moment_matrix.transpose(0, 1)
        residual = moments - target_moments
        return population, residual

    population, residual = evaluate(multipliers)
    for _ in range(max_iters):
        if residual.abs().amax() < tolerance:
            break

        jacobian = (population.unsqueeze(1) * moment_matrix.unsqueeze(0)) @ moment_matrix.transpose(0, 1)
        jacobian = jacobian + 1e-8 * identity
        delta = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)

        previous_error = residual.abs().amax(dim=-1)
        step_scale = 1.0
        accepted = False
        for _ in range(8):
            candidate = multipliers - step_scale * delta
            candidate_population, candidate_residual = evaluate(candidate)
            candidate_error = candidate_residual.abs().amax(dim=-1)
            if torch.all(candidate_error <= previous_error + 1e-9):
                multipliers = candidate
                population = candidate_population
                residual = candidate_residual
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            multipliers = multipliers - 0.25 * delta
            population, residual = evaluate(multipliers)

    return population


class EquilibriumHead(nn.Module):
    def __init__(
        self,
        alpha_layer,
        trunk_layer,
        activation,
        mode="positive",
        constraint_kind="geq",
        cv=None,
        logit_clip=15.0,
        base_measure="d2q9",
        newton_iters=20,
        newton_tolerance=1e-6,
    ):
        super().__init__()
        self.alpha = DenseNet(alpha_layer, activation)
        self.phi = DenseNet(trunk_layer, activation)
        self.mode = str(mode).lower()
        self.constraint_kind = str(constraint_kind).lower()
        self.cv = cv
        self.logit_clip = logit_clip
        self.base_measure = str(base_measure).lower()
        self.newton_iters = int(newton_iters)
        self.newton_tolerance = float(newton_tolerance)

        if self.mode not in {"positive", "constrained"}:
            raise ValueError(f"Unsupported equilibrium head mode: {mode}")
        if self.constraint_kind not in {"feq", "geq"}:
            raise ValueError(f"Unsupported constraint kind: {constraint_kind}")
        if self.constraint_kind == "geq" and self.cv is None:
            raise ValueError("cv is required for constrained Geq heads.")

    def forward(self, macro_state, basis):
        logits, flat_macro_state = _compute_logits(
            self.alpha,
            self.phi,
            macro_state,
            basis,
            self.logit_clip,
        )
        if self.mode == "positive":
            return torch.exp(logits)

        moment_matrix = _build_moment_matrix(basis)
        nonconserved_logits = _project_to_nonconserved(logits, moment_matrix)
        log_base = torch.log(
            _base_measure(
                basis.shape[0],
                device=logits.device,
                dtype=logits.dtype,
                mode=self.base_measure,
            )
        ).unsqueeze(0) + nonconserved_logits
        if self.constraint_kind == "feq":
            target_moments = _feq_targets(flat_macro_state)
        else:
            target_moments = _geq_targets(flat_macro_state, self.cv)
        return _solve_conserved_multipliers(
            log_base,
            moment_matrix,
            target_moments,
            max_iters=self.newton_iters,
            tolerance=self.newton_tolerance,
        )


class NeurDE(nn.Module):
    def __init__(
        self,
        alpha_layer,
        phi_layer=None,
        activation="relu",
        branch_layer=None,
        learn_feq=False,
        learn_geq=True,
        feq_mode="positive",
        geq_mode="positive",
        cv=None,
        logit_clip=15.0,
        feq_base_measure="d2q9",
        geq_base_measure="d2q9",
        newton_iters=20,
        newton_tolerance=1e-6,
    ):
        super().__init__()
        trunk_layer = phi_layer if phi_layer is not None else branch_layer
        if trunk_layer is None:
            raise ValueError("Either phi_layer or branch_layer must be provided.")

        self.feq_head = (
            EquilibriumHead(
                alpha_layer,
                trunk_layer,
                activation,
                mode=feq_mode,
                constraint_kind="feq",
                cv=cv,
                logit_clip=logit_clip,
                base_measure=feq_base_measure,
                newton_iters=newton_iters,
                newton_tolerance=newton_tolerance,
            )
            if learn_feq
            else None
        )
        self.geq_head = (
            EquilibriumHead(
                alpha_layer,
                trunk_layer,
                activation,
                mode=geq_mode,
                constraint_kind="geq",
                cv=cv,
                logit_clip=logit_clip,
                base_measure=geq_base_measure,
                newton_iters=newton_iters,
                newton_tolerance=newton_tolerance,
            )
            if learn_geq
            else None
        )

    def forward(self, macro_state, basis):
        feq = self.feq_head(macro_state, basis) if self.feq_head is not None else None
        geq = self.geq_head(macro_state, basis) if self.geq_head is not None else None
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
        assert self.n_layers >= 1
        if isinstance(nonlinearity, str):
            if nonlinearity.lower() == "tanh":
                self.nonlinearity = nn.Tanh()
            elif nonlinearity.lower() == "relu":
                self.nonlinearity = nn.ReLU()
            else:
                raise ValueError(f"{nonlinearity} type {type(nonlinearity)} is not supported")

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))
                self.layers.append(self.nonlinearity)

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    alpha_layer = [4, 50, 50, 50, 50]
    phi_layer = [2, 50, 50, 50, 50]
    activation = "relu"
    model = NeurDE(alpha_layer, phi_layer, activation)
    print(model)
