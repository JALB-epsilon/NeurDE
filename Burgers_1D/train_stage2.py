import argparse
import os
import random

import h5py
import torch
import torch.nn.functional as F
import yaml

from architectures import NeurDE, RESIDUAL_MODES
from burgers_solver import (
    BurgersSolver,
    default_config_path,
    resolve_config_path,
    resolve_artifact_path,
    get_model_config,
    resolve_stabilizer_kwargs,
    resolve_torch_dtype,
)


def reshape_prediction(prediction, batch_size, x_points, qn):
    return prediction.reshape(batch_size, x_points, qn).permute(0, 2, 1)


def relative_error(prediction, target, eps=1.0e-7):
    return torch.norm(prediction - target) / (torch.norm(target) + eps)


def local_variation_increase_penalty(u_new, u_old):
    if u_new.shape != u_old.shape:
        raise ValueError("u_new and u_old must have the same shape.")
    diff_new = u_new[..., 1:] - u_new[..., :-1]
    diff_old = u_old[..., 1:] - u_old[..., :-1]
    variation_growth = torch.relu(torch.abs(diff_new) - torch.abs(diff_old))
    return variation_growth.square().mean()


def local_curvature_increase_penalty(u_new, u_old):
    if u_new.shape != u_old.shape:
        raise ValueError("u_new and u_old must have the same shape.")
    if u_new.shape[-1] < 3:
        return torch.zeros((), device=u_new.device, dtype=u_new.dtype)
    diff_new = u_new[..., 1:] - u_new[..., :-1]
    diff_old = u_old[..., 1:] - u_old[..., :-1]
    curv_new = diff_new[..., 1:] - diff_new[..., :-1]
    curv_old = diff_old[..., 1:] - diff_old[..., :-1]
    curvature_growth = torch.relu(torch.abs(curv_new) - torch.abs(curv_old))
    return curvature_growth.square().mean()


def apply_loss_metric(prediction, target, metric="relative_error", huber_delta=0.01):
    metric_name = str(metric).lower()
    if metric_name == "relative_error":
        return relative_error(prediction, target)
    if metric_name == "l1":
        return F.l1_loss(prediction, target)
    if metric_name == "mse":
        return F.mse_loss(prediction, target)
    if metric_name == "huber":
        return F.huber_loss(prediction, target, delta=float(huber_delta))
    raise ValueError(f"Unsupported Burgers shock-loss metric: {metric}")


def edge_weighted_gradient_loss(prediction, target, dx, metric, huber_delta, edge_alpha, eps=1.0e-7):
    pred_gradient = (prediction[..., 1:] - prediction[..., :-1]) / max(float(dx), eps)
    target_gradient = (target[..., 1:] - target[..., :-1]) / max(float(dx), eps)
    if float(edge_alpha) > 0.0:
        gradient_scale = target_gradient.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        weights = 1.0 + float(edge_alpha) * target_gradient.abs() / gradient_scale
        pred_gradient = weights * pred_gradient
        target_gradient = weights * target_gradient
    return apply_loss_metric(pred_gradient, target_gradient, metric=metric, huber_delta=huber_delta)


def cumulative_integral_loss(prediction, target, dx, metric, huber_delta):
    pred_integral = torch.cumsum(prediction, dim=-1) * float(dx)
    target_integral = torch.cumsum(target, dim=-1) * float(dx)
    return apply_loss_metric(pred_integral, target_integral, metric=metric, huber_delta=huber_delta)


def compute_burgers_shock_loss(u_pred, target, dx, shock_loss_config=None):
    if not shock_loss_config:
        return relative_error(u_pred, target)

    metric = shock_loss_config.get("metric", "huber")
    huber_delta = float(shock_loss_config.get("huber_delta", 0.01))
    state_weight = float(shock_loss_config.get("state_weight", 1.0))
    integral_weight = float(shock_loss_config.get("integral_weight", 0.0))
    gradient_weight = float(shock_loss_config.get("gradient_weight", 0.0))
    gradient_edge_alpha = float(shock_loss_config.get("gradient_edge_alpha", 0.0))
    conservation_weight = float(shock_loss_config.get("conservation_weight", 0.0))
    range_weight = float(shock_loss_config.get("range_weight", 0.0))
    range_min = shock_loss_config.get("range_min")
    range_max = shock_loss_config.get("range_max")

    loss = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)

    if state_weight > 0.0:
        loss = loss + state_weight * apply_loss_metric(u_pred, target, metric=metric, huber_delta=huber_delta)
    if integral_weight > 0.0:
        loss = loss + integral_weight * cumulative_integral_loss(
            u_pred,
            target,
            dx=dx,
            metric=metric,
            huber_delta=huber_delta,
        )
    if gradient_weight > 0.0:
        loss = loss + gradient_weight * edge_weighted_gradient_loss(
            u_pred,
            target,
            dx=dx,
            metric=metric,
            huber_delta=huber_delta,
            edge_alpha=gradient_edge_alpha,
        )
    if conservation_weight > 0.0:
        loss = loss + conservation_weight * apply_loss_metric(
            u_pred.mean(dim=-1),
            target.mean(dim=-1),
            metric=metric,
            huber_delta=huber_delta,
        )
    if range_weight > 0.0:
        range_penalty = torch.zeros((), device=u_pred.device, dtype=u_pred.dtype)
        if range_min is not None:
            range_penalty = range_penalty + torch.relu(float(range_min) - u_pred).square().mean()
        if range_max is not None:
            range_penalty = range_penalty + torch.relu(u_pred - float(range_max)).square().mean()
        loss = loss + range_weight * range_penalty

    return loss


def mean_burgers_shock_loss(u_pred, target, dx, shock_loss_config=None):
    if u_pred.dim() <= 1:
        return compute_burgers_shock_loss(u_pred, target, dx=dx, shock_loss_config=shock_loss_config)
    return torch.stack([
        compute_burgers_shock_loss(
            u_pred[idx],
            target[idx],
            dx=dx,
            shock_loss_config=shock_loss_config,
        )
        for idx in range(u_pred.shape[0])
    ]).mean()


def resolve_rollout(rollout_schedule, epoch):
    if not rollout_schedule:
        return 1
    remaining = int(epoch)
    last_rollout = int(rollout_schedule[-1]["rollout"])
    for stage in rollout_schedule:
        rollout = int(stage["rollout"])
        stage_epochs = int(stage.get("epochs", 0))
        last_rollout = rollout
        if stage_epochs <= 0:
            return rollout
        if remaining < stage_epochs:
            return rollout
        remaining -= stage_epochs
    return last_rollout


def assert_finite_tensor(tensor, name, epoch, rollout, batch_start=None):
    if torch.isfinite(tensor).all():
        return
    location = f"epoch={epoch}, rollout_stage={rollout}"
    if batch_start is not None:
        location += f", batch_start={batch_start}"
    raise FloatingPointError(f"Non-finite tensor detected for {name} at {location}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path())
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    dtype = resolve_torch_dtype(args.dtype)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config_path = resolve_config_path(args.config)
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    if "stage2" not in config:
        raise ValueError("Stage-2 config section is required.")
    stage2 = config["stage2"]
    artifact_root = config.get("artifact_root")
    model_config = get_model_config(config)

    data_path = resolve_artifact_path(config["data_dir"], artifact_root=artifact_root)
    results_dir = resolve_artifact_path(stage2["results_dir"], artifact_root=artifact_root)
    pretrained_path = resolve_artifact_path(stage2["pretrained_path"], artifact_root=artifact_root)
    os.makedirs(results_dir, exist_ok=True)

    with h5py.File(data_path, "r") as handle:
        if "F" not in handle:
            raise ValueError(
                "Burgers stage-2 requires population states saved as dataset 'F'. "
                "Regenerate the dataset with Burgers_1D/burgers_solver.py."
            )
        total_steps = handle["u"].shape[0]
        limit = min(int(stage2.get("num_samples", total_steps)), total_steps)
        train_count = args.train_count if args.train_count is not None else int(stage2.get("train_count", limit))
        train_count = max(2, min(int(train_count), limit))
        all_F = torch.as_tensor(handle["F"][:train_count], dtype=dtype, device=args.device)
        all_u = torch.as_tensor(handle["u"][:train_count], dtype=dtype, device=args.device)
        all_feq = torch.as_tensor(handle["Feq"][:train_count], dtype=dtype, device=args.device)

    solver = BurgersSolver(
        X=config["X"],
        lam=config["lam"],
        omega=config["omega"],
        lattice=config.get("lattice", "D1Q2"),
        equilibrium_mode=config.get("equilibrium_mode", "default"),
        device=args.device,
        newton_steps=config.get("newton_steps", 50),
        newton_tol=config.get("newton_tol", 1e-10),
        domain_length=config.get("domain_length", 1.0),
        alpha=config.get("alpha", 0.5),
        s2=config.get("s2", 1.7),
        s3=config.get("s3", 1.7),
        boundary=config.get("boundary", "outflow"),
        u_left_bc=config.get("u_left"),
        u_right_bc=config.get("u_right"),
        dtype=dtype,
        **resolve_stabilizer_kwargs(config),
    )
    model = NeurDE(
        alpha_layer=[1] + [config["hidden_dim"]] * config["num_layers"],
        phi_layer=[solver.basis().shape[-1]] + [config["hidden_dim"]] * config["num_layers"],
        activation="relu",
        learn_feq=True,
        learn_geq=False,
        feq_mode=model_config["feq_mode"],
        logit_clip=model_config["logit_clip"],
    ).to(device=args.device, dtype=dtype)
    model.load_state_dict(torch.load(pretrained_path, map_location=args.device))
    basis = solver.basis().to(device=args.device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=stage2["lr"])
    epochs = args.epochs_override or int(stage2["epochs"])
    rollout_schedule = stage2.get("rollout_schedule", [{"rollout": 1, "epochs": 0}])
    supervision_mode = str(stage2.get("supervision", "macro")).lower()
    if supervision_mode not in {"macro", "feq"}:
        raise ValueError(f"Unsupported Burgers stage2.supervision: {supervision_mode}")
    rollout_batch_size = int(stage2.get("batch_size", config.get("train", {}).get("batch_size", 1)))
    if rollout_batch_size <= 0:
        raise ValueError("stage2.batch_size must be positive.")
    grad_clip_norm = float(stage2.get("grad_clip_norm", 1.0))
    use_tvd = bool(stage2.get("TVD", stage2.get("use_tvd", False)))
    tvd_weight = float(stage2.get("tvd_weight", 0.0))
    curvature_weight = float(stage2.get("curvature_weight", 0.0))
    shock_loss_config = dict(stage2.get("shock_loss", {}))
    if shock_loss_config:
        if "range_min" not in shock_loss_config and config.get("macro_range_min") is not None:
            shock_loss_config["range_min"] = float(config.get("macro_range_min"))
        if "range_max" not in shock_loss_config and config.get("macro_range_max") is not None:
            shock_loss_config["range_max"] = float(config.get("macro_range_max"))

    print(
        f"Stage-2 Burgers fine-tune on {args.device}. train_count={train_count}, "
        f"epochs={epochs}, pretrained={pretrained_path}, supervision={supervision_mode}, "
        f"data_path={data_path}, results_dir={results_dir}, feq_mode={model_config['feq_mode']}"
    )
    print(f"Rollout schedule: {rollout_schedule}")
    print(
        f"TVD enabled={use_tvd}, tvd_weight={tvd_weight}, curvature_weight={curvature_weight}, "
        f"grad_clip_norm={grad_clip_norm}"
    )
    if shock_loss_config:
        print(f"Shock-aware loss config: {shock_loss_config}")
    else:
        print("Shock-aware loss config: disabled")

    best_loss = float("inf")
    best_path = os.path.join(results_dir, "burgers_stage2_best.pt")
    best_rollout = None
    best_rollout_paths = {}
    previous_rollout = None

    x_points = solver.X
    for epoch in range(epochs):
        current_rollout = resolve_rollout(rollout_schedule, epoch)
        if current_rollout <= 0:
            raise ValueError("Rollout schedule must use positive rollout lengths.")
        if current_rollout != previous_rollout:
            best_loss = float("inf")
            previous_rollout = current_rollout

        # Burgers stage-2 compares against the next macro state after each step,
        # so a rollout of length N requires snapshots [start, start + N].
        max_start = train_count - current_rollout - 1
        if max_start < 0:
            raise ValueError(
                f"Not enough training snapshots ({train_count}) for rollout={current_rollout}. "
                f"Need at least rollout + 1 snapshots."
            )

        starts = list(range(max_start + 1))
        random.shuffle(starts)

        epoch_loss = 0.0
        model.train()
        for start_offset in range(0, len(starts), rollout_batch_size):
            batch_starts = starts[start_offset:start_offset + rollout_batch_size]
            batch_start_indices = torch.tensor(batch_starts, device=args.device, dtype=torch.long)
            batch_size = len(batch_starts)
            F = all_F[batch_start_indices].clone()
            loss = torch.zeros((), device=args.device, dtype=dtype)
            optimizer.zero_grad()
            for step in range(current_rollout):
                u_current = solver.macro(F)
                inputs = u_current.unsqueeze(1).unsqueeze(2)
                model_kwargs = {}
                if model_config["feq_mode"] in RESIDUAL_MODES:
                    with torch.no_grad():
                        model_kwargs["feq_base"] = solver.equilibrium(u_current)
                feq_pred = model(inputs, basis, **model_kwargs)
                assert_finite_tensor(
                    feq_pred,
                    name="feq_pred",
                    epoch=epoch,
                    rollout=current_rollout,
                    batch_start=batch_starts[0] if batch_starts else None,
                )
                feq_pred = reshape_prediction(feq_pred, batch_size, x_points, solver.Qn)
                if supervision_mode == "feq":
                    target_feq = all_feq[batch_start_indices + step]
                    step_loss = relative_error(feq_pred, target_feq)
                else:
                    step_loss = None
                F, _, _ = solver.step(F, feq_pred)
                u_next = solver.macro(F)
                if supervision_mode == "macro":
                    target = all_u[batch_start_indices + step + 1]
                    step_loss = mean_burgers_shock_loss(
                        u_next,
                        target,
                        dx=solver.dx,
                        shock_loss_config=shock_loss_config,
                    )
                    if use_tvd:
                        step_loss = step_loss + tvd_weight * local_variation_increase_penalty(u_next, u_current)
                        if curvature_weight > 0.0:
                            step_loss = step_loss + curvature_weight * local_curvature_increase_penalty(u_next, u_current)
                loss = loss + step_loss
            loss = loss / float(current_rollout)
            assert_finite_tensor(
                loss,
                name="stage2_loss",
                epoch=epoch,
                rollout=current_rollout,
                batch_start=batch_starts[0] if batch_starts else None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            for parameter_name, parameter in model.named_parameters():
                assert_finite_tensor(
                    parameter,
                    name=f"parameter:{parameter_name}",
                    epoch=epoch,
                    rollout=current_rollout,
                    batch_start=batch_starts[0] if batch_starts else None,
                )
            epoch_loss += float(loss.item()) * batch_size

        epoch_loss /= max(len(starts), 1)
        print(f"Epoch {epoch}: rollout={current_rollout}, loss={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_rollout = current_rollout
            torch.save(model.state_dict(), best_path)
            best_rollout_path = os.path.join(results_dir, f"burgers_stage2_best_rollout{current_rollout}.pt")
            torch.save(model.state_dict(), best_rollout_path)
            best_rollout_paths[current_rollout] = best_rollout_path

    last_path = os.path.join(results_dir, "burgers_stage2_last.pt")
    torch.save(model.state_dict(), last_path)
    print(f"Saved best stage-2 model to {best_path} (rollout={best_rollout}, loss={best_loss:.6f})")
    for rollout, rollout_path in sorted(best_rollout_paths.items()):
        print(f"Saved rollout-specific best checkpoint to {rollout_path}")
    print(f"Saved last stage-2 model to {last_path}")


if __name__ == "__main__":
    main()
