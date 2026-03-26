import torch
import numpy as np
from architectures import NeurDE, bounded_residual_population, project_conserved_moments
from utilities import *
import argparse
import yaml
from tqdm import tqdm
import os
import time
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_stage_1 import create_basis
from SOD_solver import SODSolver
import torch.nn as nn


def build_effective_feq(
    *,
    learn_feq,
    feq_mode,
    feq_residual_scale,
    feq_collision_mix,
    project_feq_moments,
    feq_nullspace_gamma,
    inputs,
    basis,
    analytic_feq,
    feq_target,
    feq_pred_grid,
):
    if not learn_feq:
        return analytic_feq

    if feq_mode == "analytic_residual":
        residual_scale = max(float(feq_residual_scale), 0.0) * max(min(float(feq_collision_mix), 1.0), 0.0)
        effective_feq = bounded_residual_population(
            base_population=analytic_feq,
            predicted_population=feq_pred_grid,
            residual_scale=residual_scale,
        )
        if project_feq_moments:
            flat_macro_state = inputs.movedim(1, -1).reshape(-1, inputs.shape[1])
            effective_feq = reshape_equilibrium_prediction(
                project_conserved_moments(
                    reshape_equilibrium_target(effective_feq),
                    flat_macro_state,
                    basis,
                    nullspace_gamma=feq_nullspace_gamma,
                ),
                effective_feq.shape,
            )
        return effective_feq

    effective_feq = feq_pred_grid
    if feq_collision_mix < 1.0:
        effective_feq = feq_collision_mix * feq_pred_grid + (1.0 - feq_collision_mix) * feq_target
    return effective_feq


def run_training_benchmark(
    model,
    sod_solver,
    basis,
    reference,
    image_dir,
    epoch,
    learn_feq,
    benchmark_steps,
    feq_mode,
    feq_residual_scale,
    project_feq_moments,
    feq_nullspace_gamma,
    feq_collision_mix,
):
    Fi0 = reference["F0"].clone()
    Gi0 = reference["G0"].clone()
    nn_start = time.perf_counter()
    last_state = None
    for _ in range(benchmark_steps):
        rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
        T = sod_solver.get_temp_from_energy(ux, uy, E)
        inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1)
        analytic_feq = sod_solver.get_Feq(rho, ux, uy, T)
        if learn_feq:
            Feq_pred, Geq_pred = model(inputs, basis)
            Feq_pred_grid = reshape_equilibrium_prediction(Feq_pred, (1, sod_solver.Qn, sod_solver.Y, sod_solver.X)).squeeze(0)
            Feq_pred_grid = build_effective_feq(
                learn_feq=learn_feq,
                feq_mode=feq_mode,
                feq_residual_scale=feq_residual_scale,
                feq_collision_mix=feq_collision_mix,
                project_feq_moments=project_feq_moments,
                feq_nullspace_gamma=feq_nullspace_gamma,
                inputs=inputs,
                basis=basis,
                analytic_feq=analytic_feq,
                feq_target=analytic_feq,
                feq_pred_grid=Feq_pred_grid,
            )
        else:
            Geq_pred = model(inputs, basis)
            Feq_pred_grid = analytic_feq
        Geq_pred_grid = reshape_equilibrium_prediction(Geq_pred, (1, sod_solver.Qn, sod_solver.Y, sod_solver.X)).squeeze(0)
        Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq_pred_grid, Geq_pred_grid, rho, ux, uy, T)
        Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)
        last_state = (rho, ux, T, rho * T)
    nn_elapsed = time.perf_counter() - nn_start

    rho_ref = reference["rho"][benchmark_steps - 1]
    ux_ref = reference["ux"][benchmark_steps - 1]
    T_ref = reference["T"][benchmark_steps - 1]
    P_ref = reference["P"][benchmark_steps - 1]
    rho_pred, ux_pred, T_pred, P_pred = last_state

    plt.figure(figsize=(16, 6))
    plt.suptitle(f"SOD Training Benchmark Epoch {epoch + 1} Step {benchmark_steps}", fontweight="bold", fontsize=20, y=0.98)

    reference_label = reference.get("label", "solver")

    plt.subplot(221)
    plt.plot(detach(rho_ref[2, :]), linewidth=2, label=reference_label)
    plt.plot(detach(rho_pred[2, :]), linewidth=2, label="nn")
    plt.title("Density")
    plt.legend()

    plt.subplot(222)
    plt.plot(detach(T_ref[2, :]), linewidth=2, label=reference_label)
    plt.plot(detach(T_pred[2, :]), linewidth=2, label="nn")
    plt.title("Temperature")
    plt.legend()

    plt.subplot(223)
    plt.plot(detach(ux_ref[2, :]), linewidth=2, label=reference_label)
    plt.plot(detach(ux_pred[2, :]), linewidth=2, label="nn")
    plt.title("Velocity in x")
    plt.legend()

    plt.subplot(224)
    plt.plot(detach(P_ref[2, :]), linewidth=2, label=reference_label)
    plt.plot(detach(P_pred[2, :]), linewidth=2, label="nn")
    plt.title("Pressure")
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"epoch_{epoch + 1:04d}_step_{benchmark_steps:04d}.png")
    plt.savefig(image_path)
    plt.close()

    rel_error = 0.25 * (
        calculate_relative_error(rho_pred, rho_ref)
        + calculate_relative_error(ux_pred, ux_ref)
        + calculate_relative_error(T_pred, T_ref)
        + calculate_relative_error(P_pred, P_ref)
    )
    return float(rel_error.detach()), nn_elapsed, image_path


def run_long_rollout_validation(
    *,
    model,
    sod_solver,
    basis,
    all_F,
    all_G,
    all_Feq,
    all_rho,
    all_ux,
    all_T,
    analytic_rho,
    analytic_ux,
    analytic_T,
    analytic_P,
    device,
    start_idx,
    eval_steps,
    learn_feq,
    feq_mode,
    feq_residual_scale,
    project_feq_moments,
    feq_nullspace_gamma,
    feq_collision_mix,
):
    dtype = torch.get_default_dtype()
    Fi0 = torch.tensor(all_F[start_idx], dtype=dtype, device=device)
    Gi0 = torch.tensor(all_G[start_idx], dtype=dtype, device=device)
    solver_errors = []
    analytic_errors = []
    fail_step = None

    with torch.no_grad():
        for i in range(eval_steps):
            step_idx = start_idx + i
            rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
            T = sod_solver.get_temp_from_energy(ux, uy, E)
            if not all(torch.isfinite(t).all() for t in (rho, ux, uy, T, E)):
                fail_step = step_idx
                break

            inputs = torch.stack([rho.unsqueeze(0), ux.unsqueeze(0), uy.unsqueeze(0), T.unsqueeze(0)], dim=1)
            analytic_feq = sod_solver.get_Feq(rho, ux, uy, T)

            if learn_feq:
                Feq_pred, Geq_pred = model(inputs, basis)
                Feq_target = torch.tensor(all_Feq[step_idx], dtype=dtype, device=device)
                Feq_pred_grid = reshape_equilibrium_prediction(Feq_pred, Feq_target.unsqueeze(0).shape).squeeze(0)
                Feq_pred_grid = build_effective_feq(
                    learn_feq=learn_feq,
                    feq_mode=feq_mode,
                    feq_residual_scale=feq_residual_scale,
                    feq_collision_mix=feq_collision_mix,
                    project_feq_moments=project_feq_moments,
                    feq_nullspace_gamma=feq_nullspace_gamma,
                    inputs=inputs,
                    basis=basis,
                    analytic_feq=analytic_feq,
                    feq_target=Feq_target,
                    feq_pred_grid=Feq_pred_grid,
                )
            else:
                Geq_pred = model(inputs, basis)
                Feq_pred_grid = analytic_feq

            Geq_target_shape = (1, sod_solver.Qn, sod_solver.Y, sod_solver.X)
            Geq_pred_grid = reshape_equilibrium_prediction(Geq_pred, Geq_target_shape).squeeze(0)
            if not all(torch.isfinite(t).all() for t in (Feq_pred_grid, Geq_pred_grid)):
                fail_step = step_idx
                break

            solver_rho_t = torch.tensor(all_rho[step_idx], dtype=dtype, device=device)
            solver_ux_t = torch.tensor(all_ux[step_idx], dtype=dtype, device=device)
            solver_T_t = torch.tensor(all_T[step_idx], dtype=dtype, device=device)
            solver_error = 0.25 * (
                calculate_relative_error(rho, solver_rho_t)
                + calculate_relative_error(ux, solver_ux_t)
                + calculate_relative_error(T, solver_T_t)
                + calculate_relative_error(rho * T, solver_rho_t * solver_T_t)
            )
            if not torch.isfinite(solver_error):
                fail_step = step_idx
                break
            solver_errors.append(float(solver_error.item()))

            if analytic_rho is not None and analytic_ux is not None and analytic_T is not None and analytic_P is not None:
                analytic_rho_t = torch.tensor(analytic_rho[step_idx], dtype=dtype, device=device)
                analytic_ux_t = torch.tensor(analytic_ux[step_idx], dtype=dtype, device=device)
                analytic_T_t = torch.tensor(analytic_T[step_idx], dtype=dtype, device=device)
                analytic_P_t = torch.tensor(analytic_P[step_idx], dtype=dtype, device=device)
                analytic_error = 0.25 * (
                    calculate_relative_error(rho, analytic_rho_t)
                    + calculate_relative_error(ux, analytic_ux_t)
                    + calculate_relative_error(T, analytic_T_t)
                    + calculate_relative_error(rho * T, analytic_P_t)
                )
                if not torch.isfinite(analytic_error):
                    fail_step = step_idx
                    break
                analytic_errors.append(float(analytic_error.item()))

            Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, Feq_pred_grid, Geq_pred_grid, rho, ux, uy, T)
            Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)
            if not all(torch.isfinite(t).all() for t in (Fi0, Gi0)):
                fail_step = step_idx + 1
                break

    finite = fail_step is None and len(solver_errors) == eval_steps
    avg_solver = float(np.mean(solver_errors)) if solver_errors else float("nan")
    final_solver = float(solver_errors[-1]) if solver_errors else float("nan")
    avg_analytic = float(np.mean(analytic_errors)) if analytic_errors else float("nan")
    final_analytic = float(analytic_errors[-1]) if analytic_errors else float("nan")

    return {
        "finite": finite,
        "fail_step": fail_step,
        "avg_solver": avg_solver,
        "final_solver": final_solver,
        "avg_analytic": avg_analytic,
        "final_analytic": final_analytic,
    }


def calculate_macro_rollout_loss(loss_func, rho_pred, ux_pred, T_pred, rho_target, ux_target, T_target):
    P_pred = rho_pred * T_pred
    P_target = rho_target * T_target
    return 0.25 * (
        loss_func(rho_pred, rho_target)
        + loss_func(ux_pred, ux_target)
        + loss_func(T_pred, T_target)
        + loss_func(P_pred, P_target)
    )


def resolve_rollout_warmup_count(max_rollout, start_rollout, warmup_batches, global_batch_step):
    if max_rollout <= 1:
        return max_rollout
    start_rollout = max(1, min(int(start_rollout), int(max_rollout)))
    warmup_batches = max(0, int(warmup_batches))
    if warmup_batches == 0 or start_rollout >= max_rollout:
        return int(max_rollout)
    if warmup_batches == 1:
        return int(max_rollout)
    progress = min(max(global_batch_step, 0), warmup_batches - 1) / float(warmup_batches - 1)
    current = start_rollout + progress * (max_rollout - start_rollout)
    return int(round(current))


def resolve_schedule_value(target_value, start_value, warmup_batches, global_batch_step):
    target_value = float(target_value)
    start_value = float(start_value)
    warmup_batches = max(0, int(warmup_batches))
    if warmup_batches == 0:
        return target_value
    if warmup_batches == 1:
        return target_value
    progress = min(max(global_batch_step, 0), warmup_batches - 1) / float(warmup_batches - 1)
    return start_value + progress * (target_value - start_value)


def resolve_rollout_schedule(schedule, fallback_rollout, global_batch_step):
    if not schedule:
        return int(fallback_rollout)
    remaining = max(int(global_batch_step), 0)
    last_rollout = int(fallback_rollout)
    for stage in schedule:
        rollout_value = int(stage.get("rollout", last_rollout))
        stage_batches = int(stage.get("batches", 0))
        last_rollout = rollout_value
        if stage_batches <= 0:
            return rollout_value
        if remaining < stage_batches:
            return rollout_value
        remaining -= stage_batches
    return last_rollout


def resolve_stride_schedule(schedule, fallback_stride, global_batch_step):
    if not schedule:
        return max(1, int(fallback_stride))
    remaining = max(int(global_batch_step), 0)
    last_stride = max(1, int(fallback_stride))
    for stage in schedule:
        stride_value = max(1, int(stage.get("stride", last_stride)))
        stage_batches = int(stage.get("batches", 0))
        last_stride = stride_value
        if stage_batches <= 0:
            return stride_value
        if remaining < stage_batches:
            return stride_value
        remaining -= stage_batches
    return last_stride


def count_supervised_steps(number_of_rollout, loss_stride):
    stride = max(1, int(loss_stride))
    count = 0
    for current_step in range(1, int(number_of_rollout) + 1):
        if (current_step % stride == 0) or (current_step == int(number_of_rollout)):
            count += 1
    return max(count, 1)


def compute_stage2_rollout_loss(
    model,
    sod_solver,
    basis,
    F_seq,
    G_seq,
    Feq_seq,
    Geq_seq,
    input_rho_seq,
    input_ux_seq,
    input_uy_seq,
    input_T_seq,
    target_rho_seq,
    target_ux_seq,
    target_uy_seq,
    target_T_seq,
    number_of_rollout,
    learn_feq,
    loss_func,
    rollout_source,
    loss_target,
    use_tvd=False,
    tvd_weight=15.0,
    curvature_weight=5.0,
    equilibrium_anchor_weight=0.0,
    geq_moment_weight=0.0,
    feq_mode="learned",
    feq_residual_scale=1.0,
    project_feq_moments=False,
    feq_nullspace_gamma=1.0,
    cv=1.0,
    feq_collision_mix=1.0,
    geq_collision_mix=1.0,
    detach_interval=0,
    loss_stride=1,
    backward_chunk_size=0,
):
    Fi0 = F_seq[:, 0, ...]
    Gi0 = G_seq[:, 0, ...]
    total_loss = 0.0
    chunk_loss = None

    ux_old = T_old = rho_old = P_old = None
    for rollout in range(number_of_rollout):
        current_step = rollout + 1
        predicted_macro_selected = (current_step % max(int(loss_stride), 1) == 0) or (current_step == number_of_rollout)
        current_macro_selected = (rollout % max(int(loss_stride), 1) == 0) or (rollout == number_of_rollout - 1)
        if rollout_source == "predicted":
            rho, ux, uy, E = sod_solver.get_macroscopic(Fi0, Gi0)
            T = sod_solver.get_temp_from_energy(ux, uy, E)
        else:
            rho = input_rho_seq[:, rollout]
            ux = input_ux_seq[:, rollout]
            uy = input_uy_seq[:, rollout]
            T = input_T_seq[:, rollout]

        inputs = torch.stack([rho, ux, uy, T], dim=1)
        flat_macro_state = inputs.movedim(1, -1).reshape(-1, inputs.shape[1])
        Geq_target = Geq_seq[:, rollout]
        analytic_feq = sod_solver.get_Feq(rho, ux, uy, T)
        if learn_feq:
            Feq_target = Feq_seq[:, rollout]
            Feq_pred, Geq_pred = model(inputs, basis)
            Feq_pred_grid = reshape_equilibrium_prediction(Feq_pred, Feq_target.shape)
            Feq_collision_grid = build_effective_feq(
                learn_feq=learn_feq,
                feq_mode=feq_mode,
                feq_residual_scale=feq_residual_scale,
                feq_collision_mix=feq_collision_mix,
                project_feq_moments=project_feq_moments,
                feq_nullspace_gamma=feq_nullspace_gamma,
                inputs=inputs,
                basis=basis,
                analytic_feq=analytic_feq,
                feq_target=Feq_target,
                feq_pred_grid=Feq_pred_grid,
            )
            Feq_loss_pred = reshape_equilibrium_target(Feq_collision_grid)
        else:
            Geq_pred = model(inputs, basis)
            Feq_collision_grid = analytic_feq
        Geq_pred_grid = reshape_equilibrium_prediction(Geq_pred, Geq_target.shape)

        if loss_target == "equilibrium":
            if learn_feq:
                loss_feq = loss_func(Feq_loss_pred, reshape_equilibrium_target(Feq_target))
                loss_geq = loss_func(Geq_pred, reshape_equilibrium_target(Geq_target))
                inner_loss = 0.5 * (loss_feq + loss_geq)
            else:
                inner_loss = loss_func(Geq_pred, reshape_equilibrium_target(Geq_target))
        else:
            if current_macro_selected:
                inner_loss = calculate_macro_rollout_loss(
                    loss_func=loss_func,
                    rho_pred=rho,
                    ux_pred=ux,
                    T_pred=T,
                    rho_target=target_rho_seq[:, rollout],
                    ux_target=target_ux_seq[:, rollout],
                    T_target=target_T_seq[:, rollout],
                )
            else:
                inner_loss = torch.zeros((), device=Fi0.device, dtype=Fi0.dtype)
            if equilibrium_anchor_weight > 0.0:
                if learn_feq:
                    loss_feq = loss_func(Feq_loss_pred, reshape_equilibrium_target(Feq_target))
                    loss_geq = loss_func(Geq_pred, reshape_equilibrium_target(Geq_target))
                    inner_loss = inner_loss + equilibrium_anchor_weight * 0.5 * (loss_feq + loss_geq)
                else:
                    inner_loss = inner_loss + equilibrium_anchor_weight * loss_func(
                        Geq_pred,
                        reshape_equilibrium_target(Geq_target),
                    )

        if geq_moment_weight > 0.0:
            geq_moment_loss = calculate_geq_moment_loss(Geq_pred, flat_macro_state, basis, cv=cv)
            inner_loss = inner_loss + geq_moment_weight * geq_moment_loss

        if rollout_source == "predicted":
            collision_feq = Feq_collision_grid
            collision_geq = Geq_pred_grid
            if geq_collision_mix < 1.0:
                collision_geq = geq_collision_mix * Geq_pred_grid + (1.0 - geq_collision_mix) * Geq_target
            Fi0, Gi0 = sod_solver.collision(Fi0, Gi0, collision_feq, collision_geq, rho, ux, uy, T)
            Fi0, Gi0 = sod_solver.streaming(Fi0, Gi0)
            if loss_target != "equilibrium":
                rho_next, ux_next, uy_next, E_next = sod_solver.get_macroscopic(Fi0, Gi0)
                T_next = sod_solver.get_temp_from_energy(ux_next, uy_next, E_next)
                if predicted_macro_selected:
                    inner_loss = calculate_macro_rollout_loss(
                        loss_func=loss_func,
                        rho_pred=rho_next,
                        ux_pred=ux_next,
                        T_pred=T_next,
                        rho_target=target_rho_seq[:, rollout + 1],
                        ux_target=target_ux_seq[:, rollout + 1],
                        T_target=target_T_seq[:, rollout + 1],
                    )
                else:
                    inner_loss = torch.zeros((), device=Fi0.device, dtype=Fi0.dtype)
                if equilibrium_anchor_weight > 0.0:
                    if learn_feq:
                        loss_feq = loss_func(Feq_loss_pred, reshape_equilibrium_target(Feq_target))
                        loss_geq = loss_func(Geq_pred, reshape_equilibrium_target(Geq_target))
                        inner_loss = inner_loss + equilibrium_anchor_weight * 0.5 * (loss_feq + loss_geq)
                    else:
                        inner_loss = inner_loss + equilibrium_anchor_weight * loss_func(
                            Geq_pred,
                            reshape_equilibrium_target(Geq_target),
                        )
                if use_tvd:
                    P_next = rho_next * T_next
                    if rollout > 0:
                        loss_variation = (
                            local_variation_increase_penalty(T_next, T_old)
                            + local_variation_increase_penalty(ux_next, ux_old)
                            + local_variation_increase_penalty(rho_next, rho_old)
                            + local_variation_increase_penalty(P_next, P_old)
                        )
                        loss_curvature = (
                            local_curvature_increase_penalty(T_next, T_old)
                            + local_curvature_increase_penalty(ux_next, ux_old)
                            + local_curvature_increase_penalty(rho_next, rho_old)
                            + local_curvature_increase_penalty(P_next, P_old)
                        )
                        inner_loss = inner_loss + tvd_weight * loss_variation + curvature_weight * loss_curvature
                    ux_old = ux_next.clone()
                    T_old = T_next.clone()
                    rho_old = rho_next.clone()
                    P_old = P_next.clone()
            if backward_chunk_size == 0 and detach_interval and (rollout + 1) % int(detach_interval) == 0 and (rollout + 1) < number_of_rollout:
                Fi0 = Fi0.detach()
                Gi0 = Gi0.detach()
                if rho_old is not None:
                    rho_old = rho_old.detach()
                    ux_old = ux_old.detach()
                    T_old = T_old.detach()
                    P_old = P_old.detach()
        elif use_tvd:
            P = rho * T
            if rollout > 0:
                loss_variation = (
                    local_variation_increase_penalty(T, T_old)
                    + local_variation_increase_penalty(ux, ux_old)
                    + local_variation_increase_penalty(rho, rho_old)
                    + local_variation_increase_penalty(P, P_old)
                )
                loss_curvature = (
                    local_curvature_increase_penalty(T, T_old)
                    + local_curvature_increase_penalty(ux, ux_old)
                    + local_curvature_increase_penalty(rho, rho_old)
                    + local_curvature_increase_penalty(P, P_old)
                )
                inner_loss = inner_loss + tvd_weight * loss_variation + curvature_weight * loss_curvature
            ux_old = ux.clone()
            T_old = T.clone()
            rho_old = rho.clone()
            P_old = P.clone()

        if backward_chunk_size > 0:
            chunk_loss = inner_loss if chunk_loss is None else chunk_loss + inner_loss
            total_loss = total_loss + float(inner_loss.detach())
        else:
            total_loss = total_loss + inner_loss

        reached_chunk_boundary = backward_chunk_size > 0 and (
            current_step % int(backward_chunk_size) == 0 or current_step == number_of_rollout
        )
        if reached_chunk_boundary:
            if not torch.isfinite(chunk_loss):
                return torch.tensor(float("nan"), device=Fi0.device, dtype=Fi0.dtype)
            chunk_loss.backward()
            chunk_loss = None
            Fi0 = Fi0.detach()
            Gi0 = Gi0.detach()
            if rho_old is not None:
                rho_old = rho_old.detach()
                ux_old = ux_old.detach()
                T_old = T_old.detach()
                P_old = P_old.detach()

    if backward_chunk_size > 0:
        return torch.tensor(total_loss, device=Fi0.device, dtype=Fi0.dtype)
    return total_loss

if __name__ == "__main__":
    set_seed(0)

    parser = argparse.ArgumentParser(description='Train Stage 2')
    parser.add_argument('--device', type=int, default=3, help='Device index')
    parser.add_argument("--compile", dest='compile', action='store_true', help='Compile', default=False)
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints (enabled by default)')
    parser.add_argument('--no_save_model', dest='save_model', action='store_false', help='Disable model checkpoint saving')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument("--batch_size", type=int, default=None, help='Rollout batch size')
    parser.add_argument("--save_frequency", type=int, default=1, help='Save model')
    parser.add_argument("--TVD", dest='TVD', action='store_true', help='TVD norm', default=False)
    parser.add_argument("--pre_trained_path", type=str, default=None)
    parser.add_argument("--rollout_source", type=str, choices=["predicted", "solver_macro", "analytic_macro"], default=None)
    parser.add_argument("--loss_target", type=str, choices=["equilibrium", "solver_macro", "analytic_macro"], default=None)
    parser.add_argument("--detach_interval", type=int, default=None, help="Detach rollout state every N predicted steps (0 = full BPTT)")
    parser.add_argument("--epochs_override", type=int, default=None, help='Optional override for stage-2 epochs')
    parser.add_argument("--training_config", type=str, default="Sod_cases_param_training.yml", help='Training YAML path')
    parser.set_defaults(save_model=True)
    args = parser.parse_args()

    device = get_device(args.device)
    if args.pre_trained_path:
        args.pre_trained_path = args.pre_trained_path.replace("SOD_shock_tube/", "")
        print(args.pre_trained_path)
        case_part = args.pre_trained_path.split('/')[1]
        case_number = ''.join(filter(str.isdigit, case_part))
        args.case = int(case_number)
        print(args.case)

    else: 
        args.case = 1

    with open("Sod_cases_param.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    case_params = config[args.case]
    case_params['device'] = device

    print(f"Case {args.case}: SOD shock tube problem")

    sod_solver = SODSolver(
        X=case_params['X'],
        Y=case_params['Y'],
        Qn=case_params['Qn'],
        alpha1=case_params['alpha1'],
        alpha01=case_params['alpha01'],
        vuy=case_params['vuy'],
        Pr=case_params['Pr'],
        muy=case_params['muy'],
        Uax=case_params['Uax'],
        Uay=case_params['Uay'],
        device=case_params['device']
    )

    with open(args.training_config, 'r') as stream:
        training_config = yaml.safe_load(stream)
    param_training = training_config[args.case]
    learn_feq = resolve_learn_feq(param_training, "stage2", default=False)
    number_of_rollout = param_training["stage2"]["N"]
    batch_size = args.batch_size or param_training["stage2"].get("batch_size", 1)
    rollout_source = args.rollout_source or param_training["stage2"].get("rollout_source", "predicted")
    loss_target = args.loss_target or param_training["stage2"].get("loss_target", "equilibrium")
    rollout_start = param_training["stage2"].get("rollout_start", number_of_rollout)
    rollout_warmup_batches = param_training["stage2"].get("rollout_warmup_batches", 0)
    rollout_schedule = param_training["stage2"].get("rollout_schedule", None)
    feq_collision_mix_start = param_training["stage2"].get("feq_collision_mix_start", 1.0)
    feq_collision_mix_warmup_batches = param_training["stage2"].get("feq_collision_mix_warmup_batches", 0)
    geq_collision_mix_start = param_training["stage2"].get("geq_collision_mix_start", 1.0)
    geq_collision_mix_warmup_batches = param_training["stage2"].get("geq_collision_mix_warmup_batches", 0)
    equilibrium_anchor_weight = param_training["stage2"].get("equilibrium_anchor_weight", 0.0)
    geq_moment_weight = param_training["stage2"].get("geq_moment_weight", param_training.get("geq_moment_weight", 0.0))
    grad_clip_norm = param_training["stage2"].get("grad_clip_norm", 1.0)
    skip_nonfinite_batches = param_training["stage2"].get("skip_nonfinite_batches", True)
    plot_frequency = param_training["stage2"].get("plot_frequency", 0)
    benchmark_steps = param_training["stage2"].get("benchmark_steps", min(number_of_rollout, 25))
    long_rollout_eval_enabled = param_training["stage2"].get("long_rollout_eval", True)
    long_rollout_eval_frequency = max(1, int(param_training["stage2"].get("long_rollout_eval_frequency", 1)))
    long_rollout_eval_min_rollout = max(1, int(param_training["stage2"].get("long_rollout_eval_min_rollout", 16)))
    long_rollout_eval_start = max(0, int(param_training["stage2"].get("long_rollout_eval_start", args.num_samples)))
    long_rollout_eval_steps = max(1, int(param_training["stage2"].get("long_rollout_eval_steps", 500)))
    long_rollout_save_top_k = max(1, int(param_training["stage2"].get("long_rollout_save_top_k", 3)))
    tvd_weight = param_training["stage2"].get("tvd_weight", 15.0)
    curvature_weight = param_training["stage2"].get("curvature_weight", 5.0)
    detach_interval = args.detach_interval if args.detach_interval is not None else param_training["stage2"].get("detach_interval", 0)
    loss_stride = param_training["stage2"].get("loss_stride", 1)
    loss_stride_schedule = param_training["stage2"].get("loss_stride_schedule", None)
    logit_clip = param_training.get("logit_clip", 15.0)
    project_feq_moments = param_training.get("project_feq_moments", False)
    enforce_sod_symmetry = param_training.get("enforce_sod_symmetry", False)
    feq_nullspace_gamma = param_training["stage2"].get("feq_nullspace_gamma", param_training.get("feq_nullspace_gamma", 1.0))
    feq_nullspace_gamma_mode = param_training["stage2"].get("feq_nullspace_gamma_mode", param_training.get("feq_nullspace_gamma_mode", "fixed"))
    feq_nullspace_gamma_min = param_training["stage2"].get("feq_nullspace_gamma_min", param_training.get("feq_nullspace_gamma_min", 0.6))
    feq_nullspace_gamma_max = param_training["stage2"].get("feq_nullspace_gamma_max", param_training.get("feq_nullspace_gamma_max", 1.0))
    geq_mode = param_training["stage2"].get("geq_mode", param_training.get("geq_mode", "learned"))
    geq_nullspace_gamma = param_training["stage2"].get("geq_nullspace_gamma", param_training.get("geq_nullspace_gamma", 1.0))
    geq_nullspace_gamma_mode = param_training["stage2"].get("geq_nullspace_gamma_mode", param_training.get("geq_nullspace_gamma_mode", "fixed"))
    geq_nullspace_gamma_min = param_training["stage2"].get("geq_nullspace_gamma_min", param_training.get("geq_nullspace_gamma_min", 0.6))
    geq_nullspace_gamma_max = param_training["stage2"].get("geq_nullspace_gamma_max", param_training.get("geq_nullspace_gamma_max", 1.0))
    feq_mode = param_training["stage2"].get("feq_mode", "learned")
    feq_residual_scale = param_training["stage2"].get("feq_residual_scale", 1.0)

    if "TVD" in param_training["stage2"]:
        args.TVD = True


    print(f"TVD Enabled: {args.TVD}")
    print(f"Rollout Source: {rollout_source}")
    print(f"Loss Target: {loss_target}")
    if learn_feq:
        print(
            f"Feq Path: learned ({feq_mode}, residual_scale={feq_residual_scale}, "
            f"gamma_mode={feq_nullspace_gamma_mode}, gamma={feq_nullspace_gamma}, "
            f"bounds=({feq_nullspace_gamma_min}, {feq_nullspace_gamma_max}))"
        )
    else:
        print("Feq Path: analytic (main-branch-compatible)")
    print(
        f"Geq Mode: {geq_mode} "
        f"(gamma_mode={geq_nullspace_gamma_mode}, gamma={geq_nullspace_gamma}, "
        f"bounds=({geq_nullspace_gamma_min}, {geq_nullspace_gamma_max}))"
    )
    print(f"SOD Symmetry Enabled: {enforce_sod_symmetry}")
    print(f"Rollout Warmup: start={rollout_start}, warmup_batches={rollout_warmup_batches}, target={number_of_rollout}")
    print(
        f"Long Rollout Eval: enabled={long_rollout_eval_enabled}, "
        f"start={long_rollout_eval_start}, steps={long_rollout_eval_steps}, "
        f"frequency={long_rollout_eval_frequency}, min_rollout={long_rollout_eval_min_rollout}, "
        f"top_k={long_rollout_save_top_k}"
    )
    if rollout_schedule:
        print(f"Rollout Schedule: {rollout_schedule}")
    if loss_stride_schedule:
        print(f"Loss Stride Schedule: {loss_stride_schedule}")
    print(
        f"Collision Warmup: feq_start={feq_collision_mix_start}, feq_batches={feq_collision_mix_warmup_batches}, "
        f"geq_start={geq_collision_mix_start}, geq_batches={geq_collision_mix_warmup_batches}, "
        f"eq_anchor={equilibrium_anchor_weight}, geq_moment_weight={geq_moment_weight}, "
        f"grad_clip={grad_clip_norm}, detach_interval={detach_interval}, "
        f"loss_stride={loss_stride}, tvd_weight={tvd_weight}, curvature_weight={curvature_weight}"
    )

    if args.save_model:
        os.makedirs(param_training["stage2"]["model_dir"], exist_ok=True)
    with h5py.File(param_training["data_dir"], "r") as handle:
        all_F = handle["Fi0"][:]
        all_G = handle["Gi0"][:]
        all_Feq = handle["Feq"][:]
        all_Geq = handle["Geq"][:]
        solver_rho = handle["rho"][:]
        solver_ux = handle["ux"][:]
        solver_uy = handle["uy"][:]
        solver_T = handle["T"][:]
        analytic_rho = handle["analytic_rho"][:] if "analytic_rho" in handle else None
        analytic_ux = handle["analytic_ux"][:] if "analytic_ux" in handle else None
        analytic_T = handle["analytic_T"][:] if "analytic_T" in handle else None
        analytic_P = handle["analytic_P"][:] if "analytic_P" in handle else None

    if rollout_source == "analytic_macro":
        if analytic_rho is None or analytic_ux is None or analytic_T is None:
            raise ValueError("rollout_source=analytic_macro requires analytic fields in the dataset.")
        input_macro_rho = analytic_rho
        input_macro_ux = analytic_ux
        input_macro_uy = np.zeros_like(analytic_ux)
        input_macro_T = analytic_T
    else:
        input_macro_rho = solver_rho
        input_macro_ux = solver_ux
        input_macro_uy = solver_uy
        input_macro_T = solver_T

    if loss_target == "analytic_macro":
        if analytic_rho is None or analytic_ux is None or analytic_T is None:
            raise ValueError("loss_target=analytic_macro requires analytic fields in the dataset.")
        target_macro_rho = analytic_rho
        target_macro_ux = analytic_ux
        target_macro_uy = np.zeros_like(analytic_ux)
        target_macro_T = analytic_T
    else:
        target_macro_rho = solver_rho
        target_macro_ux = solver_ux
        target_macro_uy = solver_uy
        target_macro_T = solver_T

    dataset = RolloutMacroBatchDataset(
        all_Fi=all_F[:args.num_samples],
        all_Gi=all_G[:args.num_samples],
        all_Feq=all_Feq[:args.num_samples],
        all_Geq=all_Geq[:args.num_samples],
        input_rho=input_macro_rho[:args.num_samples],
        input_ux=input_macro_ux[:args.num_samples],
        input_uy=input_macro_uy[:args.num_samples],
        input_T=input_macro_T[:args.num_samples],
        number_of_rollout=number_of_rollout,
        target_rho=target_macro_rho[:args.num_samples],
        target_ux=target_macro_ux[:args.num_samples],
        target_uy=target_macro_uy[:args.num_samples],
        target_T=target_macro_T[:args.num_samples],
    )

    use_workers = 4 if str(device).startswith("cuda") else 0
    use_pin_memory = str(device).startswith("cuda")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=use_workers, pin_memory=use_pin_memory)

    val_window_count = 100
    val_span = number_of_rollout + val_window_count
    val_stop = min(len(all_F), args.num_samples + val_span)
    val_dataset = RolloutMacroBatchDataset(
        all_Fi=all_F[args.num_samples:val_stop],
        all_Gi=all_G[args.num_samples:val_stop],
        all_Feq=all_Feq[args.num_samples:val_stop],
        all_Geq=all_Geq[args.num_samples:val_stop],
        input_rho=input_macro_rho[args.num_samples:val_stop],
        input_ux=input_macro_ux[args.num_samples:val_stop],
        input_uy=input_macro_uy[args.num_samples:val_stop],
        input_T=input_macro_T[args.num_samples:val_stop],
        number_of_rollout=number_of_rollout,
        target_rho=target_macro_rho[args.num_samples:val_stop],
        target_ux=target_macro_ux[args.num_samples:val_stop],
        target_uy=target_macro_uy[args.num_samples:val_stop],
        target_T=target_macro_T[args.num_samples:val_stop],
    )
    if len(val_dataset) == 0:
        raise ValueError(
            f"Validation dataset is empty for number_of_rollout={number_of_rollout}. "
            f"Need at least {number_of_rollout + 1} validation snapshots."
        )
    val_batch_size = max(1, min(batch_size, len(val_dataset)))
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    
    model = NeurDE(
        alpha_layer=[4] + [param_training["hidden_dim"]] * param_training["num_layers"],
        phi_layer=[2] + [param_training["hidden_dim"]] * param_training["num_layers"],
        activation='relu',
        learn_feq=learn_feq,
        learn_geq=True,
        logit_clip=logit_clip,
        project_feq_moments=project_feq_moments,
        enforce_sod_symmetry=enforce_sod_symmetry,
        feq_nullspace_gamma=feq_nullspace_gamma,
        feq_nullspace_gamma_mode=feq_nullspace_gamma_mode,
        feq_nullspace_gamma_min=feq_nullspace_gamma_min,
        feq_nullspace_gamma_max=feq_nullspace_gamma_max,
        geq_mode=geq_mode,
        geq_cv=sod_solver.Cv,
        geq_nullspace_gamma=geq_nullspace_gamma,
        geq_nullspace_gamma_mode=geq_nullspace_gamma_mode,
        geq_nullspace_gamma_min=geq_nullspace_gamma_min,
        geq_nullspace_gamma_max=geq_nullspace_gamma_max,
    ).to(device)

    if args.compile:
        model = torch.compile(model)
        sod_solver.collision = torch.compile(sod_solver.collision, dynamic=True, fullgraph=False)
        sod_solver.streaming = torch.compile(sod_solver.streaming, dynamic=True, fullgraph=False)
        sod_solver.shift_operator = torch.compile(sod_solver.shift_operator, dynamic=True, fullgraph=False)
        sod_solver.get_macroscopic = torch.compile(sod_solver.get_macroscopic, dynamic=True, fullgraph=False)
        sod_solver.get_Feq = torch.compile(sod_solver.get_Feq, dynamic=True, fullgraph=False)
        sod_solver.get_temp_from_energy = torch.compile(sod_solver.get_temp_from_energy, dynamic=True, fullgraph=False)
        print("Model compiled.")

    if args.pre_trained_path:
        if args.compile:
            missing_keys, unexpected_keys = load_model_checkpoint(model, args.pre_trained_path, map_location=device)
        elif not args.compile:
            missing_keys, unexpected_keys = load_model_checkpoint(model, args.pre_trained_path, map_location=device)
        print(f"Pre-trained model loaded from {args.pre_trained_path}")
        if missing_keys or unexpected_keys:
            print(f"Checkpoint mismatch. Missing: {missing_keys}, Unexpected: {unexpected_keys}")


    optimizer = dispatch_optimizer(model=model,
                                    lr=param_training["stage2"]["lr"],
                                    optimizer_type="AdamW")

    epochs = args.epochs_override or param_training["stage2"]["epochs"]
    total_steps = len(dataloader) * epochs
    scheduler_type = param_training["stage2"]["scheduler"]
    scheduler_config = param_training["stage2"].get("scheduler_config", {}).get(scheduler_type, {})
    scheduler = get_scheduler(optimizer, scheduler_type, total_steps, scheduler_config)
    
    Uax, Uay = case_params["Uax"], case_params["Uay"]
    basis = create_basis(Uax, Uay, device)

    loss_func = calculate_relative_error

    print(
        f"Training Case {args.case} on {device}. Epochs: {epochs}, Samples: {args.num_samples}, "
        f"batch_size={batch_size}, learn_feq={learn_feq}, geq_mode={geq_mode}, "
        f"geq_gamma_mode={geq_nullspace_gamma_mode}"
    )

    best_losses = [float('inf')] * 3
    best_models = [None] * 3
    best_model_paths = [None] * 3
    best_long_scores = [float('inf')] * long_rollout_save_top_k
    best_long_models = [None] * long_rollout_save_top_k
    best_long_paths = [None] * long_rollout_save_top_k
    active_validation_phase = None

    save_frequency = args.save_frequency
    epochs_since_last_save = [0] * 3
    last_epoch_loss = 0.0

    if args.TVD:
        print("Using TVD")
        if args.compile:
            TVD_norm = torch.compile(TVD_norm, dynamic=True, fullgraph=False)
    current_loss = 0.0
    global_batch_step = 0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        loss_epoch = 0
        for batch_idx, (
            F_seq,
            G_seq,
            Feq_seq,
            Geq_seq,
            input_rho_seq,
            input_ux_seq,
            input_uy_seq,
            input_T_seq,
            target_rho_seq,
            target_ux_seq,
            target_uy_seq,
            target_T_seq,
        ) in enumerate(dataloader):
            optimizer.zero_grad()
            model.train()
            F_seq = F_seq.to(device)
            G_seq = G_seq.to(device)
            Feq_seq = Feq_seq.to(device)
            Geq_seq = Geq_seq.to(device)
            input_rho_seq = input_rho_seq.to(device)
            input_ux_seq = input_ux_seq.to(device)
            input_uy_seq = input_uy_seq.to(device)
            input_T_seq = input_T_seq.to(device)
            target_rho_seq = target_rho_seq.to(device)
            target_ux_seq = target_ux_seq.to(device)
            target_uy_seq = target_uy_seq.to(device)
            target_T_seq = target_T_seq.to(device)
            current_rollout = resolve_rollout_warmup_count(
                max_rollout=number_of_rollout,
                start_rollout=rollout_start,
                warmup_batches=rollout_warmup_batches,
                global_batch_step=global_batch_step,
            )
            current_rollout = resolve_rollout_schedule(
                schedule=rollout_schedule,
                fallback_rollout=current_rollout,
                global_batch_step=global_batch_step,
            )
            current_loss_stride = resolve_stride_schedule(
                schedule=loss_stride_schedule,
                fallback_stride=loss_stride,
                global_batch_step=global_batch_step,
            )
            current_feq_collision_mix = 1.0 if not learn_feq else resolve_schedule_value(
                target_value=1.0,
                start_value=feq_collision_mix_start,
                warmup_batches=feq_collision_mix_warmup_batches,
                global_batch_step=global_batch_step,
            )
            current_geq_collision_mix = resolve_schedule_value(
                target_value=1.0,
                start_value=geq_collision_mix_start,
                warmup_batches=geq_collision_mix_warmup_batches,
                global_batch_step=global_batch_step,
            )
            chunked_backward = rollout_source == "predicted" and int(detach_interval) > 0
            total_loss = compute_stage2_rollout_loss(
                model=model,
                sod_solver=sod_solver,
                basis=basis,
                F_seq=F_seq,
                G_seq=G_seq,
                Feq_seq=Feq_seq,
                Geq_seq=Geq_seq,
                input_rho_seq=input_rho_seq,
                input_ux_seq=input_ux_seq,
                input_uy_seq=input_uy_seq,
                input_T_seq=input_T_seq,
                target_rho_seq=target_rho_seq,
                target_ux_seq=target_ux_seq,
                target_uy_seq=target_uy_seq,
                target_T_seq=target_T_seq,
                number_of_rollout=current_rollout,
                learn_feq=learn_feq,
                loss_func=loss_func,
                rollout_source=rollout_source,
                loss_target=loss_target,
                use_tvd=args.TVD,
                tvd_weight=tvd_weight if args.TVD else 0.0,
                curvature_weight=curvature_weight if args.TVD else 0.0,
                equilibrium_anchor_weight=equilibrium_anchor_weight,
                geq_moment_weight=geq_moment_weight,
                feq_mode=feq_mode,
                feq_residual_scale=feq_residual_scale,
                project_feq_moments=project_feq_moments,
                feq_nullspace_gamma=feq_nullspace_gamma,
                cv=sod_solver.Cv,
                feq_collision_mix=current_feq_collision_mix,
                geq_collision_mix=current_geq_collision_mix,
                detach_interval=detach_interval,
                loss_stride=current_loss_stride,
                backward_chunk_size=detach_interval if chunked_backward else 0,
            )
            if skip_nonfinite_batches and not torch.isfinite(total_loss):
                print(
                    f"Skipping non-finite batch at epoch={epoch}, batch={batch_idx}, "
                    f"rollout={current_rollout}, feq_mix={current_feq_collision_mix:.3f}, geq_mix={current_geq_collision_mix:.3f}"
                )
                optimizer.zero_grad(set_to_none=True)
                global_batch_step += 1
                continue
            if not chunked_backward:
                total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            loss_epoch += total_loss.item()
            global_batch_step += 1
            print(
                f"Epoch: {epoch}, Batch ID: {batch_idx}, Rollout: {current_rollout}, "
                f"Stride: {current_loss_stride}, FeqMix: {current_feq_collision_mix:.3f}, GeqMix: {current_geq_collision_mix:.3f}, "
                f"Loss: {total_loss.item()/max(current_rollout, 1):.6f}"
            )

        scheduler.step()

        current_loss = loss_epoch / len(dataloader)
        validation_rollout = current_rollout
        validation_loss_stride = current_loss_stride
        validation_feq_collision_mix = current_feq_collision_mix
        validation_geq_collision_mix = current_geq_collision_mix
        validation_phase = (int(validation_rollout), int(validation_loss_stride))

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")

        
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (
                F_val,
                G_val,
                Feq_val,
                Geq_val,
                input_rho_val,
                input_ux_val,
                input_uy_val,
                input_T_val,
                target_rho_val,
                target_ux_val,
                target_uy_val,
                target_T_val,
            ) in val_dataloader:
                val_loss += compute_stage2_rollout_loss(
                    model=model,
                    sod_solver=sod_solver,
                    basis=basis,
                    F_seq=F_val.to(device),
                    G_seq=G_val.to(device),
                    Feq_seq=Feq_val.to(device),
                    Geq_seq=Geq_val.to(device),
                    input_rho_seq=input_rho_val.to(device),
                    input_ux_seq=input_ux_val.to(device),
                    input_uy_seq=input_uy_val.to(device),
                    input_T_seq=input_T_val.to(device),
                    target_rho_seq=target_rho_val.to(device),
                    target_ux_seq=target_ux_val.to(device),
                    target_uy_seq=target_uy_val.to(device),
                    target_T_seq=target_T_val.to(device),
                    number_of_rollout=validation_rollout,
                    learn_feq=learn_feq,
                    loss_func=loss_func,
                    rollout_source=rollout_source,
                    loss_target=loss_target,
                    use_tvd=False,
                    tvd_weight=0.0,
                    curvature_weight=0.0,
                    equilibrium_anchor_weight=0.0,
                    geq_moment_weight=geq_moment_weight,
                    feq_mode=feq_mode,
                    feq_residual_scale=feq_residual_scale,
                    project_feq_moments=project_feq_moments,
                    feq_nullspace_gamma=feq_nullspace_gamma,
                    cv=sod_solver.Cv,
                    feq_collision_mix=validation_feq_collision_mix,
                    geq_collision_mix=validation_geq_collision_mix,
                    detach_interval=detach_interval,
                    loss_stride=validation_loss_stride,
                ).item()
            val_loss /= max(len(val_dataloader), 1)
            validation_supervised_steps = count_supervised_steps(validation_rollout, validation_loss_stride)
            val_score = val_loss / max(validation_supervised_steps, 1)
            print("-" * 50)
            print(
                f"Validation Loss: {val_loss:.6f} "
                f"(per_step={val_score:.6f}, "
                f"rollout={validation_rollout}, stride={validation_loss_stride}, "
                f"supervised_steps={validation_supervised_steps}, "
                f"feq_mix={validation_feq_collision_mix:.3f}, geq_mix={validation_geq_collision_mix:.3f})"
            )
            print("-" * 50)

            if plot_frequency and ((epoch + 1) % plot_frequency == 0 or epoch == 0):
                benchmark_start = min(args.num_samples, max(0, len(all_F) - benchmark_steps))
                effective_benchmark_steps = min(benchmark_steps, len(all_F) - benchmark_start, validation_rollout)
                use_analytic_reference = loss_target == "analytic_macro" and analytic_rho is not None
                benchmark_rho = analytic_rho if use_analytic_reference else solver_rho
                benchmark_ux = analytic_ux if use_analytic_reference else solver_ux
                benchmark_T = analytic_T if use_analytic_reference else solver_T
                benchmark_P = analytic_P if use_analytic_reference and analytic_P is not None else benchmark_rho * benchmark_T
                benchmark_reference = {
                    "F0": torch.tensor(all_F[benchmark_start], dtype=torch.get_default_dtype(), device=device),
                    "G0": torch.tensor(all_G[benchmark_start], dtype=torch.get_default_dtype(), device=device),
                    "rho": torch.tensor(benchmark_rho[benchmark_start:benchmark_start + effective_benchmark_steps], dtype=torch.get_default_dtype(), device=device),
                    "ux": torch.tensor(benchmark_ux[benchmark_start:benchmark_start + effective_benchmark_steps], dtype=torch.get_default_dtype(), device=device),
                    "T": torch.tensor(benchmark_T[benchmark_start:benchmark_start + effective_benchmark_steps], dtype=torch.get_default_dtype(), device=device),
                    "P": torch.tensor(benchmark_P[benchmark_start:benchmark_start + effective_benchmark_steps], dtype=torch.get_default_dtype(), device=device),
                    "label": "analytic" if use_analytic_reference else "solver",
                }
                benchmark_loss, nn_elapsed, image_path = run_training_benchmark(
                    model=model,
                    sod_solver=sod_solver,
                    basis=basis,
                    reference=benchmark_reference,
                    image_dir=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "images",
                        f"SOD_case{args.case}",
                        "training",
                        os.path.basename(param_training["stage2"]["model_dir"]),
                    ),
                    epoch=epoch,
                    learn_feq=learn_feq,
                    benchmark_steps=effective_benchmark_steps,
                    feq_mode=feq_mode,
                    feq_residual_scale=feq_residual_scale,
                    project_feq_moments=project_feq_moments,
                    feq_nullspace_gamma=feq_nullspace_gamma,
                    feq_collision_mix=validation_feq_collision_mix,
                )
                print(f"Benchmark rollout error: {benchmark_loss:.6f}, nn_time={nn_elapsed:.4f}s, plot={image_path}")

        if validation_phase != active_validation_phase:
            active_validation_phase = validation_phase
            best_losses = [float('inf')] * 3
            best_models = [None] * 3
            best_model_paths = [None] * 3
            epochs_since_last_save = [0] * 3
            print(
                f"Checkpoint ranking reset for new curriculum phase: "
                f"rollout={validation_rollout}, stride={validation_loss_stride}"
            )

        if not long_rollout_eval_enabled:
            if val_score < max(best_losses):
                max_index = best_losses.index(max(best_losses))
                best_losses[max_index] = val_score
                best_models[max_index] = model.state_dict()

                if args.save_model and epochs_since_last_save[max_index] >= save_frequency:
                    if best_model_paths[max_index] and os.path.exists(best_model_paths[max_index]):
                        os.remove(best_model_paths[max_index])
                    save_path = os.path.join(
                        param_training["stage2"]["model_dir"],
                        f"best_model_{args.case}_epoch_{epoch+1}_r{validation_rollout}_s{validation_loss_stride}_top_{max_index+1}_val_score_{val_score:.6f}.pt",
                    )
                    torch.save(best_models[max_index], save_path)
                    print(f"Top {max_index+1} model saved to: {save_path}")
                    best_model_paths[max_index] = save_path
                    epochs_since_last_save[max_index] = 0
                else:
                    epochs_since_last_save[max_index] += 1
            else:
                for i in range(3):
                    epochs_since_last_save[i] += 1
        else:
            should_run_long_rollout_eval = (
                validation_rollout >= long_rollout_eval_min_rollout
                and ((epoch + 1) % long_rollout_eval_frequency == 0)
            )
            if should_run_long_rollout_eval:
                eval_start = min(long_rollout_eval_start, max(0, len(all_F) - 1))
                eval_steps = min(long_rollout_eval_steps, len(all_F) - eval_start)
                long_metrics = run_long_rollout_validation(
                    model=model,
                    sod_solver=sod_solver,
                    basis=basis,
                    all_F=all_F,
                    all_G=all_G,
                    all_Feq=all_Feq,
                    all_rho=solver_rho,
                    all_ux=solver_ux,
                    all_T=solver_T,
                    analytic_rho=analytic_rho,
                    analytic_ux=analytic_ux,
                    analytic_T=analytic_T,
                    analytic_P=analytic_P,
                    device=device,
                    start_idx=eval_start,
                    eval_steps=eval_steps,
                    learn_feq=learn_feq,
                    feq_mode=feq_mode,
                    feq_residual_scale=feq_residual_scale,
                    project_feq_moments=project_feq_moments,
                    feq_nullspace_gamma=feq_nullspace_gamma,
                    feq_collision_mix=validation_feq_collision_mix,
                )
                if long_metrics["finite"]:
                    long_rollout_score = (
                        long_metrics["avg_analytic"]
                        if np.isfinite(long_metrics["avg_analytic"])
                        else long_metrics["avg_solver"]
                    )
                    print(
                        f"Long Rollout Eval ({eval_start}->{eval_start + eval_steps - 1}): "
                        f"avg_solver={long_metrics['avg_solver']:.6f}, final_solver={long_metrics['final_solver']:.6f}, "
                        f"avg_analytic={long_metrics['avg_analytic']:.6f}, final_analytic={long_metrics['final_analytic']:.6f}"
                    )
                    if long_rollout_score < max(best_long_scores):
                        max_index = best_long_scores.index(max(best_long_scores))
                        best_long_scores[max_index] = long_rollout_score
                        best_long_models[max_index] = model.state_dict()
                        if args.save_model:
                            if best_long_paths[max_index] and os.path.exists(best_long_paths[max_index]):
                                os.remove(best_long_paths[max_index])
                            save_path = os.path.join(
                                param_training["stage2"]["model_dir"],
                                f"best_model_{args.case}_epoch_{epoch+1}_r{validation_rollout}_s{validation_loss_stride}_top_{max_index+1}_long_score_{long_rollout_score:.6f}.pt",
                            )
                            torch.save(best_long_models[max_index], save_path)
                            print(f"Top {max_index+1} long-rollout model saved to: {save_path}")
                            best_long_paths[max_index] = save_path
                else:
                    print(
                        f"Long Rollout Eval ({eval_start}->{eval_start + eval_steps - 1}) failed: "
                        f"non-finite at step {long_metrics['fail_step']}"
                    )

        if args.save_model and epoch % 200 == 0:
            print(f"Epoch: {epoch}, Loss: {current_loss:.6f}")
            save_path = os.path.join(param_training["stage2"]["model_dir"], f"model_{args.case}_epoch_{epoch}_loss_{val_loss:.6f}.pt")
            torch.save(model.state_dict(), save_path)
            


    # Save the last model with its loss
    if args.save_model:
        last_epoch_loss = current_loss
        last_model_path = os.path.join(param_training["stage2"]["model_dir"], f"last_model_{args.case}_epoch_{epochs}_loss_{last_epoch_loss:.6f}.pt")
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to: {last_model_path}")
    print("Training complete.")
