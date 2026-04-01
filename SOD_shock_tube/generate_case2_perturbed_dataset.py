#!/usr/bin/env python
import argparse
import os

import h5py
import torch
import yaml
from tqdm import tqdm

from SOD_solver import SODSolver
from utilities import detach, get_device, set_seed


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_perturbed_case2_initial_conditions(
    sod_solver,
    rho_left_factor=1.0,
    rho_right_factor=1.0,
    pressure_left_factor=1.0,
    pressure_right_factor=1.0,
):
    rho_max = 1.0
    p_max = 0.2

    ux0 = torch.zeros((sod_solver.Y, sod_solver.X), device=sod_solver.device)
    uy0 = torch.zeros((sod_solver.Y, sod_solver.X), device=sod_solver.device)
    rho0 = torch.ones((sod_solver.Y, sod_solver.X), device=sod_solver.device)
    p0 = torch.zeros((sod_solver.Y, sod_solver.X), device=sod_solver.device)

    rho0[:, : sod_solver.Lx + 1] = 1.0 * rho_max * rho_left_factor
    rho0[:, sod_solver.Lx + 1 :] = 0.125 * rho_max * rho_right_factor
    p0[:, : sod_solver.Lx + 1] = 1.0 * p_max * pressure_left_factor
    p0[:, sod_solver.Lx + 1 :] = 0.1 * p_max * pressure_right_factor
    t0 = p0 / (rho0 * sod_solver.R)

    khi0 = torch.zeros((sod_solver.Y, sod_solver.X)).cpu().numpy()
    zetax0 = torch.zeros((sod_solver.Y, sod_solver.X)).cpu().numpy()
    zetay0 = torch.zeros((sod_solver.Y, sod_solver.X)).cpu().numpy()

    fi0 = sod_solver.get_Feq(rho0, ux0, uy0, t0).to(sod_solver.device)
    gi0, khi0, zetax0, zetay0 = sod_solver.get_Geq_Newton_solver(
        rho0, ux0, uy0, t0, khi0, zetax0, zetay0
    )
    gi0 = gi0.to(sod_solver.device)
    return fi0, gi0, khi0, zetax0, zetay0


def main():
    parser = argparse.ArgumentParser(description="Generate a perturbed case-2 SOD dataset")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1002)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--muy_override", type=float, default=None)
    parser.add_argument("--rho_left_factor", type=float, default=1.0)
    parser.add_argument("--rho_right_factor", type=float, default=1.0)
    parser.add_argument("--pressure_left_factor", type=float, default=1.0)
    parser.add_argument("--pressure_right_factor", type=float, default=1.0)
    parser.add_argument("--uax_override", type=float, default=None)
    parser.add_argument("--uay_override", type=float, default=None)
    args = parser.parse_args()

    set_seed(0)
    device = get_device(args.device)

    with open(os.path.join(SCRIPT_DIR, "Sod_cases_param.yml")) as f:
        case_params = yaml.safe_load(f)[2]

    case_params["device"] = device
    if args.muy_override is not None:
        case_params["muy"] = float(args.muy_override)
    if args.uax_override is not None:
        case_params["Uax"] = float(args.uax_override)
    if args.uay_override is not None:
        case_params["Uay"] = float(args.uay_override)

    sod_solver = SODSolver(
        X=case_params["X"],
        Y=case_params["Y"],
        Qn=case_params["Qn"],
        alpha1=case_params["alpha1"],
        alpha01=case_params["alpha01"],
        vuy=case_params["vuy"],
        Pr=case_params["Pr"],
        muy=case_params["muy"],
        Uax=case_params["Uax"],
        Uay=case_params["Uay"],
        device=case_params["device"],
    )

    fi0, gi0, khi0, zetax0, zetay0 = build_perturbed_case2_initial_conditions(
        sod_solver,
        rho_left_factor=args.rho_left_factor,
        rho_right_factor=args.rho_right_factor,
        pressure_left_factor=args.pressure_left_factor,
        pressure_right_factor=args.pressure_right_factor,
    )

    all_rho = []
    all_ux = []
    all_uy = []
    all_t = []
    all_feq = []
    all_geq = []
    all_fi0 = []
    all_gi0 = []

    print(
        "Generating perturbed case-2 dataset with "
        f"mu={case_params['muy']}, Uax={case_params['Uax']}, "
        f"rho_left_factor={args.rho_left_factor}, rho_right_factor={args.rho_right_factor}, "
        f"pressure_left_factor={args.pressure_left_factor}, "
        f"pressure_right_factor={args.pressure_right_factor}"
    )

    with torch.no_grad():
        for _ in tqdm(range(args.steps)):
            rho, ux, uy, e = sod_solver.get_macroscopic(fi0, gi0)
            t = sod_solver.get_temp_from_energy(ux, uy, e)
            feq = sod_solver.get_Feq(rho, ux, uy, t)
            geq, khi, zetax, zetay = sod_solver.get_Geq_Newton_solver(rho, ux, uy, t, khi0, zetax0, zetay0)

            all_fi0.append(detach(fi0))
            all_gi0.append(detach(gi0))
            all_rho.append(detach(rho))
            all_ux.append(detach(ux))
            all_uy.append(detach(uy))
            all_t.append(detach(t))
            all_feq.append(detach(feq))
            all_geq.append(detach(geq))

            fi0, gi0 = sod_solver.collision(fi0, gi0, feq, geq, rho, ux, uy, t)
            fi0, gi0 = sod_solver.streaming(fi0, gi0)
            khi0 = khi
            zetax0 = zetax
            zetay0 = zetay

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with h5py.File(args.output_path, "w") as f:
        f.create_dataset("rho", data=all_rho)
        f.create_dataset("ux", data=all_ux)
        f.create_dataset("uy", data=all_uy)
        f.create_dataset("T", data=all_t)
        f.create_dataset("Feq", data=all_feq)
        f.create_dataset("Geq", data=all_geq)
        f.create_dataset("Fi0", data=all_fi0)
        f.create_dataset("Gi0", data=all_gi0)
        f.attrs["case"] = 2
        f.attrs["muy"] = float(case_params["muy"])
        f.attrs["Uax"] = float(case_params["Uax"])
        f.attrs["Uay"] = float(case_params["Uay"])
        f.attrs["rho_left_factor"] = float(args.rho_left_factor)
        f.attrs["rho_right_factor"] = float(args.rho_right_factor)
        f.attrs["pressure_left_factor"] = float(args.pressure_left_factor)
        f.attrs["pressure_right_factor"] = float(args.pressure_right_factor)

    print(f"Saved perturbed dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
