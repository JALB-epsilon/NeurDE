import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def naca0012_polygon(num_points=300):
    x = np.linspace(0.0, 1.0, num_points)
    t = 0.12
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.clip(x, 1e-12, None))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1], -yt[::-1]])
    return np.vstack([upper, lower])


def draw_case(ax, name, case_cfg, related_bc_text):
    domain_x = float(case_cfg["domain"]["extent_in_chords"]["x"])
    domain_y = float(case_cfg["domain"]["extent_in_chords"]["y"])
    head_x = float(case_cfg["aerofoil_position"]["head_x_over_C"])
    head_y = case_cfg["aerofoil_position"]["head_y_over_C"]
    if head_y is None:
        head_y = float(case_cfg["aerofoil_position"].get("visualization_head_y_over_C", 0.5 * domain_y))

    airfoil = naca0012_polygon()
    airfoil_x = head_x + airfoil[:, 0]
    airfoil_y = head_y + airfoil[:, 1]

    ax.add_patch(plt.Rectangle((0.0, 0.0), domain_x, domain_y, fill=False, linewidth=2.0, color="black"))
    ax.fill(airfoil_x, airfoil_y, color="#aec7e8", edgecolor="#1f4e79", linewidth=1.5, zorder=3)

    ax.annotate("", xy=(0.0, -0.7), xytext=(domain_x, -0.7), arrowprops=dict(arrowstyle="<->", lw=1.4))
    ax.text(0.5 * domain_x, -1.05, f"{domain_x:g}C", ha="center", va="top", fontsize=10)
    ax.annotate("", xy=(-0.45, 0.0), xytext=(-0.45, domain_y), arrowprops=dict(arrowstyle="<->", lw=1.4))
    ax.text(-0.72, 0.5 * domain_y, f"{domain_y:g}C", ha="center", va="center", rotation=90, fontsize=10)

    ax.plot([head_x, head_x], [0.0, domain_y], linestyle="--", linewidth=1.0, color="#555555")
    ax.text(head_x, domain_y + 0.35, f"head x = {head_x:g}C", ha="center", va="bottom", fontsize=9)

    bc = case_cfg["boundary_conditions"]
    ax.text(
        -0.15,
        0.5 * domain_y,
        "Inlet\nDirichlet\nu, T, rho",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#0d47a1",
    )
    ax.text(
        domain_x + 0.15,
        0.5 * domain_y,
        "Outlet\nNeumann\nu, T, rho",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#8e24aa",
    )
    ax.text(
        0.5 * domain_x,
        domain_y + 0.15,
        "Top freestream",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#00695c",
    )
    ax.text(
        0.5 * domain_x,
        -0.18,
        "Bottom freestream",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#00695c",
    )
    ax.text(
        head_x + 0.5,
        head_y + 0.55,
        "Aerofoil wall\nno-slip\nT/rho: no-penetration",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#b22222",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"),
    )
    ax.text(
        domain_x + 0.35,
        0.72 * domain_y,
        related_bc_text,
        ha="left",
        va="center",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#cccccc"),
    )

    case_lines = [
        f"{name.capitalize()}",
        f"Ma∞ = {case_cfg['Ma_inf']}",
        f"Re = {case_cfg.get('Re', case_cfg.get('Re_range'))}",
        f"alpha = {case_cfg['angle_of_attack_deg']} deg",
        f"U = {case_cfg['shifted_velocity']}",
        case_cfg["aerofoil_position"]["note"],
    ]
    if "chord_resolution" in case_cfg["domain"]:
        case_lines.append(f"C = {case_cfg['domain']['chord_resolution']}")
    if "chord_resolutions" in case_cfg["domain"]:
        case_lines.append(f"C = {case_cfg['domain']['chord_resolutions']}")
    ax.text(
        domain_x + 0.35,
        0.2 * domain_y,
        "\n".join(case_lines),
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )

    ax.set_xlim(-1.0, domain_x + 4.6)
    ax.set_ylim(-1.4, domain_y + 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name.capitalize()} benchmark setup", fontsize=12, fontweight="bold")


def main():
    with open("airfoil_param.yml", "r") as stream:
        config = yaml.safe_load(stream)

    scenarios = config["benchmark_scenarios"]
    related_bc = config["related_literature"]["related_open_bc_family"]["benchmark"]["boundary_conditions"]
    related_bc_text = (
        "Related open-literature BC family\n"
        f"left: {related_bc['left']}\n"
        f"right: {related_bc['right']}\n"
        f"top/bottom: {related_bc['top']}\n"
        f"wall: {related_bc['wall']}"
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    draw_case(axes[0], "transonic", scenarios["transonic"], related_bc_text)
    draw_case(axes[1], "supersonic", scenarios["supersonic"], related_bc_text)

    fig.suptitle(
        "NACA0012 aerofoil benchmark setup\n"
        "Current YAML uses the user-supplied paper benchmark excerpt; related open-literature BC family shown separately",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("images", exist_ok=True)
    plt.savefig("images/naca0012_benchmark_setup.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
