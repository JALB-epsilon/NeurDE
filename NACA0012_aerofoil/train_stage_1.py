import yaml


if __name__ == "__main__":
    with open("airfoil_param.yml", "r") as stream:
        case_params = yaml.safe_load(stream)

    raise NotImplementedError(
        "Stage-1 training is not ported yet. "
        "This directory currently encodes the paper-aligned NACA0012 parameters "
        "and the no-BC synthetic-data setup only. "
        f"Synthetic shape: {case_params['synthetic_data']['Y']} x {case_params['synthetic_data']['X']}."
    )
