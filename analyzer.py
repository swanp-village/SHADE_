import argparse
from importlib import import_module

from config.model import SimulationConfig
from MRR.analyzer import analyze

# 10_DE4 8_DE29 6_DE7 4_DE18


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--number-of-samples", "-n", type=int)
    parser.add_argument("-L", type=float)
    parser.add_argument("-K", type=float)
    parser.add_argument("--entropy", "-e", action="store_const", const=5)
    args = parser.parse_args()
    config_name: str = args.config
    n = args.number_of_samples
    L_error_rate = args.L
    K_error_rate = args.K
    entropy = args.entropy
    if config_name[-3:] == ".py":
        config_name = config_name[:-3]
    try:
        imported_module = import_module(f"config.simulate.{config_name}")
        imported_config = getattr(imported_module, "config")
        simulation_config = SimulationConfig(**imported_config, entropy=entropy)
        result = analyze(
            n,
            L_error_rate=L_error_rate,
            K_error_rate=K_error_rate,
            L=simulation_config.L,
            K=simulation_config.K,
            n_g=simulation_config.n_g,
            n_eff=simulation_config.n_eff,
            eta=simulation_config.eta,
            alpha=simulation_config.alpha,
            center_wavelength=simulation_config.center_wavelength,
            length_of_3db_band=simulation_config.length_of_3db_band,
            FSR=simulation_config.FSR,
            max_crosstalk=simulation_config.max_crosstalk,
            H_p=simulation_config.H_p,
            H_s=simulation_config.H_s,
            H_i=simulation_config.H_i,
            r_max=simulation_config.r_max,
            weight=simulation_config.weight,
            min_ring_length=simulation_config.min_ring_length,
            format=simulation_config.format,
            lambda_limit=simulation_config.lambda_limit,
            name=simulation_config.name,
            label=simulation_config.label,
            seedsequence=simulation_config.seedsequence,
        )
        print(result)
    except ModuleNotFoundError as e:
        print(e)
