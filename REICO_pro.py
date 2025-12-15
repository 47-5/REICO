#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REICO-style random structure generator with job categories (bulk/slab/cluster/adsorption)

Enhancements:
1) General (non-orthorhombic) cell generation with configurable angle ranges in JSON.
2) Two position sampling modes:
   - "resample": sample all positions at once and accept if constraints satisfied (original).
   - "incremental": place atoms one-by-one with local distance checks (faster, fewer NxN distance matrices).
3) position_stats can be enabled/disabled, and max_nn check can be toggled to reduce rejection cost.

Output scheme (Scheme B, clarified):
- Per-structure files:
    controlled by JSON key: "format"  (e.g. "vasp", "xyz", "extxyz", ...)
    written under: outdir/<job_name>/structure_000001.<format>
- Merged files:
    ALWAYS written in extxyz format (content is extxyz regardless of filename)
    - Per job merged: outdir/<job_name>.extxyz
    - Global merged:  outdir/<merged>   (recommended name endswith .extxyz)
"""

import json
import sys
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.io import write

from joblib import Parallel, delayed
from tqdm import tqdm


# ----------------------------
# IO / utilities
# ----------------------------
def load_json_strict(path: str) -> dict:
    p = Path(path)
    if p.suffix.lower() != ".json":
        raise TypeError(f"Config file must be .json, got: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def pick_int_in_range(rng):
    """rng: [min, max] inclusive 从给定的范围内采样一个整数"""
    lo, hi = int(rng[0]), int(rng[1])
    if hi < lo:
        raise ValueError(f"Invalid range: {rng}")
    return random.randint(lo, hi)


def composition_from_ranges(elem_ranges: dict) -> tuple[list[str], dict]:
    """
    elem_ranges: {"Pt":[1,12], "O":[1,12]}
    returns:
      symbols: ['Pt','Pt',...,'O',...]
      counts: {'Pt':n_pt,'O':n_o}
    """
    counts = {}
    symbols = []
    for el, rng in elem_ranges.items():
        n = pick_int_in_range(rng)
        if n > 0:
            counts[el] = n
            symbols.extend([el] * n)
    if len(symbols) == 0:
        raise ValueError("Generated empty composition (all zeros). Please adjust ranges.")
    return symbols, counts


# ----------------------------
# Geometry / constraints
# ----------------------------
def min_dist_matrix(symbols: list[str], scale: float) -> np.ndarray:
    """获得原子最小间距矩阵"""
    radii = [covalent_radii[atomic_numbers[s]] for s in symbols]
    n = len(symbols)
    m = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = scale * (radii[i] + radii[j])
    return m


def estimate_volume_from_covalent_spheres(symbols: list[str], packing: float = 1.0) -> float:
    """
    Sum of (4/3 pi r^3) per atom * packing factor.
    packing>1 yields larger cells, fewer rejections.
    把原子当作半径为共价半径的球，估计总体积，从而估计晶胞体积
    """
    vol = 0.0
    for s in symbols:
        r = covalent_radii[atomic_numbers[s]]
        vol += 4.0 / 3.0 * math.pi * (r ** 3)
    return packing * vol


def cell_matrix_from_lengths_angles(a: float, b: float, c: float,
                                   alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """
    Build a general 3x3 cell matrix from lengths and angles (ASE-style convention).
    alpha: angle between b and c
    beta:  angle between a and c
    gamma: angle between a and b
    """
    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)

    va = np.array([a, 0.0, 0.0], dtype=float)
    vb = np.array([b * math.cos(gamma), b * math.sin(gamma), 0.0], dtype=float)

    cx = c * math.cos(beta)
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-8:
        return None

    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 1e-12:
        return None
    vc = np.array([cx, cy, math.sqrt(cz2)], dtype=float)

    return np.vstack([va, vb, vc])


def random_general_cell_from_volume(volume: float,
                                    angle_ranges: dict | None,
                                    length_ratio_range: tuple[float, float] | None,
                                    volume_factor_range: tuple[float, float] = (1.2, 5.0),
                                    max_tries: int = 5000) -> np.ndarray:
    """
    Generate a general (possibly non-orthogonal) cell matrix with volume in [fmin*V, fmax*V].
    """
    volume = max(float(volume), 1e-6)
    fmin, fmax = float(volume_factor_range[0]), float(volume_factor_range[1])
    if fmax < fmin:
        raise ValueError("volume_factor_range must satisfy fmax>=fmin")

    if angle_ranges is None:
        angle_ranges = {"alpha": (90.0, 90.0), "beta": (90.0, 90.0), "gamma": (90.0, 90.0)}
    else:
        angle_ranges = {
            "alpha": (float(angle_ranges["alpha"][0]), float(angle_ranges["alpha"][1])),
            "beta": (float(angle_ranges["beta"][0]), float(angle_ranges["beta"][1])),
            "gamma": (float(angle_ranges["gamma"][0]), float(angle_ranges["gamma"][1])),
        }

    if length_ratio_range is None:
        rmin, rmax = 1.0, 1.0
    else:
        rmin, rmax = float(length_ratio_range[0]), float(length_ratio_range[1])
        if rmin <= 0 or rmax <= 0 or rmax < rmin:
            raise ValueError("cell_length_ratio_range must be positive and satisfy max>=min")

    for _ in range(max_tries):
        alpha = random.uniform(*angle_ranges["alpha"])
        beta = random.uniform(*angle_ranges["beta"])
        gamma = random.uniform(*angle_ranges["gamma"])

        rb = random.uniform(rmin, rmax)
        rc = random.uniform(rmin, rmax)

        target_vol = random.uniform(fmin * volume, fmax * volume)

        cell0 = cell_matrix_from_lengths_angles(1.0, rb, rc, alpha, beta, gamma)
        if cell0 is None:
            continue

        vol0 = abs(np.linalg.det(cell0))
        if vol0 <= 1e-12:
            continue

        scale = (target_vol / vol0) ** (1.0 / 3.0)
        cell = cell0 * scale

        lengths = np.linalg.norm(cell, axis=1)
        if np.min(lengths) < 1e-3:
            continue

        return cell

    raise RuntimeError("Failed to generate a valid cell within constraints. "
                       "Try widening angle ranges or length ratios, or increasing max tries.")


def enforce_min_dist_by_resampling(atoms: Atoms, min_dists: np.ndarray, max_attempts: int) -> bool:
    """Sample scaled positions in [0,1)^3 until all pair distances >= min_dists. Uses MIC."""
    n = len(atoms)
    for _ in range(max_attempts):
        pos = np.random.rand(n, 3)
        atoms.set_scaled_positions(pos)
        d = atoms.get_all_distances(mic=True)
        ok = np.all((d - min_dists) * (1 - np.eye(n)) >= 0)
        if ok:
            return True
    return False


def place_atoms_incremental_general(atoms: Atoms, min_dists: np.ndarray, max_trials_per_atom: int) -> bool:
    """
    Incrementally place atoms in fractional coordinates for a general cell.
    MIC in fractional space: ds -= round(ds), dr = ds @ cell.
    """
    n = len(atoms)
    cell = np.array(atoms.get_cell())  # (3,3)
    scaled = np.empty((n, 3), dtype=float)

    for i in range(n):
        for _ in range(max_trials_per_atom):
            si = np.random.rand(3)
            if i == 0:
                scaled[i] = si
                break

            ds = si - scaled[:i]
            ds -= np.round(ds)
            dr = ds @ cell
            dist2 = np.einsum("ij,ij->i", dr, dr)

            thr = min_dists[i, :i]
            if np.all(dist2 >= thr * thr):
                scaled[i] = si
                break
        else:
            return False

    atoms.set_scaled_positions(scaled)
    return True


def position_stats(atoms: Atoms) -> tuple[float, float]:
    """Return (min nearest-neighbor distance, max nearest-neighbor distance)."""
    d = atoms.get_all_distances(mic=True)
    nn = np.partition(d, 1, axis=1)[:, 1]
    return float(np.min(nn)), float(np.max(nn))


# ----------------------------
# Parameters
# ----------------------------
@dataclass
class GenParams:
    min_dist_scale: float = 0.8
    max_attempts_pos: int = 4000
    max_attempts_struct: int = 2000
    volume_packing: float = 1.2
    tolerance_min_nn: float = 1.6
    tolerance_max_nn: float = 3.0

    sampling_mode: str = "resample"  # "resample" | "incremental"
    cell_angle_range: dict | None = None
    cell_length_ratio_range: tuple[float, float] | None = None

    enable_position_stats: bool = False
    check_max_nn: bool = False


# ----------------------------
# Structure generators
# ----------------------------
def gen_mix_structure(elem_ranges: dict, vacuum: float, params: GenParams) -> Atoms | None:
    """General structure (bulk/slab) with PBC True. If vacuum>0: center along z."""
    for _ in range(params.max_attempts_struct):
        symbols, _ = composition_from_ranges(elem_ranges)
        vol = estimate_volume_from_covalent_spheres(symbols, packing=params.volume_packing)

        cell = random_general_cell_from_volume(
            vol,
            angle_ranges=params.cell_angle_range,
            length_ratio_range=params.cell_length_ratio_range,
            volume_factor_range=(1.2, 5.0),
            max_tries=5000,
        )

        atoms = Atoms(symbols=symbols)
        atoms.set_cell(cell, scale_atoms=False)
        atoms.pbc = True

        min_d = min_dist_matrix(symbols, params.min_dist_scale)

        mode = params.sampling_mode.lower()
        if mode == "resample":
            ok = enforce_min_dist_by_resampling(atoms, min_d, params.max_attempts_pos)
        elif mode == "incremental":
            ok = place_atoms_incremental_general(atoms, min_d, params.max_attempts_pos)
        else:
            raise ValueError(f"Unknown sampling_mode: {params.sampling_mode} (use 'resample' or 'incremental')")

        if not ok:
            continue

        if params.enable_position_stats:
            min_nn, max_nn = position_stats(atoms)
            if min_nn <= params.tolerance_min_nn:
                continue
            if params.check_max_nn and (max_nn >= params.tolerance_max_nn):
                continue

        if vacuum and vacuum > 0:
            atoms.center(vacuum=vacuum, axis=2)

        return atoms
    return None


def gen_cluster_structure(elem_ranges: dict, vacuum: float, params: GenParams) -> Atoms | None:
    atoms = gen_mix_structure(elem_ranges, vacuum=0.0, params=params)
    if atoms is None:
        return None

    cellpar = atoms.cell.cellpar()
    a, b, c = float(cellpar[0]), float(cellpar[1]), float(cellpar[2])
    atoms.set_cell([a, b, c])
    atoms.pbc = True

    vac = vacuum if vacuum is not None else 8.0
    atoms.center(vacuum=vac, axis=[0, 1, 2])
    return atoms


def gen_adsorption_structure(ad_conf: dict, vacuum: float, params: GenParams) -> Atoms | None:
    base_syms, base_counts = composition_from_ranges(ad_conf["base"])
    ad_syms, ad_counts = composition_from_ranges(ad_conf["adsorption"])
    base_n = len(base_syms)
    ad_n = len(ad_syms)

    fixed_ranges = {}
    for el, n in base_counts.items():
        fixed_ranges[el] = [n, n]
    for el, n in ad_counts.items():
        fixed_ranges[el] = [n, n]

    atoms = gen_mix_structure(fixed_ranges, vacuum=vacuum, params=params)
    if atoms is None:
        return None

    pos_sorted = np.array(sorted(atoms.positions, key=lambda p: p[2]))
    base_pos = pos_sorted[:base_n].copy()
    ad_pos = pos_sorted[base_n:base_n + ad_n].copy()

    np.random.shuffle(base_pos)
    np.random.shuffle(ad_pos)
    atoms.set_positions(np.vstack([base_pos, ad_pos]))
    return atoms


# ----------------------------
# Joblib worker
# ----------------------------
def _one_structure(kind: str, payload: dict, vacuum: float, params_dict: dict) -> Atoms | None:
    params = GenParams(**params_dict)

    if kind == "mix":
        return gen_mix_structure(payload, vacuum=vacuum, params=params)
    if kind == "cluster":
        return gen_cluster_structure(payload, vacuum=vacuum, params=params)
    if kind == "ad":
        return gen_adsorption_structure(payload, vacuum=vacuum, params=params)
    raise ValueError(f"Unknown kind: {kind}")


# ----------------------------
# Main
# ----------------------------
def main(config_path: str | None = None):
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) >= 2 else "input.json"

    jdata = load_json_strict(config_path)

    nproc = int(jdata.get("nproc", 8))
    outdir = Path(jdata.get("outdir", "reico_out"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Per-structure file format (user-controlled)
    per_structure_format = str(jdata.get("format", "extxyz"))

    # Merged output (always extxyz)
    merged_name = str(jdata.get("merged", jdata.get("output", "all.extxyz")))
    merged_format = str(jdata.get("merged_format", "extxyz")).lower()
    if merged_format != "extxyz":
        raise ValueError("merged_format is fixed to 'extxyz' in this output scheme.")

    seed = jdata.get("seed", None)
    set_global_seed(seed)

    sampling_mode = str(jdata.get("sampling_mode", "resample"))
    enable_position_stats = bool(jdata.get("enable_position_stats", True))
    check_max_nn = bool(jdata.get("check_max_nn", True))

    cell_angle_range = jdata.get("cell_angle_range", None)
    if cell_angle_range is not None:
        for key in ("alpha", "beta", "gamma"):
            if key not in cell_angle_range:
                raise ValueError("cell_angle_range must contain alpha,beta,gamma ranges")

    cell_length_ratio_range = jdata.get("cell_length_ratio_range", None)
    if cell_length_ratio_range is not None:
        cell_length_ratio_range = (float(cell_length_ratio_range[0]), float(cell_length_ratio_range[1]))

    params_dict = {
        "min_dist_scale": float(jdata.get("min_dist_scale", 0.8)),
        "max_attempts_pos": int(jdata.get("max_attempts_pos", 4000)),
        "max_attempts_struct": int(jdata.get("max_attempts_struct", 2000)),
        "volume_packing": float(jdata.get("volume_packing", 1.2)),
        "tolerance_min_nn": float(jdata.get("tolerance_min_nn", 1.6)),
        "tolerance_max_nn": float(jdata.get("tolerance_max_nn", 3.0)),
        "sampling_mode": sampling_mode,
        "cell_angle_range": cell_angle_range,
        "cell_length_ratio_range": cell_length_ratio_range,
        "enable_position_stats": enable_position_stats,
        "check_max_nn": check_max_nn,
    }

    global_keys = {
        "nproc", "outdir", "format", "seed",
        "merged", "merged_format", "output",  # output kept for backward compatibility
        "min_dist_scale", "max_attempts_pos", "max_attempts_struct",
        "volume_packing", "tolerance_min_nn", "tolerance_max_nn",
        "sampling_mode", "cell_angle_range", "cell_length_ratio_range",
        "enable_position_stats", "check_max_nn",
    }

    # Parse jobs in JSON order
    jobs = []
    for job_name, conf in jdata.items():
        if job_name in global_keys:
            continue
        if not isinstance(conf, dict):
            continue

        vacuum = float(conf.get("vacuum", 0.0))
        numbers = int(conf.get("numbers", 0))
        if numbers <= 0:
            continue

        if "mix" in conf:
            kind = "mix"
            payload = conf["mix"]
        elif "cluster" in conf:
            kind = "cluster"
            payload = conf["cluster"]
        elif "ad" in conf:
            kind = "ad"
            payload = conf["ad"]
        else:
            raise ValueError(f"Job {job_name} must contain one of: mix/cluster/ad")

        jobs.append({
            "name": job_name,
            "kind": kind,
            "payload": payload,
            "vacuum": vacuum,
            "numbers": numbers,
        })

    parallel_kwargs = dict(n_jobs=nproc, backend="loky", verbose=0)

    merged_all: list[Atoms] = []
    meta_jobs = []

    for job in jobs:
        job_name = job["name"]
        kind = job["kind"]
        payload = job["payload"]
        vacuum = job["vacuum"]
        numbers = job["numbers"]

        job_dir = outdir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        desc = f"{job_name} (n={numbers}, n_jobs={nproc})"
        tasks = (delayed(_one_structure)(kind, payload, vacuum, params_dict) for _ in range(numbers))
        res = Parallel(**parallel_kwargs)(tasks)

        atoms_list = []
        for a in tqdm(res, total=numbers, desc=desc, unit="struct"):
            if a is not None:
                atoms_list.append(a)

        # Per-structure files under job directory (format controlled by per_structure_format)
        for i, atoms in enumerate(atoms_list, start=1):
            atoms.info["job"] = job_name
            atoms.info["job_index"] = i
            write(job_dir / f"structure_{i:06d}.{per_structure_format}", atoms, format=per_structure_format)

        # Per-job merged file (ALWAYS extxyz)
        job_merged_file = outdir / f"{job_name}.extxyz"
        if atoms_list:
            for i, atoms in enumerate(atoms_list, start=1):
                atoms.info["job"] = job_name
                atoms.info["job_index"] = i
            write(job_merged_file, atoms_list, format="extxyz")

        merged_all.extend(atoms_list)

        meta_jobs.append({
            "name": job_name,
            "kind": kind,
            "vacuum": vacuum,
            "numbers_requested": numbers,
            "numbers_generated": len(atoms_list),
            "dir": str(job_dir),
            "merged_file": str(job_merged_file.name),
        })

    # Global merged file (ALWAYS extxyz), concatenated in job order
    merged_path = outdir / merged_name
    if merged_all:
        for gid, atoms in enumerate(merged_all, start=1):
            atoms.info["global_id"] = gid
        write(merged_path, merged_all, format="extxyz")

    meta = {
        "n_generated": len(merged_all),
        "n_requested": int(sum(j["numbers"] for j in jobs)),
        "nproc": nproc,
        "jobs": meta_jobs,
        "params": params_dict,
        "per_structure_format": per_structure_format,
        "outdir": str(outdir),
        "merged_file": str(merged_path.name),
        "merged_format": "extxyz",
        "parallel": {"library": "joblib", "backend": "loky"},
        "progress": {"library": "tqdm"},
        "output_scheme": "B: per-job dirs (format=per_structure_format) + merged extxyz",
    }
    with (outdir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. Generated {len(merged_all)}/{meta['n_requested']} structures in {outdir}/")


if __name__ == "__main__":
    main()