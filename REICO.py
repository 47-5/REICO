#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random structure generator following REICO-style sampling
Reference: Yang et al., Nat. Catal. 8, 891–904 (2025)
"""
import json
import random
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, atomic_numbers
from ase.io import write

# ---------------------- 配置区 ----------------------
ELEMENT_POOL = ["Si", "Al", "C", "H", "O", 'Ga']
TOTAL_STRUCTURES = 500          # 需要生成的结构数量
N_ATOMS_RANGE = (5, 30)         # 每个结构的原子数范围
LATTICE_A_RANGE = (3.0, 12.0)   # 晶胞 a 轴 (Å)
LATTICE_B_RANGE = (3.0, 12.0)
LATTICE_C_RANGE = (3.0, 12.0)
ANGLE_RANGE = (60, 120)         # 晶胞角度 (°)
MIN_DIST_SCALE = 0.8            # 最小距离 = scale * (r_cov_i + r_cov_j)
OUTPUT_DIR = Path("reico_random_structs")
METADATA_FILE = OUTPUT_DIR / "metadata.json"
SEED = 2025                     # 固定随机种子便于复现
MAX_ATTEMPTS = 10000          # 最多尝试次数，防止无限循环
START_INDEX = 1              # 产生的结构的初始序号
# ----------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def random_cell():
    lengths = [
        random.uniform(*LATTICE_A_RANGE),
        random.uniform(*LATTICE_B_RANGE),
        random.uniform(*LATTICE_C_RANGE),
    ]
    angles = [
        random.uniform(*ANGLE_RANGE),
        random.uniform(*ANGLE_RANGE),
        random.uniform(*ANGLE_RANGE),
    ]
    return lengths, angles

def random_composition(n_atoms):
    """随机分配元素，保证至少含有两种元素"""
    elems = [random.choice(ELEMENT_POOL) for _ in range(n_atoms)]
    if len(set(elems)) == 1:
        other = random.choice([e for e in ELEMENT_POOL if e != elems[0]])
        idx = random.randrange(n_atoms)
        elems[idx] = other
    return elems

def min_distance_matrix(symbols):
    radii = [covalent_radii[atomic_numbers[s]] for s in symbols]
    n = len(symbols)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = MIN_DIST_SCALE * (radii[i] + radii[j])
    return matrix

def random_positions(n_atoms):
    return np.random.rand(n_atoms, 3)

def enforce_min_dist(atoms, min_dists, max_attempts=4000):
    """重复采样直至满足最小距离限制"""
    n = len(atoms)
    attempts = 0
    while attempts < max_attempts:
        pos = random_positions(n)
        atoms.set_scaled_positions(pos)
        dists = atoms.get_all_distances(mic=True)
        if np.all((dists - min_dists) * (1 - np.eye(n)) >= 0):
            return True
        attempts += 1
    return False

def main():
    meta = []
    generated = 0
    attempts = 0

    while generated < TOTAL_STRUCTURES and attempts < MAX_ATTEMPTS:
        if attempts % 100 == 0:
            print(f'attempts: {attempts} of {MAX_ATTEMPTS}')
        attempts += 1
        n_atoms = random.randint(*N_ATOMS_RANGE)
        symbols = random_composition(n_atoms)
        lengths, angles = random_cell()
        atoms = Atoms(symbols=symbols)
        atoms.set_cell(lengths + angles, scale_atoms=False)
        min_dists = min_distance_matrix(symbols)

        success = enforce_min_dist(atoms, min_dists)
        if not success:
            continue  # 不占用编号，继续尝试

        scale = random.uniform(0.85, 1.15)
        atoms.set_cell(atoms.cell * scale, scale_atoms=True)

        fname = OUTPUT_DIR / f"structure_{generated + START_INDEX:06d}.cif"
        write(fname, atoms, format="cif")

        meta.append(
            {
                "id": generated,
                "n_atoms": n_atoms,
                "elements": symbols,
                "cell_lengths": lengths,
                "cell_angles": angles,
                "scale_factor": scale,
                "file": fname.name,
            }
        )
        generated += 1

    if generated < TOTAL_STRUCTURES:
        print(f"[WARN] 仅生成 {generated} 个结构，已达到最大尝试次数 {MAX_ATTEMPTS}。")
    else:
        print(f"完成，生成 {generated} 个结构，结果位于 {OUTPUT_DIR}/")

    with METADATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()