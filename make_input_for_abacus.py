#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert structures readable by ASE (cif/POSCAR/extxyz/xyz/...) to ABACUS STRU.

Features
- Reads single structure or trajectory (extxyz, xyz, etc.)
- Writes one STRU or multiple STRU files (STRU, STRU_000001, ...)
- Writes ATOMIC_SPECIES and NUMERICAL_ORBITAL from user-provided mapping
- Supports Direct (fractional) or Cartesian coordinates
- Supports periodic cells; warns if non-periodic

Usage examples
--------------
# single file -> STRU
python ase2stru.py input.cif -o STRU

# POSCAR -> STRU with fractional coordinates
python ase2stru.py POSCAR --coord Direct -o STRU

# extxyz trajectory -> STRU_000001, STRU_000002, ...
python ase2stru.py traj.extxyz --multi --out-prefix STRU_

# provide pseudo/orbital mapping via json
python ase2stru.py input.cif --map mapping.json -o STRU
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from glob import glob
import shutil

import numpy as np
from ase import Atoms, atoms
from ase.io import read

BOHR_PER_ANG = 1.8897259886  # ABACUS example uses LATTICE_CONSTANT = 1.8897259886 (Bohr), i.e. 1 Å

DEFAULT_MAP = {
    # Example placeholders; replace with your official library filenames
    # "ElementSymbol": {"mass": 0.0, "pseudo": "X.upf", "orb": "X.orb"}
    "H": {"mass": 1.008, "pseudo": "H_ONCV_PBE-1.0.upf", "orb": "H_gga_6au_100Ry_2s1p.orb"},
    "C": {"mass": 12.011, "pseudo": "C_ONCV_PBE-1.0.upf", "orb": "C_gga_7au_100Ry_2s2p1d.orb"},
    "O": {"mass": 15.999, "pseudo": "O_ONCV_PBE-1.0.upf", "orb": "O_gga_7au_100Ry_2s2p1d.orb"},
    "Si": {"mass": 28.085, "pseudo": "Si_ONCV_PBE-1.0.upf", "orb": "Si_gga_7au_100Ry_2s2p1d.orb"},
    "Al": {"mass": 26.982, "pseudo": "Al_ONCV_PBE-1.0.upf", "orb": "Al_gga_7au_100Ry_4s4p1d.orb"},
    "B": {"mass": 10.81, "pseudo": "B_ONCV_PBE-1.0.upf", "orb": "B_gga_8au_100Ry_2s2p1d.orb"},
    "Ga": {"mass": 69.723, "pseudo": "Ga_ONCV_PBE-1.0.upf", "orb": "Ga_gga_8au_100Ry_2s2p2d1f.orb"},
    "Mg": {"mass": 24.305, "pseudo": "Mg_ONCV_PBE-1.0.upf", "orb": "Mg_gga_8au_100Ry_4s2p1d.orb"},
}


def fmt_f(x: float) -> str:
    return f"{x:.10f}".rstrip("0").rstrip(".") if abs(x) < 1e10 else f"{x:.10e}"


def load_mapping(path: str | None) -> dict:
    if path is None:
        return DEFAULT_MAP
    with open(path, "r", encoding="utf-8") as f:
        mp = json.load(f)
    return mp


def ensure_periodic(atoms: Atoms):
    pbc = atoms.pbc
    if not np.all(pbc):
        # ABACUS STRU can still be written, but typical DFT assumes PBC.
        # We'll warn; user can proceed.
        print(f"[WARN] atoms.pbc={pbc}. STRU will be written, but check if this is intended.")


def atoms_to_stru(
        atoms: Atoms,
        mapping: dict,
        coord: str = "Direct",
        lattice_constant_bohr: float = BOHR_PER_ANG,
        move_flags=(1, 1, 1),
        magmom_default: float = 0.0,
) -> str:
    """
    Build ABACUS STRU text from ASE Atoms.
    """
    ensure_periodic(atoms)

    symbols = atoms.get_chemical_symbols()
    unique = []
    for s in symbols:
        if s not in unique:
            unique.append(s)

    # Validate mapping
    missing = [s for s in unique if s not in mapping]
    if missing:
        raise KeyError(
            f"Missing mapping for elements: {missing}\n"
            f"Provide --map mapping.json with entries like "
            f'{{"Si": {{"mass": 28.085, "pseudo": "Si.upf", "orb": "Si.orb"}}}}'
        )

    cell_ang = atoms.cell.array  # Angstrom
    # ABACUS: LATTICE_CONSTANT in Bohr. LATTICE_VECTORS are in units of (1/LATTICE_CONSTANT?) in their examples:
    # In your example: LATTICE_CONSTANT=1.8897 Bohr (=1 Å), and vectors are in Å numerically.
    # So we will write:
    # LATTICE_CONSTANT = 1.8897259886
    # LATTICE_VECTORS = cell in Angstrom
    # This matches the example convention.
    lines = []
    lines.append("#This is the atom file containing all the information")
    lines.append("#about the lattice structure.")
    lines.append("")
    lines.append("ATOMIC_SPECIES")
    for s in unique:
        mass = mapping[s]["mass"]
        pseudo = mapping[s]["pseudo"]
        lines.append(f"{s:<2} {fmt_f(mass):>8}  {pseudo}")
    lines.append("")
    lines.append("NUMERICAL_ORBITAL")
    for s in unique:
        orb = mapping[s]["orb"]
        lines.append(f"{orb}")
    lines.append("")
    lines.append("LATTICE_CONSTANT")
    lines.append(f"{fmt_f(lattice_constant_bohr)}\t\t# {fmt_f(BOHR_PER_ANG)} Bohr =  1.0 Angstrom")
    lines.append("")
    lines.append("LATTICE_VECTORS")
    for v in cell_ang:
        lines.append(f"{fmt_f(v[0])} {fmt_f(v[1])} {fmt_f(v[2])}")
    lines.append("")
    lines.append("ATOMIC_POSITIONS")
    coord = coord.strip()
    if coord.lower() in ["direct", "fractional", "frac"]:
        coord_tag = "Direct"
        # fractional
        scaled = atoms.get_scaled_positions(wrap=False)
        pos_list = scaled
        coord_comment = "#Fractional"
    elif coord.lower() in ["cart", "cartesian"]:
        coord_tag = "Cartesian"
        pos_list = atoms.get_positions()
        coord_comment = "#Cartesian(Unit is LATTICE_CONSTANT)"
        # In ABACUS, Cartesian coords are in units of LATTICE_CONSTANT (Bohr).
        # With lattice_constant = 1 Å, we can write Å numerically.
        # If user sets a different lattice_constant, they'd need conversion.
        if abs(lattice_constant_bohr - BOHR_PER_ANG) > 1e-8:
            # Convert Angstrom -> (Bohr units of lattice constant)
            # x_out * (lattice_constant in Bohr) = x_in in Bohr => x_out = x_bohr / lattice_constant_bohr
            pos_bohr = pos_list * BOHR_PER_ANG
            pos_list = pos_bohr / lattice_constant_bohr
    else:
        raise ValueError("coord must be Direct or Cartesian")

    lines.append(f"{coord_tag: <24}{coord_comment}")
    lines.append("")

    # Group atoms by element, preserving first-appearance order (like example)
    idx_by_elem = defaultdict(list)
    for i, s in enumerate(symbols):
        idx_by_elem[s].append(i)

    for s in unique:
        lines.append(f"{s}")
        # Magnetic line for this element.
        # We'll write 0.0 by default; if Atoms has initial magmoms, you can extend this.
        lines.append(f"{fmt_f(magmom_default)}")
        lines.append(f"{len(idx_by_elem[s])}")

        for i in idx_by_elem[s]:
            x, y, z = pos_list[i]
            mx, my, mz = move_flags
            lines.append(f"{fmt_f(x)}  {fmt_f(y)}  {fmt_f(z)}  {mx} {my} {mz}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def make_INPUT(template_input_path='INPUT', output_root='.'):
    os.makedirs(output_root, exist_ok=True)
    shutil.copy(template_input_path, output_root)
    return None


def make_pseudo(atoms, output_root='.', mapping=DEFAULT_MAP, element_pseudo_root_path='SG15_ONCV_v1.0_upf'):
    os.makedirs(output_root, exist_ok=True)

    elements = list(set(atoms.get_chemical_symbols()))
    print(elements)
    for e in elements:
        shutil.copy(os.path.join(element_pseudo_root_path, mapping[e]['pseudo']), output_root)
    return None


def make_orb(atoms, output_root='.', mapping=DEFAULT_MAP, orb_root_path='SG15-Version1p0__StandardOrbitals-Version2p0'):
    os.makedirs(output_root, exist_ok=True)

    elements = list(set(atoms.get_chemical_symbols()))
    for e in elements:
        shutil.copy(os.path.join(orb_root_path, mapping[e]['orb']), output_root)
    return None


def make_abacus_input(
        input_file_path, input_file_format=None, output_root='.',
        mapping: dict = DEFAULT_MAP,
        coord: str = "Direct",
        lattice_constant_bohr: float = BOHR_PER_ANG,
        move_flags=(1, 1, 1),
        magmom_default: float = 0.0,
        element_pseudo_root_path='SG15_ONCV_v1.0_upf',
        orb_root_path='SG15-Version1p0__StandardOrbitals-Version2p0',
        template_input_path='INPUT',
        lcao=True,
):

    os.makedirs(output_root, exist_ok=True)
    atoms = read(input_file_path, format=input_file_format)

    make_INPUT(template_input_path=template_input_path, output_root=output_root)
    make_pseudo(atoms=atoms, output_root=output_root, mapping=mapping, element_pseudo_root_path=element_pseudo_root_path)
    if lcao:
        make_orb(atoms=atoms, output_root=output_root, mapping=mapping, orb_root_path=orb_root_path)
    x = atoms_to_stru(atoms=atoms, mapping=mapping, coord=coord, lattice_constant_bohr=lattice_constant_bohr, move_flags=move_flags,
                  magmom_default=magmom_default)
    with open(os.path.join(output_root, "STRU"), "w", encoding="utf-8") as f:
        f.write(x)



if __name__ == "__main__":
    run_mode = 'batch'

    if run_mode == 'single':
        make_abacus_input(input_file_path='test/Ga-1.cif', input_file_format='cif', output_root='abacus_test',
                          mapping=DEFAULT_MAP,
                          coord='Direct',
                          lattice_constant_bohr=BOHR_PER_ANG,
                          move_flags=(1, 1, 1),
                          magmom_default=0.0,
                          element_pseudo_root_path='SG15_ONCV_v1.0_upf', template_input_path='INPUT')

    else:
        batch_root = 'batch_abacus'

        input_file_path_list = glob('test/*.cif')
        input_file_path_list.sort()
        for i in input_file_path_list:
            output_root = os.path.join(batch_root, os.path.basename(i).split('.')[0])
            make_abacus_input(
                input_file_path=i, output_root=output_root,
                mapping=DEFAULT_MAP,
                coord='Direct',
                lattice_constant_bohr=BOHR_PER_ANG,
                move_flags=(1, 1, 1),
                magmom_default=0.0,
                element_pseudo_root_path='SG15_ONCV_v1.0_upf', template_input_path='INPUT'
                            )



