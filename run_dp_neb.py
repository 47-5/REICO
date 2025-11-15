from ase.io import read, write
from ase.optimize import BFGS
from ase.mep import NEB
from ase.constraints import FixAtoms
from deepmd.calculator import DP
import os
from typing import List


def run_neb(image_path_list: List[str], image_number=7, constraint=[],
            out_path='neb_result.xyz',
            dp_model_path='CH_model_01.pth',
            max_step=0.05, f_max=0.05, steps=1000):
    assert len(image_path_list) == 2 or len(image_path_list) == image_number

    if len(image_path_list) == 2:
        IS = read(image_path_list[0])
        FS = read(image_path_list[-1])
        add_number = image_number - len(image_path_list)
        images = [IS]
        images += [IS.copy() for _ in range(add_number)]
        images.append(FS)
    elif len(image_path_list) == image_number:
        images = [read(i) for i in image_path_list]
    else:
        raise ValueError('image_path_list must have 2 or image_number elements')

    # add calculator
    for image in images:
        image.calc = DP(model=dp_model_path)

    # make NEB object
    neb = NEB(images, k=0.1, climb=True)
    neb.interpolate(mic=True)

    energies = []
    for mol in images:
        mol.calc = DP(model=dp_model_path)
        energy = mol.get_potential_energy()
        forces = mol.get_forces()
        energies.append(energy)
        write(filename='neb_init.xyz', images=mol, format='extxyz', append=True)
    print(energies)

    # apply constraint
    # if constraint is not None:
    for image in images:
        image.set_constraint(constraint)

    # opt
    optimizer = BFGS(neb, trajectory='neb.traj', maxstep=max_step)
    optimizer.run(fmax=f_max, steps=steps)
    traj = read('neb.traj', index=f'{-image_number}:')
    energies = []
    for mol in traj:
        mol.calc = DP(model=dp_model_path)
        energy = mol.get_potential_energy()
        forces = mol.get_forces()
        energies.append(energy)
        write(filename=out_path, images=mol, format='extxyz', append=True)

    print(energies)
    return traj


if __name__ == '__main__':

    image_path_list = ['IS_opt.xyz', 'FS_opt.xyz',]
    run_neb(image_path_list, image_number=7, dp_model_path='CH_model_01.pth', f_max=3)
