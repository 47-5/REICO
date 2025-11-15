from ase.io import read, write, Trajectory
from ase.optimize import BFGS
from ase.constraints import FixAtoms
import ase
import deepmd
from deepmd.calculator import DP
import os


__all__ = ['opt_a_mol']


def opt_a_mol(read_file_path, read_file_format=None, write_file_path='opted.xyz', write_file_format='xyz',
              dp_model_path=None, head=None, only_relax_H=False, maxstep=0.05, f_max=0.05,
              traj_path=None, traj_format='proteindatabank', traj_interval=1):
    """
    Optimizes a molecular structure using a deep potential model and writes the optimized structure to a file.

    Summary:
    This function reads a molecular structure from a specified file, optimizes it using a deep potential model,
    and then writes the optimized structure to another file. The optimization can be restricted to only hydrogen atoms
    if specified. Optionally, it can record the optimization trajectory in an ASE-compatible format.

    Parameters:
    - read_file_path (str): Path to the file containing the initial molecular structure.
    - read_file_format (Optional[str]): Format of the input file. If not provided, ASE infers it from the file extension.
    - write_file_path (str): Path to the file where the optimized structure will be written. Default is 'opted.xyz'.
    - write_file_format (str): Format of the output file. Default is 'xyz'.
    - dp_model_path (Optional[str]): Path to the deep potential model file. Defaults to 'DPA2_medium_28_10M_rc0.pt'.
    - only_relax_H (bool): If True, only hydrogen atoms are allowed to move during the optimization. Default is False.
    - traj_path (Optional[str]): Path to the trajectory file to create. If None, no trajectory is written.
    - traj_format (str): Format to use when writing the trajectory (e.g., 'xyz', 'pdb'). Default is 'xyz'.
    - traj_interval (int): Write every Nth step to the trajectory file. Default is 1 (every step).
    - maxstep (float): Maximum step size for the optimizer. Default is 0.05.
    - f_max (float): Force convergence criterion for the optimizer. Default is 0.05.

    Returns:
    - ase.Atoms: The optimized molecular structure as an ASE Atoms object.
    """

    if dp_model_path is None:
        dp_model_path = os.path.join('DPA2_medium_28_10M_rc0.pt')

    atoms = read(filename=read_file_path, format=read_file_format)

    if only_relax_H:
        mask = [index for index, atomic_number in enumerate(atoms.get_atomic_numbers()) if atomic_number != 1]
        constraint = FixAtoms(indices=mask)
        atoms.set_constraint(constraint)

    calculator = DP(model=dp_model_path, head=head)
    atoms.calc = calculator

    if traj_path is not None:
        # 写入第一帧
        write(traj_path, atoms, format=traj_format, append=False)

    class TrajCallback:
        def __init__(self, path, fmt, interval):
            self.path = path
            self.fmt = fmt
            self.interval = interval
            self.counter = 0

        def __call__(self):
            if self.path is None:
                return
            self.counter += 1
            if self.counter % self.interval == 0:
                write(self.path, atoms, format=self.fmt, append=True)

    callback = TrajCallback(traj_path, traj_format, traj_interval)

    optimizer = BFGS(atoms, maxstep=maxstep, logfile='-')
    optimizer.attach(callback, interval=1)
    optimizer.run(fmax=f_max)

    write(filename=write_file_path, images=atoms, format=write_file_format)
    return atoms


if __name__ == '__main__':
    # atoms = opt_a_mol(
    #     read_file_path='reico_random_structs/structure_000001.cif',
    #     read_file_format='cif',
    #     write_file_path='opted.cif',
    #     write_file_format='cif',
    #     dp_model_path='DPA3_finetune_zeo_iter009_GaHY_01.pth',
    #     head='GaHY',
    #     traj_path='optimization_traj.pdb',
    #     traj_format='proteindatabank',
    #     traj_interval=1
    # )
    from glob import glob

    opted_root = 'reico_random_structs_opted'
    os.makedirs(opted_root, exist_ok=True)

    for file in glob('reico_random_structs/*.cif'):
        print(file)
        atoms = opt_a_mol(
            read_file_path=file,
            read_file_format='cif',
            write_file_path=os.path.join(opted_root, os.path.basename(file)),
            write_file_format='cif',
            dp_model_path='DPA3_finetune_zeo_iter009_GaHY_01.pth',
            head='GaHY',
            traj_path=os.path.join(opted_root, f'{os.path.basename(file).split('.')[0]}_optimization_traj.pdb'),
            traj_format='proteindatabank',
            traj_interval=1
        )