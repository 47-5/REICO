from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import FixAtoms, UnitCellFilter
import deepmd
from deepmd.calculator import DP
import os


__all__ = ['opt_a_mol']


def opt_a_mol(read_file_path, read_file_format=None, write_file_path='opted.xyz', write_file_format='xyz',
              dp_model_path=None, head=None, only_relax_H=False, maxstep=0.05, f_max=0.05,
              traj_path=None, traj_format='xyz', traj_interval=1, opt_cell=False):
    """
    Optimizes a molecular structure using a deep potential model and writes the optimized structure to a file.

    Summary:
    This function reads a molecular structure from a specified file, optimizes it using a deep potential model,
    and then writes the optimized structure to another file. The optimization can be restricted to only hydrogen atoms
    if specified. Optionally, it can record the optimization trajectory in an ASE-compatible format, and when
    supported by the calculator, it can also optimize the cell parameters.

    Parameters:
    - read_file_path (str): Path to the file containing the initial molecular structure.
    - read_file_format (Optional[str]): Format of the input file. If not provided, ASE infers it from the extension.
    - write_file_path (str): Path where the optimized structure will be written. Default: 'opted.xyz'.
    - write_file_format (str): Format of the output file. Default: 'xyz'.
    - dp_model_path (Optional[str]): Path to the deep potential model file. Defaults to 'DPA2_medium_28_10M_rc0.pt'.
    - head (Optional[str]): Optional head name for multi-head DeepMD models.
    - only_relax_H (bool): If True, only hydrogen atoms are allowed to move during optimization.
    - maxstep (float): Maximum optimizer step size. Default: 0.05.
    - f_max (float): Force convergence criterion. Default: 0.05.
    - traj_path (Optional[str]): Path to the trajectory file. If None, no trajectory is written.
    - traj_format (str): Format used when writing the trajectory (e.g. 'xyz', 'pdb'). Default: 'xyz'.
    - traj_interval (int): Write every Nth step to the trajectory. Default: 1.
    - opt_cell (bool): If True, attempt to optimize the cell parameters (requires stress from the calculator).

    Returns:
    - ase.Atoms: The optimized atomic structure.
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

    optimizer_target = atoms
    trajectory_atoms = atoms

    if opt_cell:
        try:
            _ = atoms.get_stress()
        except Exception as exc:
            raise RuntimeError("DP 模型当前不提供应力，无法开启晶胞优化。") from exc

        uc_filter = UnitCellFilter(atoms)
        optimizer_target = uc_filter
        trajectory_atoms = uc_filter.atoms

    if traj_path is not None:
        write(traj_path, trajectory_atoms, format=traj_format, append=False)

    class TrajCallback:
        def __init__(self, path, fmt, interval, atoms_ref):
            self.path = path
            self.fmt = fmt
            self.interval = interval
            self.counter = 0
            self.atoms_ref = atoms_ref

        def __call__(self):
            if self.path is None:
                return
            self.counter += 1
            if self.counter % self.interval == 0:
                write(self.path, self.atoms_ref, format=self.fmt, append=True)

    callback = TrajCallback(traj_path, traj_format, traj_interval, trajectory_atoms)

    optimizer = BFGS(optimizer_target, logfile='-')
    optimizer.attach(callback, interval=1)
    optimizer.run(fmax=f_max)

    write(filename=write_file_path, images=atoms, format=write_file_format)
    return atoms


if __name__ == '__main__':
    atoms = opt_a_mol(
        read_file_path='reico_random_structs/structure_000001.cif',
        read_file_format='cif',
        write_file_path='opted.cif',
        write_file_format='cif',
        dp_model_path='DPA3_finetune_zeo_iter009_GaHY_01.pth',
        head='GaHY',
        traj_path='optimization_traj.pdb',
        traj_format='proteindatabank',
        traj_interval=1,
        opt_cell=True
    )