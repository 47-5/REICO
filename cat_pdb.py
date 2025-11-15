from ase.io import read, write
from glob import glob

# pdb文件必须原子数一样

if __name__ == '__main__':

    root_path = 'reico_random_structs_opted'
    atoms_list = []
    for index, file_path in enumerate(glob(f'{root_path}/*.pdb')):
        print(file_path)
        atoms = read(filename=file_path, index=':')
        atoms_list += atoms

    print(atoms_list)
    print(len(atoms_list))

    write('all.pdb', images=atoms_list)

