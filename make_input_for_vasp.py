from ase.io import read, write
from ase.io.vasp import write_vasp
import os
import shutil
from glob import glob


def make_POSCAR(input_file_path, input_file_format=None, output_root='.'):
    atoms = read(input_file_path, format=input_file_format)
    os.makedirs(output_root, exist_ok=True)
    write_vasp(os.path.join(output_root, 'POSCAR'), atoms=atoms, sort=True)
    return atoms


def make_POTCAR(input_POSCAR_path, output_root='.', element_potcar_root_path='potpaw_PBE'):
    poscar = open(input_POSCAR_path).readlines()
    elements = [i for i in poscar[5].strip().split(' ') if i]
    print(elements)
    os.makedirs(output_root, exist_ok=True)

    # 定义输出 POTCAR 文件路径
    output_POTCAR_path = os.path.join(output_root, 'POTCAR')
    # 将所有元素的 POTCAR 文件内容合并到一个 POTCAR 文件中
    with open(output_POTCAR_path, 'w') as outfile:
        for element in elements:
            # 假设每个元素的 POTCAR 文件命名格式为 "ELEMENT/POTCAR"
            potcar_file_path = os.path.join(element_potcar_root_path, element, 'POTCAR')
            with open(potcar_file_path) as infile:
                outfile.write(infile.read())
    return None


def make_INCAR(template_incar_path='INCAR', output_root='.'):
    os.makedirs(output_root, exist_ok=True)
    shutil.copy(template_incar_path, output_root)
    return None


def make_vasp_input(input_file_path, input_file_format=None, output_root='.', element_potcar_root_path='potpaw_PBE', template_incar_path='INCAR'):
    os.makedirs(output_root, exist_ok=True)
    atoms = make_POSCAR(input_file_path=input_file_path, input_file_format=input_file_format, output_root=output_root)
    make_POTCAR(input_POSCAR_path=os.path.join(output_root, 'POSCAR'), output_root=output_root, element_potcar_root_path=element_potcar_root_path)
    make_INCAR(template_incar_path=template_incar_path, output_root=output_root)
    return atoms

if __name__ == '__main__':

    run_mode = 'batch'
    if run_mode == 'single':
        make_vasp_input(input_file_path='reico_random_structs_opted/structure_000001.cif',
                        output_root='vasp_input', element_potcar_root_path='potpaw_PBE', template_incar_path='INCAR')

    elif run_mode == 'batch':
        batch_root = 'batch_vasp'

        input_file_path_list = glob('./reico_random_structs_opted/*.cif')
        input_file_path_list.sort()

        for i in input_file_path_list:
            output_root = os.path.join(batch_root, os.path.basename(i).split('.')[0])
            make_vasp_input(input_file_path=i,
                            output_root=output_root, element_potcar_root_path='potpaw_PBE', template_incar_path='INCAR')