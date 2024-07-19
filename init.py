import numpy as np
import sys
sys.path.append('/home/bj21/programs/')
from martini3 import molecules
from martini3 import init_cell
import hoomd
import os


def gen_grid(low_bound, up_bound, num_spaces=100):
    grid = []
    x_arr = np.linspace(low_bound, up_bound, num_spaces)
    for x in x_arr:
        for y in x_arr:
            grid.append((x, y))
    return grid


def main(seq,id,name_a,name_b):

    path = "data/" + str(id) +"/"
    if not os.path.exists(path):
        os.makedirs(path)
    # contents setup
    contents = molecules.Contents()
    x_box_size = 50
    z_box_size = 50
    total_polymers = 1
    polym_placed = 0
    positions = np.empty((0, 3))
    np.random.seed()
    low_bound = -x_box_size / 2 + .25
    up_bound = x_box_size / 2 - .25
    low_bound_z = -z_box_size / 2 + .5
    up_bound_z = z_box_size/2 -.5

    grid_polym = gen_grid(low_bound, up_bound, num_spaces=int(x_box_size * 50))
    z_grid = np.linspace(low_bound_z, up_bound_z, int(z_box_size * 400))

    num_molecule_beads = 0
    while polym_placed < total_polymers:
        # 1. generate a coordinate and rotation
        # 2. pick if we want to place polym or lipid
        # 3. attempt to place and compare distance
        # 4. if placed, add position to list
        polym_percent = polym_placed / total_polymers
        # lipid_percent = lipid_placed/total_lipids
        is_inverted = bool(np.random.randint(0, 2))
        z_shift = z_grid[int(np.random.randint(0, len(z_grid)))]
        
        rand_pos = int(np.random.randint(0, len(grid_polym)))
        x_shift, y_shift = grid_polym[rand_pos]
        # molecule = molecules.make_PCLbPEO(contents,5,10,x_shift = x_shift,y_shift = y_shift, z_shift = z_shift, is_inverted=is_inverted)

        x_shift, y_shift, z_shift = 0, 0, 0

        molecule = molecules.make_seq_def(
            contents,
            name_a,
            name_b,
            seq,
            id,
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
            is_inverted=is_inverted,
            bond_angle=135
        )
        
        del grid_polym[rand_pos]
        molecule_position = np.array(molecule.position)
        print("molecule_position = ", molecule_position)
        print("First boolean: ", (
            np.all(
                np.linalg.norm(molecule_position[:, np.newaxis, :] - positions, axis=2)
                > 0.43
            )))
        print("second boolean: ", (np.all(
                (molecule_position[:,2])**2 <low_bound_z**2) and np.all(
                (molecule_position[:,1])**2 <low_bound**2) and np.all((molecule_position[:,0])**2 <low_bound**2)))
        if (len(positions) == 0 or (
            np.all(
                np.linalg.norm(molecule_position[:, np.newaxis, :] - positions, axis=2)
                > 0.43
            ))) and  (np.all(
                (molecule_position[:,2])**2 <low_bound_z**2) and np.all(
                (molecule_position[:,1])**2 <low_bound**2) and np.all((molecule_position[:,0])**2 <low_bound**2)):
            num_molecule_beads+=20
            contents.add_molecule(molecule)
            print(positions)
            positions = np.append(positions, molecule_position, axis=0)
            print(positions)
            polym_placed = polym_placed + 1
            print("if statement entered")
            print(contents.contents[0].position)

    num_water = 0
    x_place_water = np.linspace(
        -x_box_size / 2 + 0.25, x_box_size / 2 - 0.25, 15 #int(x_box_size/10)
    )
    y_place_water = np.linspace(
        -x_box_size / 2 + 0.25, x_box_size / 2 - 0.25, 1 #int(x_box_size/10)
    )
    z_place_water = np.linspace(
        -z_box_size / 2 + 0.25, z_box_size / 2 - 0.25, 1 #int(z_box_size/10)
    )
    all_pos = np.array(positions)
    for i in x_place_water:
        for j in y_place_water:
            for k in z_place_water:
                if np.min(
                        np.linalg.norm(np.array([i, j, k]) - all_pos, axis=1) > 0.4
                    ):
                    if num_water+num_molecule_beads<8.4*x_box_size**2*z_box_size:
                        contents = molecules.add_water(
                            contents, x_shift=i, y_shift=j, z_shift=k
                        )
                        num_water = num_water+1
                    else:
                        1+1
    lj, coulomb, bond_harmonic, angle_forces, dihedrals,improper,rigid = init_cell.init_cell(
        contents, path, box_size=[x_box_size, x_box_size, z_box_size], pair_on=False
    )

    # init cell also saves the gsd


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 init.py sequence id name_a name_b")
        sys.exit(1)
    import csv
    file_name = "seq_id.csv"
    with open(file_name,'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(sys.argv[3]),str(sys.argv[4]),str(sys.argv[1]),str(sys.argv[2])])
    main(str(sys.argv[1]), str(sys.argv[2]),str(sys.argv[3]),str(sys.argv[4]))
