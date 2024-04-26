import numpy as np


def check_bonds(bonds_new, bonds_old):
    for j in range(bonds_new.shape[0]):
        for i in range(bonds_new.shape[1]):

            # Up neighbor
            if i%4==0:

                if j != 0:
                    row = 128 + (j-1)

                else:
                    row = -1

                if j%2 == 0:

                    check = bonds_new[j,i] == bonds_old[row,2*int(i/4)]

                else:
                    check = bonds_new[j,i] == bonds_old[row,2*int(i/4)+1]

                if check == False:
                    print("Up error", i)

            # Down neighbor
            if i%4==1:

                row = 128 + j

                if j%2 == 0:

                    check = bonds_new[j,i] == bonds_old[row,2*int(i/4)]

                else:
                    check = bonds_new[j,i] == bonds_old[row, 2*int(i/4)+1]

                if check == False:
                    print("Down error", i)

            # Left neighbor
            if i%4 == 2:

                loc_i = int(i/4)

                if j%2 == 0:
                    if loc_i == 0:
                        check = bonds_new[j,i] == bonds_old[j, -1]

                    else:
                        check = bonds_new[j,i] == bonds_old[j, 2*loc_i - 1]

                    if check == False:
                        print("LEFT error", j, i)

                else:
                    check = bonds_new[j,i] == bonds_old[j, 2*loc_i]

                    if check == False:
                        print("LEFT error uneven", j, i)

            # Right neighbor
            if i%4 == 3:

                loc_i = int(i/4)

                if j%2 == 0:
                    check = bonds_new[j,i] == bonds_old[j, 2*loc_i]

                    if check == False:
                        print("RIGHT ERROR even", j,i)

                else:
                    if loc_i == 63:
                        check = bonds_new[j,i] == bonds_old[j,-1]

                    else:
                        check = bonds_new[j,i] == bonds_old[j,2*loc_i+1]

                    if check == False:
                        print("RIGHT ERROR", j,i)

def get_interactions(bonds, X, Y):
    """
    This function transforms the given bonds from the optimized code version to the bonds used in
    the basic cuda example. For this function to work, the bonds need to be written out by the
    corresponding c++ function with hamW_d as an input.

    Args:
        bonds (np.ndarray): hamW_d interactions written out by c++
        X (int): row-size of lattice
        Y (int): column size of lattice

    Returns:
        interactions: List
    """
    Y2 = int(Y/2)

    interactions = np.zeros((2*X*Y))

    for i in range(bonds.shape[0]):
        for j in range(bonds.shape[1]):

            j_spin = int(j/4)

            ipp = (i + 1) if (i + 1 < X) else 0 #Für open boundary hier einfach nur bis zum vorletzten iterieren? und wird auch garnicht genutzt
            inn = (i - 1) if (i - 1 >= 0) else X - 1
            jpp = (j_spin + 1) if (j_spin + 1 < Y2) else 0 #warum Y2 hier?
            jnn = (j_spin - 1) if (j_spin - 1 >= 0) else Y2 - 1

            icpp = 2*(X-1)*Y2 + 2*(Y2*(i+1) + j_spin) + i%2 # das wird unten nochmal so initialisiert - kommt heir eigentlich noch ein Part dazwischen dann?
            icnn = 2*(X-1)*Y2 + 2*(Y2*(inn+1) + j_spin) + i%2

            joff = jpp if (i%2) else jnn


            if (i % 2):
                if (j + 1 > Y2):
                    jcoff = 2*(i*Y2 + j_spin + 1) - 1

                else:
                    jcoff = 2*(i*Y2 + joff) - 1
            else:
                jcoff = 2 * (i*Y2 + joff) + 1

            icpp = 2*(X-1)*Y2 + 2*(Y2*(i+1) + j_spin) + i%2

            # Up neighbor
            if j%4 == 0:
                interactions[icnn] = bonds[i,j]

            # Down neighbor
            if j%4 == 1:
                interactions[icpp] = bonds[i,j]

            # Left neighbor
            if j%4 == 2:
                if i%2 == 0:
                    interactions[jcoff] = bonds[i,j]
                else:
                    interactions[2*(i*Y2 + j_spin)] = bonds[i,j]

            # Right neighbor
            if j%4 == 3:
                if i%2 == 0:
                    interactions[2*(i*Y2 + j_spin)] = bonds[i,j]
                else:
                    interactions[jcoff] = bonds[i,j]

    interactions[interactions==1] = -1
    interactions[interactions==0] = 1

    # # Path anpassen
    # np.savetxt(f'test_rng/bonds/bonds_seed_{seed*10}.txt', interactions, fmt = "%i")

    return interactions