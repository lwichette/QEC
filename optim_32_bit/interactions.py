import numpy as np

def get_interactions(bonds, X, Y, seed):
    """
    This function transforms the given bonds from the optimized code version to the bonds used in 
    the basic cuda example. For this function to work, the bonds need to be written out by the 
    corresponding c++ function with hamW_d as an input.
    
    Args:
        bonds (np.ndarray): hamW_d interactions written out by c++
        X (int): row-size of lattice
        Y (int): column size of lattice 
        seed (int): seed used to write lattice to corresponding folder

    Returns:
        interactions: List
    """
    Y2 = int(Y/2)
    
    interactions = np.zeros((2*X*Y))

    for i in range(bonds.shape[0]):
        for j in range(bonds.shape[1]):

            j_spin = int(j/4)

            ipp = (i + 1) if (i + 1 < X) else 0
            inn = (i - 1) if (i - 1 >= 0) else X - 1
            jpp = (j_spin + 1) if (j_spin + 1 < Y2) else 0
            jnn = (j_spin - 1) if (j_spin - 1 >= 0) else Y2 - 1

            icpp = 2*(X-1)*Y2 + 2*(Y2*(i+1) + j_spin) + i%2
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
    
    # Path anpassen
    np.savetxt(f'test_rng/bonds/bonds_seed_{seed*10}.txt', interactions, fmt = "%i")

    return interactions