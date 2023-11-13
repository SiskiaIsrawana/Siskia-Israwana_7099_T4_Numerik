from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def generate_random_matrix(rows, cols):
    return np.random.rand(rows, cols)

def main():
    ordo = 3

    baris = ordo
    kolom = ordo + 1

    A = generate_random_matrix(baris, kolom)

    start_time = time.time()

    for i in range(baris):
        diag = A[i][i]
        for j in range(kolom):
            A[i][j] /= diag
        for k in range(i + 1, baris):
            diag1 = A[k][i]
            for j in range(0, kolom):
                A[k][j] = A[k][j] - diag1 * A[i][j]

    comm.barrier()

    if rank == 0:
        print("Eliminasi Gauss:")
        for i in range(min(5, baris)):
            for j in range(kolom):
                print(f"{A[i][j]:.2f}", end=" ")
            print("...")

        x = np.zeros(ordo, dtype=float)
        for i in range(ordo - 1, -1, -1):
            x[i] = A[i][kolom - 1]
            for j in range(i + 1, ordo):
                x[i] -= A[i][j] * x[j]

        end_time = time.time()
        elapsed_time = end_time - start_time

        for i in range(min(5, ordo)):
            print(f"x{i + 1} = {x[i]:.2f}")

        print(f"Estimasi Waktu proses: {elapsed_time:.6f} detik")

if _name_ == '_main_':
    main()