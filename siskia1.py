from siskia1 import MPI
import siskia1 as np
import time

def parallel_bubble_sort(data, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_data = np.array_split(data, size)[rank]

    n = len(local_data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if local_data[j] > local_data[j + 1]:
                local_data[j], local_data[j + 1] = local_data[j + 1], local_data[j]

    sorted_data = comm.gather(local_data, root=0)

    if rank == 0:
        merged_data = np.concatenate(sorted_data)
        merged_data.sort()
        return merged_data
    else:
        return None

if __name__ == "_main_":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        num_elements = 5
        data = np.random.randint(0, 100000000, num_elements)

    else:
        data = None

    data = comm.bcast(data, root=0)

    start_time = time.time()
    sorted_data = parallel_bubble_sort(data, comm)
    end_time = time.time()

    if rank == 0:
        print("Array yang diurutkan:", sorted_data)
        elapsed_time = end_time - start_time
        print(f"Waktu pemrosesan: {elapsed_time:.4f} detik")