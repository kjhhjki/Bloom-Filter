import numpy as np
from numba import njit, prange

from bloom import BloomFilter

import tqdm

lb = np.uint64(10 ** 9)
ub = np.uint64(10 ** 10 - 1)
@njit(parallel = True)
def generate(items : np.ndarray, start : np.uint64, p : np.double, item_n : np.uint64):
    n = items.shape[0]
    for i in prange(n):
        if np.random.rand() < p:
            items[i] = np.random.randint(lb, ub)
        else:
            items[i] = start + item_n + np.random.randint(0, item_n)
@njit(parallel = True)
def check(val : np.ndarray, result : np.ndarray, start : np.uint64, item_n : np.uint64) -> np.uint32:
    n = val.shape[0]
    error_cnt = np.uint32(0)
    for i in prange(n):
        # print(val[i], result[i]);
        if result[i]  != (start <= val[i] < start + item_n):
            error_cnt += 1
    return error_cnt

def work():
    bloom = BloomFilter(n = 10 ** 8, p = 0.01)
    print("Bloom Filter created.")
    print(f"Bloom Filter parameters: n = {bloom.n}, p = {bloom.p}, m = {bloom.m}, k = {bloom.k}")

    item_n = 10 ** 8
    start = 2 * 10 ** 9

    batch_size = 262144
    print(f"Inserting {item_n} items into the Bloom Filter.")
    with tqdm.tqdm(total = item_n, desc = "Inserting items", unit = "items") as pbar:
        for batch_start in range(0, item_n, batch_size):
            cur_size = min(batch_size, item_n - batch_start)
            bloom.batch_insert(np.arange(start + batch_start, start + batch_start + cur_size, dtype = np.uint64))
            pbar.update(cur_size)

    print(f"Testing {item_n} items against the Bloom Filter.")
    test_inside_p = np.double(0.237)
    error_cnt = 0
    with tqdm.tqdm(total = item_n, desc = "Testing items", unit = "items") as pbar:
        for batch_start in range(0, item_n, batch_size):
            cur_size = min(batch_size, item_n - batch_start)
            test_items = np.empty(cur_size, dtype = np.uint64)
            generate(test_items, np.uint64(start), test_inside_p, np.uint64(item_n))
            results = bloom.batch_contains(test_items)
            error_cnt += check(test_items, results, np.uint64(start), np.uint64(item_n))
            pbar.update(cur_size)

    print(f"Total errors: {error_cnt}")
    print(f"False positive rate: {error_cnt / item_n:.7f}")
    print(f"Expected false positive rate: {bloom.p:.7f}")

if __name__ == "__main__":
    for case in range(10):
        print(f"#{case + 1}")
        work()
        print()
    
"""

"""