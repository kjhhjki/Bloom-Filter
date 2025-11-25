import numpy as np
from numba import njit, prange

gseed1 = np.random.randint(2 ** 32, 2**64, dtype = np.uint64)
gseed2 = np.random.randint(2 ** 32, 2**64, dtype = np.uint64)
while gseed2 == gseed1:
    gseed2 = np.random.randint(0, 2**64, dtype = np.uint64)
@njit(inline = 'always')
def mix(h : np.uint64) -> np.uint64:
    h ^= (h >> 30)
    h *= 0xBF58476D1CE4E5B9
    h ^= (h >> 27)
    h *= 0x94D049BB133111EB
    h ^= (h >> 31)
    return h
@njit(inline = 'always')
def hash(x : np.uint64, seed1 : np.uint64 = gseed1, seed2 : np.uint64 = gseed2) -> np.uint64:
    h1 = seed1
    h2 = seed2
    for i in range(8):
        h1 ^= (np.uint8(x) + 0x9E3779B97F4A7C15 + (h1 << 6) + (h1 >> 2))
        h2 ^= (np.uint8(x) + 0xC6EF372FE94F82BE + (h2 << 6) + (h2 >> 2))
        x >>= 8
    return mix(h1), mix(h2)

@njit
def bloomfilter__insert(bit_array, item : np.uint64, k : np.uint8, m : np.uint32):
    h1, h2 = hash(item)
    for i in range(k):
        pos = np.uint32((h1 + i * h2) % m)
        bit_array[pos >> 3] |= np.uint8(1 << (pos & 7))
@njit
def bloomfilter__contains(bit_array, item : np.uint64, k : np.uint8, m : np.uint32):
    h1, h2 = hash(item)
    for i in range(k):
        pos = np.uint32((h1 + i * h2) % m)
        if (bit_array[pos >> 3] & np.uint8(1 << (pos & 7))) == 0:
            return False
    return True

@njit(parallel = True)
def bloomfilter__batch_insert(bit_array, items : np.ndarray, k : np.uint8, m : np.uint32):
    for idx in prange(items.shape[0]):
        bloomfilter__insert(bit_array, items[idx], k, m)
@njit(parallel = True)
def bloomfilter__batch_contains(bit_array, items : np.ndarray, k : np.uint8, m : np.uint32) -> np.ndarray:
    results = np.empty(items.shape[0], dtype = np.bool_)
    for idx in prange(items.shape[0]):
        results[idx] = bloomfilter__contains(bit_array, items[idx], k, m)
    return results
class BloomFilter:
    def __init__(self, n : int, p : float):
        self.n = np.uint32(n)
        self.p = p
        self.m = self._get_size(n, p)
        self.k = self._get_hash_count(self.m, n)
        self.bit_array = np.zeros((self.m + 7) // 8, dtype = np.uint8)
        
    def _get_size(self, n : int, p : float) -> int:
        m = -(n * np.log(p)) / (np.log(2) ** 2)
        return np.uint32(np.ceil(m))
    def _get_hash_count(self, m : int, n : int) -> int:
        k = (m / n) * np.log(2)
        return np.uint8(np.ceil(k))

    def insert(self, item):
        bloomfilter__insert(self.bit_array, item, self.k, self.m)
    def contains(self, item) -> bool:
        return bloomfilter__contains(self.bit_array, item, self.k, self.m)

    def batch_insert(self, items : np.ndarray):
        bloomfilter__batch_insert(self.bit_array, items, self.k, self.m)
    def batch_contains(self, items : np.ndarray) -> np.ndarray:
        return bloomfilter__batch_contains(self.bit_array, items, self.k, self.m)

    def signin(self, item : np.uint64): 
        if (not self.contains(item)):

            self.insert(item)
