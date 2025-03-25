import numpy as np
import sys
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

cdef class CacheManager_cy:
    """Unified cache manager for FASTPT with memory efficiency features"""
    
    def __cinit__(self, Py_ssize_t max_size_mb=500):
        """Initialize cache with optional maximum size in MB"""
        self.cache = {}
        self.hit_counts = {}
        self.cache_size = 0
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.hits = 0
        self.misses = 0

    cpdef str measure_actual_size(self):
        """Measure actual memory usage of the cache"""
        try:
            from pympler import asizeof
            actual_size = asizeof.asizeof(self.cache) / (1024 * 1024)
            return f"{actual_size:.2f}"
        except ImportError:
            pass
        
        cdef Py_ssize_t basic_size = sys.getsizeof(self.cache) / (1024 * 1024)
        return f"Basic cache size (sys.getsizeof): {basic_size:.2f} MB"
    
    cdef Py_ssize_t _get_array_size(self, object arr):
        """Calculate size of objects in bytes, accounting for Python objects"""
        cdef Py_ssize_t container_overhead = 0
        
        if isinstance(arr, np.ndarray):
            return arr.nbytes
        elif isinstance(arr, (tuple, list)):
            container_overhead = sys.getsizeof(arr) - sum(sys.getsizeof(0) for _ in range(len(arr)))
            return container_overhead + sum(self._get_array_size(item) for item in arr)
        elif isinstance(arr, (int, float, str, bool)):
            return sys.getsizeof(arr)
        elif arr is None:
            return sys.getsizeof(None)
        else:
            try:
                return sys.getsizeof(arr)
            except:
                return 64  # Default estimate if sys.getsizeof fails
    
    cpdef get(self, str category, object hash_key):
        """Get an item from cache using category and arguments as key"""
        cdef tuple key = (category, hash_key)
        if key in self.cache:
            self.hits += 1
            self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
            return self.cache[key]
        self.misses += 1
        return None
    
    cpdef set(self, value, str category, object hash_key):
        """Store an item in cache using category and arguments as key"""
        cdef tuple key = (category, hash_key)
        cdef Py_ssize_t key_size = self._get_array_size(key)
        cdef Py_ssize_t old_size = 0
        cdef Py_ssize_t value_size
        cdef Py_ssize_t total_size
        
        if key in self.cache:
            old_size = self._get_array_size(self.cache[key])
        else:
            self.hit_counts[key] = 0
        
        value_size = self._get_array_size(value)
        total_size = key_size + value_size
    
        if self.max_size_bytes > 0 and (self.cache_size - old_size + total_size) > self.max_size_bytes:
            self._evict(total_size - old_size)
    
        self.cache[key] = value
        self.cache_size = self.cache_size - old_size + total_size
        return value
    
    cdef void _evict(self, Py_ssize_t required_size):
        """Evict items from cache until there's room for required_size"""
        cdef list items = list(self.cache.items())
        cdef Py_ssize_t freed = 0
        cdef tuple key
        cdef object value
        cdef Py_ssize_t key_size
        cdef Py_ssize_t value_size
        cdef Py_ssize_t total_size
        
        np.random.shuffle(items)
    
        for key, value in items:
            if freed >= required_size:
                break
            
            key_size = self._get_array_size(key)
            value_size = self._get_array_size(value)
            total_size = key_size + value_size
        
            del self.cache[key]
            self.hit_counts.pop(key, None)
            self.cache_size -= total_size
            freed += total_size
    
    cpdef void clear(self):
        """Clear the entire cache"""
        self.cache.clear()
        self.hit_counts.clear()
        self.cache_size = 0
    
    cpdef dict stats(self):
        """Return statistics about the cache usage"""
        cdef Py_ssize_t key_size = 0
        cdef Py_ssize_t value_size = 0
        cdef Py_ssize_t total_size_bytes
        cdef double total_size_mb
        cdef double key_size_mb
        cdef double value_size_mb
        cdef Py_ssize_t total_accesses
        cdef double hit_rate
        
        for key, value in self.cache.items():
            key_size += self._get_array_size(key)
            value_size += self._get_array_size(value)
        
        total_size_bytes = self.cache_size
        total_size_mb = total_size_bytes / (1024 * 1024)
        key_size_mb = key_size / (1024 * 1024)
        value_size_mb = value_size / (1024 * 1024)
        
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            'items': len(self.cache),
            'size_mb': total_size_mb,
            'key_size_mb': key_size_mb,
            'value_size_mb': value_size_mb,
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses
        }