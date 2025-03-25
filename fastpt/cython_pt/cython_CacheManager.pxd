cimport numpy as np

cdef class CacheManager_cy:
    cdef dict cache
    cdef dict hit_counts
    cdef Py_ssize_t cache_size
    cdef Py_ssize_t max_size_bytes
    cdef Py_ssize_t hits
    cdef Py_ssize_t misses
    
    cdef Py_ssize_t _get_array_size(self, object arr)
    cdef void _evict(self, Py_ssize_t required_size)
    
    cpdef str measure_actual_size(self)
    cpdef get(self, str category, object hash_key)
    cpdef set(self, value, str category, object hash_key)
    cpdef void clear(self)
    cpdef dict stats(self)