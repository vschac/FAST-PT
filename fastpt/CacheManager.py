import numpy as np
import sys

class CacheManager:
    """Unified cache manager for FASTPT with memory efficiency features"""
    
    def __init__(self, max_size_mb=500):
        """Initialize cache with optional maximum size in MB"""
        self.cache = {}
        self.cache_size = 0
        self.max_size_bytes = max_size_mb * 1024 * 1024
        #^^ 1000 MB = 1000*1024 KB = 1000*1024*1024 bytes (1024 instead of 1000 due to binary memory 2^10=1024)
        self.hits = 0
        self.misses = 0
    
    def _get_array_size(self, arr):
        """Calculate size of numpy array in bytes"""
        if isinstance(arr, np.ndarray):
            return arr.nbytes
        return 0
    
    def _create_key(self, category, *args):
        """Create unified cache key from category and arguments"""
        # Hash numpy arrays, pass through other hashable types
        hashed_args = []
        for arg in args:
            if arg is None:
                hashed_args.append(None)
            elif isinstance(arg, np.ndarray):
                hashed_args.append(hash(arg.tobytes()))
            elif isinstance(arg, (list, tuple)):
                # Handle nested structures
                hashed_args.append(self._hash_nested(arg))
            else:
                hashed_args.append(hash(arg))
        
        return (category, tuple(hashed_args))
    
    def _hash_nested(self, obj):
        """Hash nested lists or tuples containing arrays"""
        if isinstance(obj, (list, tuple)):
            return tuple(self._hash_nested(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return hash(obj.tobytes())
        else:
            return hash(obj)
    
    def get(self, category, *args):
        """Get an item from cache using category and arguments as key"""
        key = self._create_key(category, *args)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, value, category, *args):
        """Store an item in cache using category and arguments as key"""
        key = self._create_key(category, *args)
        old_size = 0
        if key in self.cache.keys(): # Check if we're replacing an existing entry
            old_val = self.cache[key]
            if isinstance(old_val, np.ndarray):
                old_size = self._get_array_size(old_val)
            elif isinstance(old_val, (list, tuple)):
                for item in old_val:
                    old_size += self._get_array_size(item)
        # Calculate size of new entry
        size = 0
        if isinstance(value, np.ndarray):
            size = self._get_array_size(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                size += self._get_array_size(item)
        
        # Check if we need to make room in the cache
        if self.max_size_bytes > 0 and (self.cache_size + size) > self.max_size_bytes:
            self._evict(size)
        
        # Store item and update size
        self.cache[key] = value
        self.cache_size -= old_size
        self.cache_size += size
        return value
    
    def _evict(self, required_size):
        """Evict items from cache until there's room for required_size"""
        # Would LRU be better? 
        # Pros: Keeps frequently used items in cache
        # Cons: Requires additional bookkeeping, may not be worth it for small cache sizes
        items = list(self.cache.items())
        np.random.shuffle(items)
        
        freed = 0
        for key, value in items:
            if freed >= required_size:
                break
                
            size = 0
            if isinstance(value, np.ndarray):
                size = self._get_array_size(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    size += self._get_array_size(item)
            
            del self.cache[key]
            self.cache_size -= size
            freed += size
    
    def clear(self):
        """Clear the entire cache"""
        self.cache.clear()
        self.cache_size = 0
    
    def stats(self):
        """Return cache statistics"""
        return {
            'size_mb': self.cache_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'items': len(self.cache),
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }