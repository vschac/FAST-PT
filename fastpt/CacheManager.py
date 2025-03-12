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
        hashed_args = []
        for arg in args:
            hashed_args.append(self._hash_arrays(arg))
        
        return (category, tuple(hashed_args))
    
    def _hash_arrays(self, arrays):
        """Helper function to create a hash from multiple numpy arrays or scalars"""
        if arrays is None: return None
        if isinstance(arrays, (tuple, list)):
            result = []
            for item in arrays:
                if isinstance(item, np.ndarray):
                    result.append(hash(item.tobytes()))
                elif isinstance(item, (tuple, list)):
                    result.append(self._hash_arrays(item))
                else:
                    result.append(hash(item))
            return tuple(result)
    
        # Single item case
        if isinstance(arrays, np.ndarray):
            return hash(arrays.tobytes())
        return hash(arrays)
    
    def get(self, category, *args):
        """Get an item from cache using category and arguments as key"""
        #key = self._create_key(category, *args)
        key = (category, args)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, value, category, *args):
        """Store an item in cache using category and arguments as key"""
        #key = self._create_key(category, *args)
        key = (category, args)
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
    
    def __repr__(self):
        """Return a string representation of the cache"""
        stats = self.stats()
        result = [
            f"CacheManager: {stats['size_mb']:.2f}MB/{stats['max_size_mb']:.2f}MB used",
            f"Items: {stats['items']}, Hit rate: {stats['hit_rate']:.2%}"
        ]
        
        if stats['items'] > 0:
            result.append("\nCached items:")
            for key, value in self.cache.items():
                category, args_hash = key
                
                # Format the value representation
                if isinstance(value, np.ndarray):
                    shape_str = f"shape={value.shape}"
                    dtype_str = f"dtype={value.dtype}"
                    if value.size > 5:
                        # For large arrays, show a few elements and shape/dtype
                        val_repr = f"ndarray({shape_str}, {dtype_str}, first few: {value.flat[:3]}...)"
                    else:
                        # For small arrays, show all values
                        val_repr = f"ndarray({shape_str}, {dtype_str}, values: {value})"
                elif isinstance(value, (list, tuple)):
                    val_type = type(value).__name__
                    length = len(value)
                    if length > 3:
                        first_items = []
                        for i, item in enumerate(value[:2]):
                            if isinstance(item, np.ndarray):
                                first_items.append(f"ndarray(shape={item.shape})")
                            else:
                                first_items.append(str(item))
                        val_repr = f"{val_type} of {length} items: [{', '.join(first_items)}...]"
                    else:
                        val_repr = f"{val_type} of {length} items"
                else:
                    val_repr = str(value)
                
                # Truncate representation if too long
                if len(val_repr) > 100:
                    val_repr = val_repr[:97] + "..."
                
                result.append(f"  • {category}: {val_repr}")
        
        return "\n".join(result)