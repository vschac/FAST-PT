import numpy as np
import sys
import sys

class CacheManager:
    """Unified cache manager for FASTPT with memory efficiency features"""
    
    def __init__(self, max_size_mb=500):
        """Initialize cache with optional maximum size in MB"""
        self.cache = {}
        self.hit_counts = {}  # Track hits per cache item
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
    
    def get(self, category, *args):
        """Get an item from cache using category and arguments as key"""
        key = (category, args)
        if key in self.cache:
            self.hits += 1
            # Track hit counts per item
            if key in self.hit_counts:
                self.hit_counts[key] += 1
            else:
                self.hit_counts[key] = 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, value, category, *args):
        """Store an item in cache using category and arguments as key"""
        key = (category, args)
        old_size = 0
        if key in self.cache.keys(): # Check if we're replacing an existing entry
            old_val = self.cache[key]
            if isinstance(old_val, np.ndarray):
                old_size = self._get_array_size(old_val)
            elif isinstance(old_val, (list, tuple)):
                for item in old_val:
                    old_size += self._get_array_size(item)
        else:
            # Initialize hit counter for new entries
            self.hit_counts[key] = 0
        
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
            # Also remove hit count for evicted item
            if key in self.hit_counts:
                del self.hit_counts[key]
            self.cache_size -= size
            freed += size
    
    def clear(self):
        """Clear the entire cache"""
        self.cache.clear()
        self.hit_counts.clear()
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
            # Group items by category
            categories = {}
            for key, value in self.cache.items():
                category, args = key
                if category not in categories:
                    categories[category] = []
                hit_count = self.hit_counts.get(key, 0)
                categories[category].append((args, value, hit_count))
        
            result.append("\nCached items by category:")
            for category, items in sorted(categories.items()):
                result.append(f"\n  Category: {category} ({len(items)} items)")
            
                for args, value, hit_count in items:
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
                
                    # Format arguments based on FAST-PT patterns
                    if len(args) == 1:
                        # Single argument - likely a hashed power spectrum or common parameter
                        if isinstance(args[0], int) and abs(args[0]) > 10000000:
                            args_repr = "(hashed P)"
                        else:
                            args_repr = f"({args[0]})"
                        
                    elif len(args) == 2:
                        # Two arguments - often (hashed P, P_window)
                        if isinstance(args[0], int) and abs(args[0]) > 10000000:
                            if isinstance(args[1], int) and abs(args[1]) > 10000000:
                                args_repr = "(hashed P, hashed P_window)"
                            elif args[1] is None:
                                args_repr = "(hashed P, None)"
                            else:
                                args_repr = f"(hashed P, {args[1]})"
                        else:
                            args_repr = f"({args[0]}, {args[1]})"
                        
                    elif len(args) == 3:
                        # Three arguments - often (hashed P, P_window, C_window)
                        if isinstance(args[0], int) and abs(args[0]) > 10000000:
                            if isinstance(args[1], int) and abs(args[1]) > 10000000:
                                if isinstance(args[2], float) and (args[2] == 0.0 or args[2] == 1.0 or args[2] == 0.75):
                                    args_repr = f"(hashed P, hashed P_window, C_window={args[2]})"
                                elif args[2] is None:
                                    args_repr = "(hashed P, hashed P_window, None)"
                                else:
                                    args_repr = f"(hashed P, hashed P_window, {args[2]})"
                            else:
                                if args[2] is None:
                                    args_repr = f"(hashed P, {args[1]}, None)"
                                else:
                                    args_repr = f"(hashed P, {args[1]}, {args[2]})"
                        else:
                            args_repr = f"({args[0]}, {args[1]}, {args[2]})"
                        
                    elif len(args) > 3:
                        # For multiple hashes in fourier coefficients or convolution caching
                        hash_parts = []
                        for arg in args:
                            if isinstance(arg, int) and abs(arg) > 10000000:
                                hash_parts.append("hash")
                            elif arg is None:
                                hash_parts.append("None")
                            elif isinstance(arg, (float, int)) and -10 <= arg <= 10:
                                hash_parts.append(str(arg))
                            else:
                                hash_parts.append(f"{type(arg).__name__}")
                    
                        # Common patterns in FAST-PT cache keys
                        if category == "fourier_coefficients" and len(hash_parts) == 2:
                            args_repr = "(hashed P_b, C_window)"
                        elif category == "convolution" and len(hash_parts) == 6:
                            args_repr = "(hashed c1, hashed c2, hashed g_m, hashed g_n, hashed h_l, hashed two_part_l)"
                        elif category == "J_k_scalar" and len(hash_parts) == 5:
                            args_repr = "(hashed P, hashed X, nu, hashed P_window, C_window)"
                        elif category == "J_k_tensor" and len(hash_parts) == 4:
                            args_repr = "(hashed P, hashed X, hashed P_window, C_window)"
                        else:
                            args_repr = f"({', '.join(hash_parts)})"
                    else:
                        args_repr = "()"
                
                    result.append(f"    • {args_repr} [hits: {hit_count}]: {val_repr}")
    
        return "\n".join(result)