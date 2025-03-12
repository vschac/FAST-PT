import pytest
import numpy as np
from fastpt.CacheManager import CacheManager

@pytest.fixture
def cache_manager():
    """Create a basic cache manager instance for testing"""
    return CacheManager(max_size_mb=10)  # Smaller size for testing

@pytest.fixture
def sample_arrays():
    """Create sample arrays of different sizes for testing"""
    small_array = np.ones((10, 10))          # 800 bytes
    medium_array = np.ones((100, 100))       # 80,000 bytes
    large_array = np.ones((500, 500))        # 2,000,000 bytes
    return small_array, medium_array, large_array

####################INITIALIZATION TESTS####################
def test_init_default():
    """Test initialization with default parameters"""
    cm = CacheManager()
    assert cm.max_size_bytes == 500 * 1024 * 1024  # 500 MB
    assert cm.cache == {}
    assert cm.cache_size == 0
    assert cm.hits == 0
    assert cm.misses == 0

def test_init_custom_size():
    """Test initialization with custom size"""
    cm = CacheManager(max_size_mb=100)
    assert cm.max_size_bytes == 100 * 1024 * 1024  # 100 MB

def test_init_unlimited_cache(sample_arrays):
    """Test initialization with unlimited cache size"""
    cm = CacheManager(max_size_mb=0)
    cm.set(sample_arrays[2], "category", "key")
    assert cm.max_size_bytes == 0  # No limit

####################GET/SET TESTS####################
def test_get_set_simple(cache_manager):
    """Test basic get and set operations with simple values"""
    # Set and get a simple value
    cache_manager.set("test_value", "category", 1, 2, 3)
    result = cache_manager.get("category", 1, 2, 3)
    assert result == "test_value"
    
    # Get a non-existent value
    result = cache_manager.get("category", 4, 5, 6)
    assert result is None

def test_get_set_array(cache_manager, sample_arrays):
    """Test get and set operations with arrays"""
    small_array, _, _ = sample_arrays
    
    # Set and get an array
    cache_manager.set(small_array, "array_category", "key1")
    result = cache_manager.get("array_category", "key1")
    assert np.array_equal(result, small_array)

def test_get_set_list_of_arrays(cache_manager, sample_arrays):
    """Test get and set operations with lists of arrays"""
    small_array, _, _ = sample_arrays
    
    # Create a list of arrays
    array_list = [small_array, small_array.copy() * 2]
    
    # Set and get a list of arrays
    cache_manager.set(array_list, "list_category", "key1")
    result = cache_manager.get("list_category", "key1")
    
    # Verify result is a list with correct arrays
    assert isinstance(result, list)
    assert len(result) == 2
    assert np.array_equal(result[0], small_array)
    assert np.array_equal(result[1], small_array * 2)

def test_cache_hits_misses(cache_manager):
    """Test hit and miss counting"""
    # Initial state
    assert cache_manager.hits == 0
    assert cache_manager.misses == 0
    
    # First access (miss)
    cache_manager.get("category", "key1")
    assert cache_manager.hits == 0
    assert cache_manager.misses == 1
    
    # Set value and access again (hit)
    cache_manager.set("value", "category", "key1")
    cache_manager.get("category", "key1")
    assert cache_manager.hits == 1
    assert cache_manager.misses == 1
    
    # Another miss
    cache_manager.get("category", "key2")
    assert cache_manager.hits == 1
    assert cache_manager.misses == 2

####################CACHE SIZE TESTS####################
def test_array_size_calculation(cache_manager, sample_arrays):
    """Test calculation of array sizes"""
    small_array, medium_array, large_array = sample_arrays
    
    # Test size calculation
    small_size = cache_manager._get_array_size(small_array)
    medium_size = cache_manager._get_array_size(medium_array)
    large_size = cache_manager._get_array_size(large_array)
    
    # Verify sizes (8 bytes per float64 value)
    assert small_size == 800  # 10x10x8
    assert medium_size == 80000  # 100x100x8
    assert large_size == 2000000  # 500x500x8
    
    # Test with non-array
    assert cache_manager._get_array_size("string") == 0

def test_cache_size_tracking(cache_manager, sample_arrays):
    """Test that cache size is correctly tracked when adding arrays"""
    small_array, medium_array, _ = sample_arrays
    
    # Initial size
    assert cache_manager.cache_size == 0
    
    # Add small array
    cache_manager.set(small_array, "category", "small")
    assert cache_manager.cache_size == 800
    
    # Add medium array
    cache_manager.set(medium_array, "category", "medium")
    assert cache_manager.cache_size == 800 + 80000

def test_cache_size_tracking_with_list(cache_manager, sample_arrays):
    """Test cache size tracking with lists of arrays"""
    small_array, medium_array, _ = sample_arrays
    
    # Create a list of arrays
    array_list = [small_array, medium_array]
    
    # Add list to cache
    cache_manager.set(array_list, "category", "list")
    assert cache_manager.cache_size == 800 + 80000

####################EVICTION TESTS####################
def test_eviction_when_full(sample_arrays):
    """Test eviction when cache is full"""
    # Create a very small cache (3 MB)
    cm = CacheManager(max_size_mb=3)
    
    _, medium_array, large_array = sample_arrays
    
    # Add medium array (about 0.08 MB)
    cm.set(medium_array, "category", "medium")
    assert "category" in [k[0] for k in cm.cache.keys()]
    
    # Add large array (about 2 MB) - should trigger eviction
    cm.set(large_array, "category", "large")
    
    # Verify both can fit
    assert cm.get("category", "medium") is not None
    assert cm.get("category", "large") is not None
    
    # Add another large array - should trigger eviction
    cm.set(large_array, "category", "large2")
    
    # The total size would be ~4MB, over the limit
    # One of the entries must have been evicted
    entries_found = 0
    if cm.get("category", "medium") is not None:
        entries_found += 1
    if cm.get("category", "large") is not None:
        entries_found += 1
    if cm.get("category", "large2") is not None:
        entries_found += 1
    
    # At least one entry must have been evicted
    assert entries_found < 3
    
    # The large2 entry should be there (most recently added)
    assert cm.get("category", "large2") is not None

def test_unlimited_cache(sample_arrays):
    """Test behavior with unlimited cache size"""
    cm = CacheManager(max_size_mb=0)  # No limit
    _, medium_array, large_array = sample_arrays
    
    # Add arrays - should not trigger eviction
    for i in range(10):
        cm.set(medium_array, "category", f"medium{i}")
        cm.set(large_array, "category", f"large{i}")
    
    # Verify all entries are still in cache
    for i in range(10):
        assert cm.get("category", f"medium{i}") is not None
        assert cm.get("category", f"large{i}") is not None

####################CLEAR AND STATS TESTS####################
def test_clear(cache_manager, sample_arrays):
    """Test clearing the cache"""
    small_array, medium_array, _ = sample_arrays
    
    # Add some items
    cache_manager.set(small_array, "category", "small")
    cache_manager.set(medium_array, "category", "medium")
    
    # Verify items are in cache
    assert cache_manager.get("category", "small") is not None
    assert cache_manager.get("category", "medium") is not None
    
    # Clear cache
    cache_manager.clear()
    
    # Verify cache is empty
    assert cache_manager.cache == {}
    assert cache_manager.cache_size == 0
    assert cache_manager.get("category", "small") is None
    assert cache_manager.get("category", "medium") is None

def test_stats(cache_manager, sample_arrays):
    """Test the stats function"""
    small_array, medium_array, _ = sample_arrays
    
    # Add some items and perform some gets
    cache_manager.set(small_array, "category", "small")
    cache_manager.set(medium_array, "category", "medium")
    cache_manager.get("category", "small")
    cache_manager.get("category", "medium")
    cache_manager.get("category", "nonexistent")
    
    # Get stats
    stats = cache_manager.stats()
    
    # Verify stats
    assert stats['items'] == 2
    assert stats['size_mb'] == (800 + 80000) / (1024 * 1024)
    assert stats['max_size_mb'] == 10
    assert stats['hit_rate'] == 2/3  # 2 hits, 1 miss

def test_stats_empty_cache(cache_manager):
    """Test stats with empty cache"""
    stats = cache_manager.stats()
    assert stats['items'] == 0
    assert stats['size_mb'] == 0
    assert stats['hit_rate'] == 0

####################EDGE CASES####################
def test_overwrite_same_key(cache_manager, sample_arrays):
    """Test overwriting an existing key"""
    small_array, medium_array, _ = sample_arrays
    
    # Set a value
    cache_manager.set(small_array, "category", "key")
    original_size = cache_manager.cache_size
    
    # Overwrite with a different value
    cache_manager.set(medium_array, "category", "key")
    
    # Verify the value was updated
    result = cache_manager.get("category", "key")
    assert np.array_equal(result, medium_array)
    
    # Verify the size was updated correctly
    assert cache_manager.cache_size == original_size - 800 + 80000

def test_set_non_array_value(cache_manager):
    """Test setting a non-array value"""
    # Set a non-array value
    cache_manager.set("string value", "category", "string")
    
    # Verify the value was stored
    assert cache_manager.get("category", "string") == "string value"
    
    # Verify no size was added
    assert cache_manager.cache_size == 0