# Cython ops package initialization
try:
    from . import hive_ops
except ImportError:
    import hive_ops
