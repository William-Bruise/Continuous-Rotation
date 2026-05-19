__all__ = ['DIV2KASISRDataset', 'BenchmarkSRDataset']

def __getattr__(name):
    if name == 'DIV2KASISRDataset':
        from .div2k_asisr import DIV2KASISRDataset
        return DIV2KASISRDataset
    if name == 'BenchmarkSRDataset':
        from .benchmark_sr import BenchmarkSRDataset
        return BenchmarkSRDataset
    raise AttributeError(name)
