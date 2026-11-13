"""GPU/CPU selection and VRAM usage diagnostics; optional PyTorch."""
def get_device(preferred='auto'):
    import torch
    if preferred not in ('auto','cpu','cuda'):
        raise ValueError('preferred must be auto, cpu or cuda')
    if preferred == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    return torch.device('cuda' if (preferred == 'cuda' or (preferred == 'auto' and torch.cuda.is_available())) else 'cpu')


def cuda_memory_stats():
    import torch
    if not torch.cuda.is_available():
        return {'available': False, 'allocated_mib': 0.0, 'reserved_mib': 0.0, 'peak_mib': 0.0}
    return {'available': True,
            'allocated_mib': torch.cuda.memory_allocated()/2**20,
            'reserved_mib': torch.cuda.memory_reserved()/2**20,
            'peak_mib': torch.cuda.max_memory_allocated()/2**20}
