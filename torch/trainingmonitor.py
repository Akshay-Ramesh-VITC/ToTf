import torch
import psutil
import time
from tqdm.auto import tqdm
from typing import Dict, Optional

class TrainingMonitor:
    def __init__(self, total_steps: int, desc: str = "Training"):
        self.total_steps = total_steps
        self.pbar = tqdm(total=total_steps, desc=desc, dynamic_ncols=True)
        self.start_time = time.time()

    def _get_resources(self) -> Dict[str, str]:
        """Fetches current RAM and VRAM usage."""
        resources = {}
        
        # System RAM
        ram_gb = psutil.virtual_memory().used / (1024**3)
        resources["RAM"] = f"{ram_gb:.1f}GB"
        
        # GPU VRAM (if available)
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_reserved() / (1024**3)
            resources["VRAM"] = f"{vram_gb:.1f}GB"
            
        return resources

    def update(self, step: int, metrics: Dict[str, float], prefix: str = ""):
        """
        Updates the progress bar with metrics and resource usage.
        """
        # Combine metrics with resource tracking
        stats = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}
        stats.update(self._get_resources())
        
        # Update tqdm description and postfix
        if prefix:
            self.pbar.set_description(f"{prefix}")
        
        self.pbar.set_postfix(stats)
        self.pbar.update(step - self.pbar.n)

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()