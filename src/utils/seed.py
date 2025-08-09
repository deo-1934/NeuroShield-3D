# English: Utilities for reproducibility in experiments.
# فارسی: ابزارهای تکرارپذیری در آزمایش‌ها.

import os
import random
import numpy as np

def set_seed(seed: int = 42, deterministic: bool = False):
    """
    English:
        Sets the random seed for Python, NumPy, and PyTorch (if installed) to ensure reproducible results.
        Optionally enforces deterministic operations in PyTorch for exact reproducibility.
    
    فارسی:
        سید تصادفی را برای پایتون، نامپای و پای‌تورچ (در صورت نصب) تنظیم می‌کند تا نتایج تکرارپذیر باشند.
        در صورت نیاز، عملیات پای‌تورچ را به حالت دترمینیستیک می‌برد تا نتایج دقیقاً قابل بازتولید باشند.
    """
    # Python's built-in random
    # تولید تصادفی داخلی پایتون
    random.seed(seed)

    # NumPy
    # تنظیم سید نامپای
    np.random.seed(seed)

    try:
        import torch
        
        # PyTorch CPU seed
        # تنظیم سید برای CPU در پای‌تورچ
        torch.manual_seed(seed)
        
        # PyTorch GPU seed (if available)
        # تنظیم سید برای GPU در پای‌تورچ (در صورت وجود)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            # Force deterministic algorithms in cuDNN
            # مجبور کردن الگوریتم‌های cuDNN به حالت دترمینیستیک
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    except ImportError:
        # PyTorch not installed
        # اگر پای‌تورچ نصب نباشد
        pass
