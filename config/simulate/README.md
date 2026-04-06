# config

## example

```python
import numpy as np

config = {
    'eta': 1,  # coupiling loss
    'alpha': 52.96,  # propagation loss coefficient
    'K': np.array([0.5046040246268209, 0.15363162694655808, 0.12089146531089001, 0.151544046298763, 0.5133452680909094]),  # coupling rate
    'L': np.array([0.00018269, 0.00018269, 0.00018269, 0.00018269]),  # ring length
    'n': 3.3938,  # refractive index
    'center_wavelength': 1550e-9
}
```



