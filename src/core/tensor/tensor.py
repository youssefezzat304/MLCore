from __future__ import annotations
import numpy as np

class Tensor:
  def __init__(self, data, dtype: np.dtype=np.float32) -> None:
    self._data = self._validate_and_convert(data, dtype)
  
  @staticmethod 
  def _validate_and_convert(data, dtype) -> np.ndarray:
    try:
      array = np.array(data, dtype)
    except (TypeError, ValueError) as exc:
      raise TypeError(
        "Tensor data must be convertible to a numeric NumPy array."
      ) from exc
    
    if not np.issubdtype(array.dtype, np.number):
      raise TypeError(
        "Tensor data must be numeric."
      )
    
    return array