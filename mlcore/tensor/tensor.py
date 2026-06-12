from __future__ import annotations
from typing import Any

import numpy as np
import numpy.typing as npt


class Tensor:
  def __init__(self, data: Any, dtype: npt.DTypeLike = np.float32) -> None:
    """A NumPy-backed tensor object used as the core data structure in MLCore.

    Parameters
    ----------
    data:
        Input data used to create the tensor. This can be a scalar, list,
        nested list, tuple, or NumPy array, as long as it can be converted
        to numeric tensor data.
    dtype:
        NumPy dtype used for internal storage. Defaults to np.float32.
    """
    self._data = self._validate_and_convert(data, dtype)
  
  @staticmethod 
  def _validate_and_convert(data: Any, dtype: npt.DTypeLike) -> np.ndarray:
    """Convert input data into a validated numeric NumPy array.

    Raises
    ------
    TypeError
        If the input data cannot be converted to a numeric NumPy array.
    """
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
  
  @property
  def shape(self) -> tuple[int, ...]:
    """Shape of the tensor."""
    return self._data.shape
  
  @property
  def dtype(self) -> np.dtype:
    """Data type of the tensor elements."""
    return self._data.dtype
  
  @property
  def ndim(self) -> int:
    """Number of tensor dimensions."""
    return self._data.ndim
  
  @property
  def size(self) -> int:
    """Total number of elements in the tensor."""
    return self._data.size
  
  def __repr__(self):
    return f"Tensor(data={self._data}, dtype={self.dtype})"