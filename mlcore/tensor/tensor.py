from __future__ import annotations
from collections.abc import Callable

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
    
  def __repr__(self):
    return f"""
            Tensor(data={self._data},
            dtype={self.dtype})
            """
  
  def __add__(self, other: Tensor | int | float) -> Tensor:
    return self._apply_binary_operation(other, np.add, "addition")

  def __sub__(self, other: Tensor | int | float) -> Tensor:
    return self._apply_binary_operation(other, np.subtract, "subtraction")

  def __mul__(self, other: Tensor | int | float) -> Tensor:
    return self._apply_binary_operation(other, np.multiply, "multiplication")

  def __truediv__(self, other: Tensor | int | float) -> Tensor:
    return self._apply_binary_operation(other, np.divide, "division")
  
  def __radd__(self, other: int | float) -> Tensor:
    return self.__add__(other)

  def __rmul__(self, other: int | float) -> Tensor:
    return self.__mul__(other)
  
  def __rsub__(self, other):
    other_data = self._to_operand_array(other)
    return Tensor(np.subtract(other_data, self._data))
  
  def __rtruediv__(self, other):
    other_data = self._to_operand_array(other)
    return Tensor(np.divide(other_data, self._data))
  
  def __neg__(self) -> Tensor:
    return Tensor(-self._data)
  
  def __pos__(self) -> Tensor:
    return Tensor(self._data)
  
  def __eq__(self, value) -> bool:
    return (
    isinstance(value, Tensor)
    and self.shape == value.shape
    and self.dtype == value.dtype
    and np.array_equal(self._data, value._data)
  )
  
  def __ne__(self, value) -> bool:
    return not self.__eq__(value)
  
  def _apply_binary_operation(
    self,
    other: Tensor | int | float,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    operation_name: str
  ) -> Tensor:
    """Apply a NumPy binary operation between a tensor and another operand."""

    other_data = self._to_operand_array(other)

    try:
      result = operation(self._data, other_data)
    except ValueError as exc:
      raise ValueError(
        f"Tensor shapes are not compatible for {operation_name}: "
        f"{self.shape} and {other_data.shape}."
      ) from exc

    return Tensor(result)
  
  def _to_operand_array(self, other: Tensor | int | float) -> np.ndarray:
    if isinstance(other, Tensor):
      return other._data
    
    if isinstance(other, (int, float)):
      return np.array(other, dtype=self.dtype)
    
    raise TypeError(
      "Binary operations are only supported between Tensor objects and numeric scalars."
    )
  
  def numpy(self) -> np.ndarray:
    """Return a copy of the tensor data as a NumPy array."""
    return self._data.copy()
  
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