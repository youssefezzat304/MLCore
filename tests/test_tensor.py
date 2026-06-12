import numpy as np
import pytest

from mlcore import Tensor

def test_tensor_created_from_scalar():
  tensor = Tensor(3)
  
  assert tensor.shape == ()
  assert tensor.ndim == 0
  assert tensor.size == 1
  assert tensor.dtype == np.float32
  
def test_tensor_created_from_list():
  tensor = Tensor([1, 2, 3])
  
  assert tensor.shape == (3, )
  assert tensor.ndim == 1
  assert tensor.size == 3
  assert tensor.dtype == np.float32
  
def test_tensor_created_from_nested_list():
  tensor = Tensor([[1, 2, 3], [4, 5, 6]])
  
  assert tensor.shape == (2, 3)
  assert tensor.ndim == 2
  assert tensor.size == 6
  assert tensor.dtype == np.float32
  
def test_tensor_created_from_numpy_array():
  tensor = Tensor(np.array([1, 2, 3]))
  
  assert tensor.shape == (3, )
  assert tensor.ndim == 1
  assert tensor.size == 3
  assert tensor.dtype == np.float32
  
def test_tensor_rejects_string():
  with pytest.raises(TypeError):
    Tensor("Hello")
    
def test_tensor_rejects_mixed_input():
  with pytest.raises(TypeError):
    Tensor([1, "Hello", 3])
    
def test_tensor_repr_contains_tensor_information():
  tensor = Tensor([1, 2, 3])

  result = repr(tensor)

  assert "Tensor" in result
  assert "dtype=float32" in result
  
def test_tensor_numpy_return_numpy_array():
  tensor = Tensor(3)
  
  assert isinstance(tensor.numpy(), np.ndarray)
  assert tensor._data == 3
  
def test_tensor_numpy_returns_copy():
  tensor = Tensor([1, 2])
  
  array = tensor.numpy()
  array[0] = 2
  
  assert tensor.numpy()[0] == 1
  