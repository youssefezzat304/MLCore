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
  np.testing.assert_array_equal(
    tensor.numpy(),
    np.array(3, dtype=np.float32)
  )
  
def test_tensor_numpy_returns_copy():
  tensor = Tensor([1, 2])
  
  array = tensor.numpy()
  array[0] = 2
  
  assert tensor.numpy()[0] == 1
  
def test_add_scalar_tensors():
  tensor1 = Tensor(1)
  tensor2 = Tensor(1)

  result = tensor1 + tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array(2, dtype=np.float32)
  )

def test_add_tensors_with_same_shape():
  tensor1 = Tensor([1, 2, 3])
  tensor2 = Tensor([1, 1, 1])

  result = tensor1 + tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([2, 3, 4], dtype=np.float32)
  )

def test_add_tensors_supports_broadcasting():
  tensor1 = Tensor([1, 2, 3])
  tensor2 = Tensor(1)

  result = tensor1 + tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([2, 3, 4], dtype=np.float32)
  )

def test_subtract_scalar_tensors():
  tensor1 = Tensor(3)
  tensor2 = Tensor(1)

  result = tensor1 - tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array(2, dtype=np.float32)
  )

def test_subtract_tensors_with_same_shape():
  tensor1 = Tensor([3, 4, 5])
  tensor2 = Tensor([1, 1, 1])

  result = tensor1 - tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([2, 3, 4], dtype=np.float32)
  )

def test_subtract_tensors_supports_broadcasting():
  tensor1 = Tensor([3, 4, 5])
  tensor2 = Tensor(1)

  result = tensor1 - tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([2, 3, 4], dtype=np.float32)
  )

def test_multiply_scalar_tensors():
  tensor1 = Tensor(3)
  tensor2 = Tensor(2)

  result = tensor1 * tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array(6, dtype=np.float32)
  )

def test_multiply_tensors_with_same_shape():
  tensor1 = Tensor([1, 2, 3])
  tensor2 = Tensor([2, 2, 2])

  result = tensor1 * tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([2, 4, 6], dtype=np.float32)
  )

def test_multiply_tensors_supports_broadcasting():
  tensor1 = Tensor([1, 2, 3])
  tensor2 = Tensor(2)

  result = tensor1 * tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([2, 4, 6], dtype=np.float32)
  )

def test_divide_scalar_tensors():
  tensor1 = Tensor(6)
  tensor2 = Tensor(2)

  result = tensor1 / tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array(3, dtype=np.float32)
  )

def test_divide_tensors_with_same_shape():
  tensor1 = Tensor([2, 4, 6])
  tensor2 = Tensor([2, 2, 2])

  result = tensor1 / tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([1, 2, 3], dtype=np.float32)
  )

def test_divide_tensors_supports_broadcasting():
  tensor1 = Tensor([2, 4, 6])
  tensor2 = Tensor(2)

  result = tensor1 / tensor2

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(
    result.numpy(),
    np.array([1, 2, 3], dtype=np.float32)
  )
  
@pytest.mark.parametrize(
  "operation",
  [
    lambda tensor1, tensor2: tensor1 + tensor2,
    lambda tensor1, tensor2: tensor1 - tensor2,
    lambda tensor1, tensor2: tensor1 * tensor2,
    lambda tensor1, tensor2: tensor1 / tensor2
  ],
)
def test_binary_operation_rejects_incompatible_shapes(operation):
  tensor1 = Tensor([1, 2, 3])
  tensor2 = Tensor([1, 2])
  
  with pytest.raises(ValueError):
    operation(tensor1, tensor2)
    
@pytest.mark.parametrize(
  "operation",
  [
    lambda tensor: tensor + "hello",
    lambda tensor: tensor - "hello",
    lambda tensor: tensor * "hello",
    lambda tensor: tensor / "hello",
    lambda tensor: tensor + [1, 2, 3],
  ],
)
def test_binary_operations_reject_invalid_operands(operation):
  tensor = Tensor([1, 2, 3])

  with pytest.raises(TypeError):
    operation(tensor)
    
@pytest.mark.parametrize(
  "operation, expected",
  [
    (lambda tensor: tensor + 1, np.array([2, 3, 4], dtype=np.float32)),
    (lambda tensor: tensor - 1, np.array([0, 1, 2], dtype=np.float32)),
    (lambda tensor: tensor * 2, np.array([2, 4, 6], dtype=np.float32)),
    (lambda tensor: tensor / 2, np.array([0.5, 1.0, 1.5], dtype=np.float32)),
  ],
)
def test_binary_operations_support_numeric_scalars(operation, expected):
  tensor = Tensor([1, 2, 3])

  result = operation(tensor)

  assert isinstance(result, Tensor)
  np.testing.assert_array_equal(result.numpy(), expected)