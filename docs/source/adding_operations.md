# Adding New Operations in the Library

This document outlines the process for integrating a new operation into the existing code library. The integration process involves several key steps: defining the operation's interface, implementing the operation ensuring compatibility with the library's architecture, and providing testing to validate the operation.

An example of implementing new operations can be found here: [Implementing reduce operations](https://github.com/intel/intel-npu-acceleration-library/commit/4f17015a75c146fe8d569ac71a2e2a0a960fc652)

## Step 1: Defining the OpenVINO interface

The first step is defining the call to the OpenVino method of the new operation through the OpenVINO Runtime C++ API. This is done in the `nn_factory.h` header. In this file, a new operation is created by interfacing with the OpenVINO operation. This includes specifying input and output parameters, and data types of the operation's interface and then calling and returning the OpenVINO method. The interface should align with the library's existing design patterns and naming conventions.

A simple example of defining a new operation:
```
ov::op::Op* new_operation(ov::op::Op* input) {
    auto new_operation = std::make_shared<ov::opset1::NewOp>(input->output(0));
    operations.push_back(new_operation);
    return new_operation.get();
}
```
## Step 2: Defining the C++ bindings

The next step is defining the C++ binding in the `binding.cpp` source file. This is the method that will be called in Python. This method has the operation's input node as a parameter and additional arguments of the operation are defined in the method.

An example of defining the binding:
```
intel_npu_acceleration_library_DLL_API ov::op::Op* new_operation(intel_npu_acceleration_library::ModelFactory* factory, ov::op::Op* input) {
    return factory->new_operation(input);
}
```

## Step 3: Adding new operation to list of supported operation

The new operation is added to the list of supported NPU operations in the `ops.py` script.
The information of the new operation that must be provided is:
- the operation name
- the number of inputs
- the optional parameters types

## Step 4: Adding extra functionality to the operation's function
Ctypes is used to interface between C++ and Python. (Documentation is found here: [Python Ctypes](https://docs.python.org/3/library/ctypes.html))

If there is additional logic that you may want to add to the function, this can be done by defining a Python function that calls the C++ method in the `factory.py` file.
Otherwise, if you directly call the functions to C++, then you do not need to define a Python function.

## Step 5: Adding PyTorch wrapper for the new operation
Additionally, to define a wrapper to use PyTorch native functions, this can be implemented in the `functional.py` file. In this step, a function of the same name as the PyTorch equivalent is created, which is used instead of the PyTorch implementation of the operation.
If there is additional logic that you may want to add to the function to interface with the new operation, it can also be added in this function.

It is common for the new operation to have the same name as the PyTorch equivalent operation, however this is not always the case and to show which operation we are referring to, we refer to the newly implemented operation as `new_operation` and the PyTorch operation and `operation`.

The basic structure of PyTorch wrapper for a PyTorch operation, referred to as `torch.operation`, which returns the output of the implemented `new_operation`:
```
@implements(torch.operation)
def operation(x: Tensor) -> Tensor:
    """Return the output tensor of the operation.

    Args:
        x (Tensor): The input tensor.
    Returns:
        Tensor: Output tensor.
    """
    return generate_op(x, "new_operation")
```
## Step 6: Building the library
To update the library, run the command:
```
pip install .
```

## Step 7: Adding tests for the new operation
A test for the new operation can be added in the `test_op.py` script. The new operation should be compared with a reference to ensure correct implementation.

The following is a basic structure to use the new operation:
```
X = torch.rand((16, 128)).to(torch.float16)  # defining the input tensor

model = NNFactory()
input = model.parameter(X.shape)             # creating the input node
_ = model.new_operation(input)               # _ = torch.operation(input) is equivalent if using the PyTorch wrapper
model.compile()
out = model.run(X.numpy())
```

Using pytest to run all of the tests in the file:
```
pytest <name of the file>
```

Using pytest to run a single test in the file:
```
pytest <name of the file>::<name of the test>
```
