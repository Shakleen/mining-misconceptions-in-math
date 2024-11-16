import torch


def get_dtype_size(dtype: torch.dtype) -> int:
    """
    Get size in bytes for a given torch dtype.

    Args:
        dtype: PyTorch dtype (e.g., torch.int32, torch.float32, torch.bool)

    Returns:
        Size in bytes for the dtype
    """
    dtype_sizes = {
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.long: 8,
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.bool: 1,
    }

    return dtype_sizes.get(dtype, 4)  # Default to 4 bytes if unknown


def calculate_tensor_memory(shape: tuple, dtype: torch.dtype) -> int:
    """
    Calculate memory usage in bytes for a tensor with given shape and dtype.

    Args:
        shape: Tuple representing tensor dimensions
        dtype: PyTorch dtype

    Returns:
        Memory usage in bytes
    """
    num_elements = 1
    for dim in shape:
        num_elements *= dim

    return num_elements * get_dtype_size(dtype)


def calculate_dataset_row_memory(tensors_info: dict) -> dict:
    """
    Calculate memory usage for a dataset row given tensor information.

    Args:
        tensors_info: Dictionary with keys as tensor names and values as tuples of (shape, dtype)
        Example:
        {
            'question_ids': ((1, 512), torch.int32),
            'question_mask': ((1, 512), torch.bool),
            'misconception_ids': ((1, 10, 64), torch.int32),
            'misconception_mask': ((1, 10, 64), torch.bool),
            'label': ((1,), torch.int32)
        }

    Returns:
        Dictionary containing:
        - memory usage per tensor
        - total memory usage
        - memory usage in human readable format
    """
    memory_usage = {}
    total_bytes = 0

    for tensor_name, (shape, dtype) in tensors_info.items():
        bytes_used = calculate_tensor_memory(shape, dtype)
        memory_usage[tensor_name] = bytes_used
        total_bytes += bytes_used

    # Convert to human readable format
    units = ["B", "KB", "MB", "GB", "TB"]
    human_readable = total_bytes
    unit_index = 0

    while human_readable >= 1024 and unit_index < len(units) - 1:
        human_readable /= 1024
        unit_index += 1

    return {
        "per_tensor": memory_usage,
        "total_bytes": total_bytes,
        "human_readable": f"{human_readable:.2f} {units[unit_index]}",
    }


# Example usage:
if __name__ == "__main__":
    dataset_info = {
        "question_ids": ((1, 512), torch.int32),
        "question_mask": ((1, 512), torch.bool),
        "misconception_ids": ((1, 10, 64), torch.int32),
        "misconception_mask": ((1, 10, 64), torch.bool),
        "label": ((1,), torch.int32),
    }

    memory_info = calculate_dataset_row_memory(dataset_info)

    print("\nMemory usage per tensor:")
    for tensor_name, bytes_used in memory_info["per_tensor"].items():
        print(f"{tensor_name}: {bytes_used:,} bytes")

    print(f"\nTotal memory per row: {memory_info['total_bytes']:,} bytes")
    print(f"Human readable: {memory_info['human_readable']}")
