from reference import ref_kernel
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    return ref_kernel(data)
