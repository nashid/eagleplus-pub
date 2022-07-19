"""
constants defined for the fuzzer
"""
from enum import Enum, auto

# threshold on how many permutations fuzzer wants to do
# if there will be more, will try to reduce the permutations
MAX_PERMUTE = 10000

# generated Tensor cannot be more than 5D
MAX_NUM_DIM = 5

# max length for each dimension
MAX_DIM = 20

# default values if user doesn't specify --max_iter or --max_time
DEFAULT_MAX_ITER = 10000
DEFAULT_MAX_TIME = 12000


# enum class for the fuzzer to signal the testing status
class Status(Enum):
    INIT = auto()
    PASS = auto()
    FAIL = auto()
    TIMEOUT = auto()
    SIGNAL = auto()
