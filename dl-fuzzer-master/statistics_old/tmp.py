
import numpy as np
import typing
from numpy import array as nparray
import numpy.typing as npt

def sum(a: int,b: typing.Union[int, float]) -> npt.array:
    return nparray([a,b])
    # b=Annotated[int, ValueRange(0,5)]
    # return a+b

print(sum(1,2))