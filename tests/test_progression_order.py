import numpy as np
import pytest

from ojph import imwrite_to_memory
from ojph._imread import OJPHImageFile


@pytest.mark.parametrize("progression", ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"])
def test_imwrite_to_memory_sets_progression_order(progression):
    image = np.random.randint(0, 256, (32, 48), dtype=np.uint8)
    compressed = imwrite_to_memory(image, progression_order=progression)
    file_obj = OJPHImageFile.from_memory(np.asarray(compressed, dtype=np.uint8))
    assert file_obj.progression_order == progression
