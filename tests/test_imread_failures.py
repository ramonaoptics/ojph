from ojph import imread
import pytest


def test_imread_file_does_not_exist(tmp_path):
    # A double free can cause a segmentation fault. So make sure just an error
    # is raised and not a full interpreter shutdown.
    with pytest.raises(RuntimeError):
        imread(tmp_path / 'does_not_exist.j2c')
