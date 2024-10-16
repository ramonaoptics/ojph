import ojph


def test_init():
    ojph.__version__


def test_imports():
    from ojph import imread
    from ojph import imwrite
    from ojph._imread import OJPHImageFile

    # Ensure that this stuff "works"
    imread
    imwrite
    OJPHImageFile
