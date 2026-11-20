import scalping


def test_package_imports_and_has_version():
    assert isinstance(scalping.__version__, str)
    assert scalping.__version__.count(".") == 2
