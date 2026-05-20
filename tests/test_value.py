from autograd import Value


def test_value_can_be_instantiated():
    v = Value(5.0)
    assert isinstance(v, Value)
    assert v.data == 5.0
