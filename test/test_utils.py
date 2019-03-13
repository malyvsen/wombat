import numpy as np
from wombat.utils import smart_call


def test_smart_call():
    assert smart_call([1, 2, lambda: 3]) == [1, 2, 3]
    assert smart_call((lambda x: x * 3, lambda y: y + 2), x=5, y=6) == (15, 8)
    np_test = np.array([[5, 4], [3, 2]])
    assert np.all(smart_call(np_test) == np_test)
