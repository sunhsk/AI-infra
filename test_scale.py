# test_scale.py
import sys

sys.path.insert(0, "build")
import fast_scale

actual = fast_scale.scale([1.0, -2.0, 3.5], 2.0)
expected = [2.0, -4.0, 7.0]
assert actual == expected
print(actual)
