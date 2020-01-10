import unittest
import tensorflow as tf


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_version(self):
        print(tf.version())


if __name__ == '__main__':
    unittest.main()
