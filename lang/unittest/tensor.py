import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import tensorflow as tf
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
