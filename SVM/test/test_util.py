from util import Util
import unittest
class TestUtil(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sigmoid(self):
        self.assertEqual(Util.sigmoid(0), 0.5)
