import unittest
from MRR.Simulator import mymath


class Test(unittest.TestCase):
    def test_is_zero(self):
        test_case = [
            (1, 0.01, True),
            (0.1, 0.01, True),
            (0.01, 0.01, True)
        ]

        for x, y, expect in test_case:
            with self.subTest(x=x, y=y):
                self.assertEqual(mymath.is_zero(x, y), expect)

    def test_mod(self):
        test_case = [
            (3, 2, 1),
            (4, 2, 0),
            (4.5, 1.5, 0),
            (3.9, 2.6, 1.3),
            (2.5, 1.2, 0.1)
        ]

        for x, y, expect in test_case:
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(mymath.mod(x, y), expect, places=7)

    def test_gcd(self):
        test_case = [
            (3, 2, 1),
            (4, 2, 2),
            (4.5, 1.5, 1.5),
            (3.9, 2.6, 1.3)
        ]

        for x, y, expect in test_case:
            with self.subTest(x=x, y=y):
                self.assertAlmostEqual(mymath._gcd(x, y), expect, places=7)

    def test_lcm(self):
        test_case = [
            ([2, 3], 6, '整数のlcm 1'),
            ([2, 4], 4, '整数のlcm 2'),
            ([1.5, 4.5], 4.5, '実数のlcm 1'),
            ([3.9, 2.6], 7.8, '実数のlcm 2'),
            ([3.91e-9, 2.65e-9], 1.04e-06, '再帰回数の制限')
        ]

        for xs, expect, msg in test_case:
            with self.subTest(xs=xs):
                self.assertAlmostEqual(mymath.lcm(xs), expect, places=7, msg=msg)

if __name__ == '__main__':
    unittest.main()
