import unittest
import os.path
import autom8


class TestCsvUtils(unittest.TestCase):
    def test_boston(self):
        boston = _load_csv('boston.csv')
        self.assertEqual(len(boston), 507)

        head = 'CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV'
        self.assertEqual(boston[0], head.split(','))
        self.assertEqual(boston[1], [
            0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2,
            4.09, 1.0, 296.0, 15.3, 396.9, 4.98, 24.0,
        ])
        self.assertEqual(boston[-1], [
            0.04741, 0.0, 11.93, 0.0, 0.573, 6.03, 80.8,
            2.505, 1.0, 273.0, 21.0, 396.9, 7.88, 11.9,
        ])

    def test_iris(self):
        iris = _load_csv('iris.csv')
        self.assertEqual(len(iris), 151)

        head = 'sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),class'
        self.assertEqual(iris[0], head.split(','))
        self.assertEqual(iris[1], [5.1, 3.5, 1.4, 0.2, 'setosa'])
        self.assertEqual(iris[-1], [5.9, 3.0, 5.1, 1.8, 'virginica'])

    def test_wine(self):
        wine = _load_csv('wine.csv')
        self.assertEqual(len(wine), 179)

        head = ('alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,'
            'total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,'
            'color_intensity,hue,od280/od315_of_diluted_wines,proline,class')
        self.assertEqual(wine[0], head.split(','))
        self.assertEqual(wine[1], [
            14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28,
            2.29, 5.64, 1.04, 3.92, 1065.0, 'class_0',
        ])
        self.assertEqual(wine[-1], [
            14.13, 4.1, 2.74, 24.5, 96.0, 2.05, 0.76, 0.56,
            1.35, 9.2, 0.61, 1.6, 560.0, 'class_2',
        ])

    def test_decode_csv_with_hex_numbers(self):
        dataset = autom8.decode_csv(
            'foo,bar,baz\n'
            '1,2.2,0xfa\n'
            '3.3,4,0xfb\n'
        )
        self.assertEqual(dataset, [
            ['foo', 'bar', 'baz'],
            [1, 2.2, 0xfa],
            [3.3, 4, 0xfb],
        ])

        self.assertIsInstance(dataset[0][0], str)
        self.assertIsInstance(dataset[0][1], str)
        self.assertIsInstance(dataset[0][2], str)

        self.assertIsInstance(dataset[1][0], int)
        self.assertIsInstance(dataset[1][1], float)
        self.assertIsInstance(dataset[1][2], int)

        self.assertIsInstance(dataset[2][0], float)
        self.assertIsInstance(dataset[2][1], int)
        self.assertIsInstance(dataset[2][2], int)


def _load_csv(name):
    testdir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(testdir, 'datasets', name)
    return autom8.load_csv(path)
