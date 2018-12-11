import os.path

from autom8.formats import (
    decode_csv,
    excel_column_index,
    excel_column_name,
    read_csv,
)


def test_boston():
    boston = _read_csv('boston.csv')
    assert len(boston) == 507

    head = 'CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV'
    assert boston[0] == head.split(',')
    assert boston[1] == [
        0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2,
        4.09, 1.0, 296.0, 15.3, 396.9, 4.98, 24.0,
    ]
    assert boston[-1] == [
        0.04741, 0.0, 11.93, 0.0, 0.573, 6.03, 80.8,
        2.505, 1.0, 273.0, 21.0, 396.9, 7.88, 11.9,
    ]


def test_iris():
    iris = _read_csv('iris.csv')
    assert len(iris) == 151

    head = 'sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),class'
    assert iris[0] == head.split(',')
    assert iris[1] == [5.1, 3.5, 1.4, 0.2, 'setosa']
    assert iris[-1] == [5.9, 3.0, 5.1, 1.8, 'virginica']


def test_wine():
    wine = _read_csv('wine.csv')
    assert len(wine) == 179

    head = ('alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,'
        'total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,'
        'color_intensity,hue,od280/od315_of_diluted_wines,proline,class')
    assert wine[0] == head.split(',')
    assert wine[1] == [
        14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28,
        2.29, 5.64, 1.04, 3.92, 1065.0, 'class_0',
    ]
    assert wine[-1] == [
        14.13, 4.1, 2.74, 24.5, 96.0, 2.05, 0.76, 0.56,
        1.35, 9.2, 0.61, 1.6, 560.0, 'class_2',
    ]


def test_decode_csv_with_hex_numbers():
    dataset = decode_csv(
        'foo,bar,baz\n'
        '1,2.2,0xfa\n'
        '3.3,4,0xfb\n'
    )
    assert dataset == [
        ['foo', 'bar', 'baz'],
        [1, 2.2, 0xfa],
        [3.3, 4, 0xfb],
    ]

    assert isinstance(dataset[0][0], str)
    assert isinstance(dataset[0][1], str)
    assert isinstance(dataset[0][2], str)

    assert isinstance(dataset[1][0], int)
    assert isinstance(dataset[1][1], float)
    assert isinstance(dataset[1][2], int)

    assert isinstance(dataset[2][0], float)
    assert isinstance(dataset[2][1], int)
    assert isinstance(dataset[2][2], int)


def _read_csv(name):
    testdir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(testdir, 'datasets', name)
    return read_csv(path)


def test_excel_column_name():
    assert excel_column_name(0) == 'A'
    assert excel_column_name(1) == 'B'
    assert excel_column_name(2) == 'C'
    assert excel_column_name(25-2) == 'X'
    assert excel_column_name(25-1) == 'Y'
    assert excel_column_name(25-0) == 'Z'

    for index, name in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        assert excel_column_name(index) == name

    assert excel_column_name(26+0) == 'AA'
    assert excel_column_name(26+1) == 'AB'
    assert excel_column_name(26+2) == 'AC'
    assert excel_column_name(26+25-2) == 'AX'
    assert excel_column_name(26+25-1) == 'AY'
    assert excel_column_name(26+25-0) == 'AZ'

    for index, name in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        assert excel_column_name(26+index) == 'A' + name

    assert excel_column_name(26*8+0) == 'HA'
    assert excel_column_name(26*8+1) == 'HB'
    assert excel_column_name(26*8+2) == 'HC'
    assert excel_column_name(26*8+25-2) == 'HX'
    assert excel_column_name(26*8+25-1) == 'HY'
    assert excel_column_name(26*8+25-0) == 'HZ'

    for index, name in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        assert excel_column_name(26*8+index) == 'H' + name


def test_excel_column_index():
    assert excel_column_index('A') == 0
    assert excel_column_index('B') == 1
    assert excel_column_index('C') == 2
    assert excel_column_index('X') == 25-2
    assert excel_column_index('Y') == 25-1
    assert excel_column_index('Z') == 25-0

    for index, name in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        assert excel_column_index(name) == index

    assert excel_column_index('AA') == 26+0
    assert excel_column_index('AB') == 26+1
    assert excel_column_index('AC') == 26+2
    assert excel_column_index('AX') == 26+25-2
    assert excel_column_index('AY') == 26+25-1
    assert excel_column_index('AZ') == 26+25-0

    for index, name in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        assert excel_column_index('A' + name) == 26+index

    assert excel_column_index('HA') == 26*8+0
    assert excel_column_index('HB') == 26*8+1
    assert excel_column_index('HC') == 26*8+2
    assert excel_column_index('HX') == 26*8+25-2
    assert excel_column_index('HY') == 26*8+25-1
    assert excel_column_index('HZ') == 26*8+25-0

    for index, name in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        assert excel_column_index('H' + name) == 26*8+index


def test_excel_column_conversions():
    for i in range(2000):
        assert excel_column_index(excel_column_name(i)) == i

    for s in ['FOO', 'BAR', 'BAZ', 'FIZ', 'BUZ', 'ZIM', 'ZAM', 'BIM', 'BAM']:
        assert excel_column_name(excel_column_index(s)) == s
