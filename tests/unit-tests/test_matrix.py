import unittest
import numpy as np
import autom8


class TestMatrix(unittest.TestCase):
    def test_invalid_arguments(self):
        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*list'):
            autom8.create_matrix(0)

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*list'):
            autom8.create_matrix('')

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*rows'):
            autom8.create_matrix({})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*rows'):
            autom8.create_matrix({'schema': []})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*rows'):
            autom8.create_matrix({'metadata': None})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*schema.*dict'):
            autom8.create_matrix({'rows': [[1]], 'schema': ['']})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*schema.*name'):
            autom8.create_matrix({'rows': [[1]], 'schema': [{}]})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*schema.*role'):
            autom8.create_matrix({'rows': [[1]], 'schema': [{}]})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*schema.*name'):
            autom8.create_matrix({'rows': [[1]], 'schema': [{'role': None}]})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*schema.*role'):
            autom8.create_matrix({'rows': [[1]], 'schema': [{'name': 'count'}]})

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*list'):
            autom8.create_matrix({
                'rows': 'hi',
                'schema': [{'name': 'msg', 'role': 'textual'}],
            })

        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*role in'):
            autom8.create_matrix({
                'rows': [[-1]],
                'schema': [{'name': 'count', 'role': 'int'}],
            })

    def test_empty_datasets(self):
        for data in [[], (), {'rows': [], 'schema': []}, np.array([])]:
            acc = autom8.Accumulator()
            matrix = autom8.create_matrix(data, observer=acc)
            self.assertEqual(matrix.columns, [])
            self.assertEqual(acc.warnings, [])

    def test_empty_dataset_with_empty_rows(self):
        # Assert that we see one warning when we have three empty rows.
        acc = autom8.Accumulator()
        matrix = autom8.create_matrix([[], [], []], observer=acc)
        self.assertEqual(matrix.columns, [])
        self.assertEqual(len(acc.warnings), 1)

    def test_empty_dataset_warning_message(self):
        a1 = autom8.Accumulator()
        a2 = autom8.Accumulator()
        a3 = autom8.Accumulator()
        autom8.create_matrix([[]], observer=a1)
        autom8.create_matrix([[], []], observer=a2)
        autom8.create_matrix([[], [], []], observer=a3)
        self.assertEqual(a1.warnings, ['Dropped 1 empty row from dataset.'])
        self.assertEqual(a2.warnings, ['Dropped 2 empty rows from dataset.'])
        self.assertEqual(a3.warnings, ['Dropped 3 empty rows from dataset.'])

    def test_extra_columns_warning_message(self):
        a1 = autom8.Accumulator()
        a2 = autom8.Accumulator()
        m1 = autom8.create_matrix([[1, 2], [1, 2, 3]], observer=a1)
        m2 = autom8.create_matrix(
            [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]], observer=a2
        )

        self.assertTrue(len(m1.columns), 2)
        self.assertEqual(a1.warnings, [
            'Dropped 1 extra column from dataset.'
            ' Keeping first 2 columns.'
            ' To avoid this behavior, ensure that each row in the dataset has'
            ' the same number of columns.'
        ])

        self.assertTrue(len(m2.columns), 1)
        self.assertEqual(a2.warnings, [
            'Dropped 3 extra columns from dataset.'
            ' Keeping first 1 column.'
            ' To avoid this behavior, ensure that each row in the dataset has'
            ' the same number of columns.'
        ])

    def test_creating_simple_matrix_with_schema(self):
        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(
            {
                'rows': [['hi', True], ['bye', False]],
                'schema': [
                    {'name': 'msg', 'role': 'textual'},
                    {'name': 'flag', 'role': 'encoded'},
                ],
            },
            observer=acc,
        )

        c1, c2 = matrix.columns
        e1 = np.array(['hi', 'bye'], dtype=object)
        e2 = np.array([True, False], dtype=None)

        self.assertTrue(np.array_equal(c1.values, e1))
        self.assertTrue(np.array_equal(c2.values, e2))

        self.assertEqual(c1.name, 'msg')
        self.assertEqual(c2.name, 'flag')

        self.assertEqual(c1.role, 'textual')
        self.assertEqual(c2.role, 'encoded')

        self.assertEqual(c1.is_original, True)
        self.assertEqual(c2.is_original, True)

        self.assertEqual(acc.warnings, [])

    def test_creating_simple_matrix_from_list(self):
        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(
            [['hi', 1, True], ['bye', 2, False]],
            observer=acc,
        )

        c1, c2, c3 = matrix.columns
        e1 = np.array(['hi', 'bye'], dtype=object)
        e2 = np.array([1, 2], dtype=None)
        e3 = np.array([True, False], dtype=None)

        self.assertTrue(np.array_equal(c1.values, e1))
        self.assertTrue(np.array_equal(c2.values, e2))
        self.assertTrue(np.array_equal(c3.values, e3))

        self.assertEqual(c1.name, 'Column-1')
        self.assertEqual(c2.name, 'Column-2')
        self.assertEqual(c3.name, 'Column-3')

        self.assertEqual(c1.role, None)
        self.assertEqual(c2.role, None)
        self.assertEqual(c3.role, None)

        self.assertEqual(c1.is_original, True)
        self.assertEqual(c2.is_original, True)
        self.assertEqual(c3.is_original, True)

        self.assertEqual(acc.warnings, [])

    def test_creating_simple_matrix_from_numpy_array(self):
        acc = autom8.Accumulator()
        matrix = autom8.create_matrix(
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            observer=acc,
        )

        c1, c2, c3 = matrix.columns
        e1 = np.array([1, 4, 7, 10], dtype=object)
        e2 = np.array([2, 5, 8, 11], dtype=None)
        e3 = np.array([3, 6, 9, 12], dtype=None)

        self.assertTrue(np.array_equal(c1.values, e1))
        self.assertTrue(np.array_equal(c2.values, e2))
        self.assertTrue(np.array_equal(c3.values, e3))

    def test_len_method(self):
        m1 = autom8.create_matrix([
            ['hi', 1, True],
            ['so', 2, True],
            ['bye', 3, False],
        ])
        m2 = autom8.create_matrix([[1], [2], [3], [4], [5], [6], [7]])
        self.assertEqual(len(m1), 3)
        self.assertEqual(len(m2), 7)

    def test_copy_method(self):
        m1 = autom8.create_matrix([
            ['hi', 1.1, True],
            ['so', 2.2, True],
            ['bye', 3.3, False],
        ])
        m2 = autom8.create_matrix([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        n1, n2 = m1.copy(), m2.copy()

        self.assertTrue(m1 is not n1)
        self.assertTrue(m2 is not n2)

        self.assertEqual(len(m1.columns), len(n1.columns))
        self.assertEqual(len(m2.columns), len(n2.columns))

        for a, b in zip(m1.columns + m2.columns, n1.columns + n2.columns):
            self.assertTrue(a is not b)
            self.assertTrue(a.values is not b.values)
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.role, b.role)
            self.assertEqual(a.is_original, b.is_original)
            self.assertTrue(np.array_equal(a.values, b.values))

    def test_schema_property(self):
        schema = [
            {'name': 'A', 'role': 'textual', 'dtype': 'int64'},
            {'name': 'B', 'role': 'encoded', 'dtype': 'int64'},
            {'name': 'C', 'role': None, 'dtype': 'int64'},
        ]
        matrix = autom8.create_matrix({'rows': [[1, 2, 3]], 'schema': schema})
        self.assertEqual(matrix.schema, schema)

    def test_tolist_method(self):
        m1 = autom8.create_matrix({
            'rows': [['hi', True], ['bye', False]],
            'schema': [
                {'name': 'msg', 'role': 'textual'},
                {'name': 'flag', 'role': 'encoded'},
            ],
        })
        m2 = autom8.create_matrix([[1, 2.0], [3, 4.0], [5, 6.0]])

        self.assertEqual(m1.tolist(), [
            ['msg', 'flag'],
            ['hi', True],
            ['bye', False],
        ])

        self.assertEqual(m2.tolist(), [
            ['Column-1', 'Column-2'],
            [1, 2.0],
            [3, 4.0],
            [5, 6.0],
        ])

    def test_to_array_method(self):
        m1 = autom8.create_matrix([[1], [2], [3], [4]])
        m2 = autom8.create_matrix([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(m1.to_array(), np.array([1, 2, 3, 4])))
        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*one column'):
            m2.to_array()

    def test_append_column(self):
        matrix = autom8.create_matrix([[1], [2], [3], [4]])
        matrix.append_column(np.array([2, 4, 6, 8]), 'foo', 'encoded')
        c1, c2 = matrix.columns
        self.assertEqual(c2.name, 'foo')
        self.assertEqual(c2.role, 'encoded')
        self.assertEqual(c2.is_original, False)
        self.assertTrue(np.array_equal(c2.values, np.array([2, 4, 6, 8])))
        self.assertFalse(np.array_equal(c2.values, np.array([1, 2, 3, 4])))

    def test_drop_columns_by_index(self):
        m1 = autom8.create_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m2 = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        m1.drop_columns_by_index([0, 2])
        m2.drop_columns_by_index([1, 2])
        self.assertEqual(len(m1.columns), 1)
        self.assertEqual(len(m2.columns), 2)
        self.assertTrue(np.array_equal(m1.columns[0].values, np.array([2, 5, 8])))
        self.assertEqual(m1.tolist(), [['Column-2'], [2], [5], [8]])
        self.assertEqual(m2.tolist(), [
            ['Column-1', 'Column-4'], [1, 4], [5, 8], [9, 12]
        ])

    def test_select_rows(self):
        mat = autom8.create_matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        m1 = mat.select_rows([0, 2, 4])
        m2 = mat.select_rows([1, 3])
        head = ['Column-1', 'Column-2']
        self.assertEqual(m1.tolist(), [head, [1, 2], [5, 6], [9, 10]])
        self.assertEqual(m2.tolist(), [head, [3, 4], [7, 8]])

    def test_exclude_rows(self):
        mat = autom8.create_matrix([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        m1 = mat.exclude_rows([0, 2, 4])
        m2 = mat.exclude_rows([1, 3])
        head = ['Column-1', 'Column-2']
        self.assertEqual(m1.tolist(), [head, [3, 4], [7, 8]])
        self.assertEqual(m2.tolist(), [head, [1, 2], [5, 6], [9, 10]])

    def test_select_columns(self):
        mat = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        m1 = mat.select_columns([0, 2, 3])
        m2 = mat.select_columns([1])
        self.assertEqual(m1.tolist()[1:], [[1, 3, 4], [5, 7, 8], [9, 11, 12]])
        self.assertEqual(m2.tolist()[1:], [[2], [6], [10]])

    def test_exclude_columns(self):
        mat = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        m1 = mat.exclude_columns([0, 2, 3])
        m2 = mat.exclude_columns([1])
        self.assertEqual(m1.tolist()[1:], [[2], [6], [10]])
        self.assertEqual(m2.tolist()[1:], [[1, 3, 4], [5, 7, 8], [9, 11, 12]])

    def test_column_indices_where(self):
        matrix = autom8.create_matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        pred = lambda x: x.name == 'Column-2' or x.name == 'Column-4'
        indices = matrix.column_indices_where(pred)
        self.assertEqual(indices, [1, 3])


class TestColumn(unittest.TestCase):
    def test_len_method(self):
        matrix = autom8.create_matrix([[1, 2], [3, 4], [5, 6], [7, 8]])
        c1, c2 = matrix.columns
        self.assertEqual(len(c1), 4)
        self.assertEqual(len(c2), 4)

    def test_column_dtype_property(self):
        matrix = autom8.create_matrix([
            ['hi', 10, 1.1, True, None],
            ['so', 20, 2.2, True, None],
            ['bye', 30, 3.3, False, None],
        ])
        c1, c2, c3, c4, c5 = matrix.columns
        self.assertEqual(c1.dtype, np.dtype('O'))
        self.assertEqual(c2.dtype, np.dtype('int64'))
        self.assertEqual(c3.dtype, np.dtype('float64'))
        self.assertEqual(c4.dtype, np.dtype('bool'))
        self.assertEqual(c5.dtype, np.dtype('O'))

    def test_properties_of_valid_roles(self):
        matrix = autom8.create_matrix([
            [1, 2, 3, 4, 'a'],
            [2, 3, 4, 5, 'b'],
            [3, 4, 5, 6, 'c'],
        ])
        c1, c2, c3, c4, c5 = matrix.columns
        c1.role = None
        c2.role = 'categorical'
        c3.role = 'encoded'
        c4.role = 'numerical'
        c5.role = 'textual'

        self.assertFalse(c1.is_numerical)
        self.assertFalse(c2.is_numerical)
        self.assertFalse(c3.is_numerical)
        self.assertTrue(c4.is_numerical)
        self.assertFalse(c5.is_numerical)

        self.assertIsNone(c1.role)
        self.assertEqual(c2.role, 'categorical')
        self.assertEqual(c3.role, 'encoded')
        self.assertEqual(c4.role, 'numerical')
        self.assertEqual(c5.role, 'textual')

    def test_setting_an_invalid_role(self):
        matrix = autom8.create_matrix([[1], [2], [3]])
        col = matrix.columns[0]
        with self.assertRaisesRegex(autom8.Autom8Exception, 'Expected.*role in'):
            col.role = 'foo'
