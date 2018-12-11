"""Utility module that allows multiple functions to share documentation."""

from string import Template

_render = lambda docstring: _strip(Template(docstring).substitute(globals()))
_strip = lambda section: section.strip() + '\n'


dataset_parameters = _strip("""
        dataset (list or tuple or numpy.ndarray or autom8.Matrix): The dataset.

            The `dataset` parameter may be a list of rows, a tuple of rows,
            a `numpy.ndarray` object of rows, or an autom8.Matrix object.

            Each row should contain booleans, numbers, strings, or None values.
            Columns with other kinds of values may be dropped by autom8.

            If the dataset contains any empty rows, or rows that only contain
            blank values, then autom8 will remove those rows from the matrix.

            If some rows contain more values than other rows, then autom8 will
            drop the extra columns, so that each row contains the same number
            of columns.

        column_names (list[str] or str or None): The column names for this
            dataset. Defaults to None.

            If `column_names` is a list of strings, then autom8 will use these
            strings as the names of the dataset's columns.

            If `column_names` is the string `included`, then autom8 will treat
            the first value in each column as the name of that column.

            If `column_names` is the string `missing`, then autom8 will
            generate a name for each column. When generating column names,
            autom8 follows the Excel-style convention of A, B, C, ..., AA, AB,
            ..., BA, BB, etc.

            If `column_names` is the string `unknown` or the value `None`, then
            autom8 will attempt to guess whether it should treat the first
            value of each column as its name, or generate a name for each
            column.

        column_roles (list or dict or None): The roles of some or all of
            the dataset's columns. Defaults to None.

            If `column_roles` is a list of roles, then autom8 will assign each
            role to the corresponding column. (For example, autom8 will assign
            the first role in the list to the first column, the second role to
            the second column, and so on.)

            If `column_roles` is a dict, then the keys may be strings or
            integers. Each string must be the name of a column, and each number
            must be a valid column index.

            The dict does not need to contain a key for each column.

            For each entry in the dict, autom8 looks up the column indicated by
            the entry's key, and assigns it the role indicated by the entry's
            value.

            Each role must be one of these values:

            - `categorical` -- Indicates the column contains categorical data.
            - `encoded` -- Indicates the column's data is already encoded and
              should be treated as opaque data.
            - `numerical` -- Indicates the column contains numerical values.
            - `textual` -- Indicates the column contains text.
            - None -- Indicates that autom8 should attempt to guess the role
              of the column's data.
""")


problem_parameters = _strip("""
        target_column (str or int or None): The column that you want your model
            to predict. Defaults to None.

            If `target_column` is a string, then it must be the name of a
            column in the dataset.

            If `target_column` is an integer, then it must be the index of a
            column in the dataset, or `-1` for the last column.

            If `target_column` is None or `-1`, then autom8 uses the dataset's
            last column as the target column.

        problem_type (str or None): The string `regression` or `classification`
            or None. Defaults to None.

            If the `problem_type` is None, then autom8 will use the role of the
            target column to determine the problem type. If the target column's
            role is `numerical`, then autom8 will use `regression`. Otherwise,
            it will use `classification`.

        test_ratio (float or None): A float between 0 and 1, or None. Defaults
            to None.

            The `test_ratio` indicates how many rows autom8 should use in its
            test dataset.

            If the `test_ratio` is None, autom8 uses a reasonable default
            (currently `0.2`).

        random_state (int or None): The seed used by each estimator's
            random number generator. Defaults to None.

        allow_multicore (bool): Allow estimators to use multiple cores.
            Defaults to True.

        executor_class (class or callable or None): An optional class for
            scheduling estimators to run. Defaults to None.

            The `executor_class` may be a class or a callable. `autom8` calls
            this object when it wants to run multiple estimators in parallel.
            This should return a new object with a few methods:

            - `fit(context, estimator)`: Schedules `context.fit(estimator)` to
              run at some point in the future (which may be right now).
            - `submit(func, *args, **kwargs)`: Schedules `func(*args, **kwargs)`
              to run at some point in the future (which may be right now).
            - `shutdown(self, wait=True)`: Shuts down the executor. If `wait`
              is True, then blocks until all tasks are complete.
""")


receiver_parameter = _strip("""
        receiver (autom8.Receiver or None): The optional Receiver object, for
            receiving out-of-band data from autom8. Defaults to None.

            If the receiver is `None`, autom8 creates a new `autom8.Receiver`
            object that ignores contexts and candidates, and passes any
            warnings that it receives to Python's built-in `exceptions.warn()`
            function.
""")


selector_parameters = _strip("""
        selector (str or list[str] or callable or None): Indicates how to
            select the best candidate pipeline.

            If `selector` is a string, then it must be the name of a metric.
            In this case, autom8 will use that metric to select the best
            pipeline.

            If `selector` is a list of strings, then the strings must be the
            names of metrics. In this case, autom8 will use these metrics to
            select the best pipeline. (Metrics later in the list are used when
            earlier metrics are tied.)

            If `selector` is a callable object, then it should take two
            `autom8.Candidate` objects, and it should return the candidate that
            it thinks is better.

            If `selector` is None, then autom8 will use an appropriate default
            metric, based on the type of problem. For classification problems,
            autom8 will use the `f1_score` metrics. And for regression
            problems, autom8 will use `r2_score`.
""")


common_context_parameters = _render("""
        $dataset_parameters
        $problem_parameters
""")


all_context_parameters = _render("""
        $common_context_parameters
        $receiver_parameter
""")


def render_docstring(obj):
    obj.__doc__ = _render(obj.__doc__)
    return obj
