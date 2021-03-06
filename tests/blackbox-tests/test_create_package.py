import io
import json
import pickle
import zipfile

import autom8
import datasets


def test_create_package():
    acc = datasets.run('iris.csv')
    candidate = acc.candidates[-1]

    example = autom8.create_example_input(
        pipeline=candidate.pipeline,
        dataset=datasets.load('iris.csv'),
        indices=acc.test_indices[1:3],
    )

    package_bytes = autom8.create_package(
        package_name='autom8-test',
        pipeline=candidate.pipeline,
        example_input=example,
        extra_notes='foo bar baz',
    )

    with zipfile.ZipFile(io.BytesIO(package_bytes)) as z:
        assert sorted(z.namelist()) == sorted([
            'autom8-test/.dockerignore',
            'autom8-test/Dockerfile',
            'autom8-test/LICENSE',
            'autom8-test/Makefile',
            'autom8-test/README.md',
            'autom8-test/pipeline.pickle',
            'autom8-test/requirements.txt',
            'autom8-test/service.py',
            'autom8-test/tests.py',
        ])

        def read(name):
            with z.open(f'autom8-test/{name}') as f:
                return f.read().decode('utf-8')

        assert 'requirements.txt' in read('Dockerfile')
        assert 'MIT License' in read('LICENSE')

        with z.open('autom8-test/pipeline.pickle') as f:
            pipeline = pickle.load(f)

        readme = read('README.md')
        assert 'foo bar baz' in readme

        sample_input = _extract_json(readme, '--data \'')
        expected_output = _extract_json(readme, '\nThis will return:\n')
        received_output = pipeline.run(sample_input['rows'])
        assert expected_output['predictions'] == received_output.predictions


def _extract_json(contents, marker):
    i = contents.index(marker)
    j = contents.index('{', i)
    k = contents.index('}', j)
    return json.loads(contents[j : k + 1])
