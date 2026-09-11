import ast
import pytest

from irene_brain.data.foundation_corpus import complete_exchange
from irene_brain.data.signature_conditioned_code import function_declaration, signature_conditioned_exchange


@pytest.mark.parametrize('source',[
    'def identity(x): return x',
    'async def fetch(url: str) -> str:\n    return url',
    'def choose(\n    x: dict[str, int],\n    callback=lambda value: value,\n    *, flag=True,\n):\n    return callback(x)',
    '@decorate(\n    enabled=True\n)\ndef identity(x):\n    return x',
    'def κόσμος(x, /, *args, **kwargs):\n    return x',
    'def colon_value(x={"key": ":"}) -> str:\n    return x["key"]',
    'def unicode_value(x="a\u2028b"):\n    return x',
    'def windows_lines(\r\n    x: int,\r\n):\r\n    return x',
])
def test_exact_interface_is_preserved_without_function_body(source):
    name,declaration = function_declaration(source)
    original = ast.parse(source).body[0]
    rebuilt = ast.parse(declaration+'\n    pass').body[0]
    assert name == original.name == rebuilt.name
    assert ast.dump(original.args) == ast.dump(rebuilt.args)
    assert (ast.dump(original.returns) if original.returns else None) == (ast.dump(rebuilt.returns) if rebuilt.returns else None)
    assert [ast.dump(item) for item in original.decorator_list] == [ast.dump(item) for item in rebuilt.decorator_list]
    assert 'return ' not in declaration


def test_conditioning_preserves_answer_and_old_isolation_keys():
    row = {'repository_name':'Owner/Repo','func_documentation_string':'Return the supplied value.',
           'func_code_string':'def identity(x):\n    return x'}
    old = complete_exchange(row,'code')
    new = signature_conditioned_exchange(row)
    assert new['answer'] == old['answer']
    assert set(old['isolation_keys']) <= set(new['isolation_keys'])
    assert 'def identity(x):' in new['question'] and 'return x' not in new['question']
    assert new['text_sha256'] != old['text_sha256']
    assert row['func_code_string'] == 'def identity(x):\n    return x'
    assert 'token_count' not in new and 'partition' not in new


@pytest.mark.parametrize('source',['def broken(', 'x = 1', 'def f(): pass\ndef g(): pass'])
def test_ambiguous_or_invalid_source_is_rejected(source):
    with pytest.raises(ValueError):
        function_declaration(source)
