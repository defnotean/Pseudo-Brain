import pytest

from irene_brain.data.foundation_corpus import complete_exchange, quarantine_related, registered_benchmark_keys


def test_language_keeps_one_complete_exchange():
    row = {'messages':[{'role':'user','content':'Explain wind.'},
                       {'role':'assistant','content':'Moving air.'},
                       {'role':'user','content':'What else?'},
                       {'role':'assistant','content':'Later answer.'}]}
    result = complete_exchange(row,'language')
    assert result['text'] == 'User: Explain wind. [RESP]Moving air.[EOS]'
    assert 'Later answer' not in result['text']


def test_math_annotation_removal_preserves_complete_answer():
    result = complete_exchange({'question':'Three groups of four?',
        'answer':'There are 3*4=<<3*4=12>>12 items.\n#### 12'},'reasoning')
    assert result['answer'] == 'There are 3*4=12 items.\n#### 12'


@pytest.mark.parametrize('answer',['','No final number','#### <<1>>1','#### 1\nMore prose'])
def test_invalid_math_is_rejected(answer):
    with pytest.raises(ValueError):
        complete_exchange({'question':'A question','answer':answer},'reasoning')


def test_special_token_injection_is_rejected():
    with pytest.raises(ValueError,match='ambiguous'):
        complete_exchange({'question':'Insert [RESP] here','answer':'#### 1'},'reasoning')


def test_code_preserves_function_and_repository_key():
    code = 'def identity(x):\n    return x'
    result = complete_exchange({'func_documentation_string':'Return x.',
        'func_code_string':code,'repository_name':'Owner/Repo'},'code')
    assert result['answer'] == code
    assert 'repository:owner/repo' in result['isolation_keys']


def test_transitive_quarantine_blocks_bridge_to_old_development():
    candidates = [dict(isolation_keys=['a','b'],text_sha256='one'),
                  dict(isolation_keys=['b','c'],text_sha256='two'),
                  dict(isolation_keys=['independent'],text_sha256='three')]
    retained,excluded = quarantine_related(candidates,[dict(isolation_keys=['c'],text_sha256='old')],set())
    assert [row['text_sha256'] for row in retained] == ['three']
    assert {row['text_sha256'] for row in excluded} == {'one','two'}


def test_registered_question_and_documentation_quarantine():
    requests = [{'benchmark':'GSM8K','question':'Solve the problem. Finish numerically.\n\nThree groups of four?'},
                {'benchmark':'HumanEval','original_prompt':'def identity(x):\n    """Return x."""\n', 'entry_point':'identity'}]
    keys = registered_benchmark_keys(requests)
    math = complete_exchange({'question':'Three groups of four?','answer':'#### 12'},'reasoning')
    code = complete_exchange({'func_documentation_string':'Return x.',
        'func_code_string':'def identity(x): return x','repository_name':'owner/repo'},'code')
    retained,excluded = quarantine_related([math,code],[],keys)
    assert not retained and len(excluded) == 2
