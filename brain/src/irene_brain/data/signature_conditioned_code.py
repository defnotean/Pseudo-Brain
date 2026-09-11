"""Opt-in code instruction formatting; frozen v4 records remain unchanged."""
import ast
import io
import re
import tokenize

from irene_brain.data.broad_corpus import digest, normalized
from irene_brain.data.foundation_corpus import complete_exchange


def function_declaration(source):
    """Preserve the exact declaration, including decorators and multiline args."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError('unparseable_python_function') from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0],(ast.FunctionDef,ast.AsyncFunctionDef)):
        raise ValueError('expected_one_function_definition')
    function = tree.body[0]
    first_line = min([function.lineno]+[item.lineno for item in function.decorator_list])
    # Tokenizer line numbers follow physical newlines, not Unicode separators.
    lines = re.findall(r'[^\r\n]*(?:\r\n|\r|\n|$)',source)
    depth = 0
    in_declaration = False
    for token in tokenize.generate_tokens(io.StringIO(source,newline=None).readline):
        if token.type == tokenize.NAME and token.string == 'def' and token.start[0] == function.lineno:
            in_declaration = True
        if not in_declaration or token.type != tokenize.OP:
            continue
        if token.string in ('(','[','{'):
            depth += 1
        elif token.string in (')',']','}'):
            depth -= 1
        elif token.string == ':' and depth == 0:
            end_line,end_column = token.end
            declaration = ''.join(lines[first_line-1:end_line-1])+lines[end_line-1][:end_column]
            return function.name,declaration
    raise ValueError('missing_function_declaration_boundary')


def signature_conditioned_exchange(row):
    result = complete_exchange(row,'code')
    name,declaration = function_declaration(result['answer'])
    question = ('Implement the Python function below according to its documentation.\n\n'
                +declaration+'\n\nDocumentation:\n'+row['func_documentation_string'].strip())
    text = 'User: '+question+' [RESP]'+result['answer']+'[EOS]'
    # Retain old keys as well, so changing a prompt cannot bypass old quarantine.
    keys = set(result['isolation_keys'])
    keys.update(('question:'+digest(normalized(question)),'text:'+digest(normalized(text))))
    result.update(question=question,text=text,text_sha256=digest(text),
                  isolation_keys=sorted(keys),entry_point=name,declaration=declaration,
                  format_version='signature_conditioned_v1')
    return result
