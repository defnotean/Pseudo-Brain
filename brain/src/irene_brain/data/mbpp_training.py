"""Training-only MBPP case adaptation; preserved references, explicit interfaces."""
import ast
import copy
import hashlib
import unicodedata

from irene_brain.data.procedural_repair_bank import PROTOCOL


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def structural_code_key(code):
    tree = ast.parse(code)
    declarations = [node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    names = {node.name: 'PUBLIC_'+str(index) for index, node in enumerate(declarations)}
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in names:
                node.id = names[node.id]
            return node
        def visit_FunctionDef(self, node):
            if node.name in names:
                node.name = names[node.name]
            return self.generic_visit(node)
        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        def generic_visit(self, node):
            node = super().generic_visit(node)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    node.body = node.body[1:]
            return node
    return 'code_ast:' + digest(ast.dump(Normalize().visit(tree), include_attributes=False))


def row_keys(row):
    keys = {'question:'+digest(normalized(row['text'])), 'documentation:'+digest(normalized(row['text'])),
            'code:'+digest(normalized(row['code']))}
    try:
        keys.add(structural_code_key(row['code']))
    except (SyntaxError, ValueError):
        pass  # The later adapter records an explicit format rejection.
    return sorted(keys)


def public_interface(tree):
    declarations = []
    for original in tree.body:
        if isinstance(original, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = copy.deepcopy(original)
            node.body = [ast.Expr(ast.Constant(Ellipsis))]
            node.decorator_list = []
            declarations.append(ast.unparse(node))
        elif isinstance(original, ast.ClassDef):
            node = copy.deepcopy(original)
            node.decorator_list = []
            methods = []
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method.body = [ast.Expr(ast.Constant(Ellipsis))]
                    method.decorator_list = []
                    methods.append(method)
            node.body = methods or [ast.Expr(ast.Constant(Ellipsis))]
            declarations.append(ast.unparse(node))
    if not declarations:
        raise ValueError('missing_public_declarations')
    return '\n'.join(declarations)


def adapt_case(row):
    task_id = row['task_id']
    if type(task_id) is not int or not 601 <= task_id <= 974:
        raise ValueError('not_official_training_id')
    code = row['code'].strip()
    tree = ast.parse(code)
    if len(row['test_list']) != 3 or row['challenge_test_list']:
        raise ValueError('unexpected_pinned_test_layout')
    for assertion in row['test_list']:
        nodes = ast.parse(assertion).body
        if len(nodes) != 1 or not isinstance(nodes[0], ast.Assert):
            raise ValueError('unsupported_test_statement')
    setup = row['test_setup_code'].strip()
    ast.parse(setup)
    module = 'mbpp_'+str(task_id)
    required_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not required_names:
        raise ValueError('missing_public_declarations')
    def wrap(assertions, name):
        # Wildcard imports are legal only at module level. Keep setup at module
        # scope too, preserving globals used by setup-created objects/functions.
        # Wildcard imports omit underscore-prefixed names such as MBPP's _sum.
        # Explicitly import every declared API too; preserve other exports used
        # by the original tests/setup through the wildcard import.
        prefix = 'from '+module+' import *\nfrom '+module+' import '+', '.join(required_names)+'\n' + (setup+'\n' if setup else '')
        body = '\n'.join(assertions)
        result = prefix+'\ndef '+name+'():\n'+'\n'.join('    '+line for line in body.splitlines())+'\n'
        ast.parse(result)
        return result
    public = wrap(row['test_list'][:1], 'test_public')
    private = wrap(row['test_list'], 'test_private')
    interface = public_interface(tree)
    prompt = PROTOCOL+'Task: '+row['text'].strip()+'\nTarget: '+module+'.py\nRequired public interface (bodies omitted):\n'+interface+'\n'
    if any(marker in text for text in (prompt, code, public, private) for marker in ('[RESP]', '[EOS]', '[PAD]', '[THREAD:', '[OBSERVATION]')):
        raise ValueError('ambiguous_reserved_boundary')
    return {'source_id':'mbpp-train:'+str(task_id), 'task_id':task_id, 'partition':'train',
            'initial_prompt':prompt, 'initial_files':{'test_public.py':public}, 'private_check':private,
            'target_module':module+'.py', 'reference_solution':code, 'public_interface':interface,
            'family_sha256':structural_code_key(code).removeprefix('code_ast:'), 'isolation_keys':row_keys(row)}
