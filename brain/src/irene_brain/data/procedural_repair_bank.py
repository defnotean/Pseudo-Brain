"""Partitionable procedural tool-policy development cases, not frontier tasks."""
import ast
import hashlib

from irene_brain.data.executed_trajectory import TeacherAction


def solution_family(task):
    tree=ast.parse(task.reference_solution)
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id==task.target_function: node.id='TARGET'
            return node
        def visit_FunctionDef(self,node):
            if node.name==task.target_function: node.name='TARGET'
            return self.generic_visit(node)
        def visit_ClassDef(self,node):
            if node.name==task.target_function: node.name='TARGET'
            return self.generic_visit(node)
    normalized=ast.dump(Normalize().visit(tree),include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()


def split_checks(source):
    """Public prefix through the first assertion; full suite stays private.

    Keep the first assertion in the full validator too: evaluating assertions
    can mutate queues/stacks needed by subsequent checks. No private suffix is
    added to the public prefix.
"""
    tree=ast.parse(source)
    assertions=[i for i,node in enumerate(tree.body) if isinstance(node,ast.Assert)]
    if len(assertions)<2:
        raise ValueError('At least two top-level assertions are required')
    first=assertions[0]
    public=tree.body[:first+1]
    private=[node for node in tree.body if not
             (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
              and isinstance(node.value.func,ast.Name) and node.value.func.id=='print')]
    def wrap(nodes,name):
        body='\n'.join(ast.unparse(node) for node in nodes)
        return 'def '+name+'():\n'+'\n'.join('    '+line for line in body.splitlines())+'\n'
    return wrap(public,'test_public'),wrap(private,'test_private')


PROTOCOL='''Use one raw action per response. Commands:
ACTION: READ_FILE path
ACTION: WRITE_FILE path\nfile content
ACTION: EDIT_FILE path\n<<<TARGET\nexact old text\n===\nreplacement text\n>>>
ACTION: RUN_TESTS [path]
ACTION: RETRIEVE_MEMORY query
ACTION: FINISH summary
Public tests are available in test_public.py. Completion is checked externally.
'''


def repair_case(task, *, repair):
    public,private=split_checks(task.hidden_tests_code)
    prompt=PROTOCOL+'Task: '+task.goal+'\nTarget: '+task.target_module+'\n'
    actions=[TeacherAction('ACTION: READ_FILE test_public.py')]
    reference=task.reference_solution.strip()
    if repair:
        # Guaranteed observable fault, not a model-generated mistake.
        stub='raise RuntimeError("Injected training fault")'
        actions += [TeacherAction('ACTION: WRITE_FILE '+task.target_module+'\n'+stub,'injected_fault'),
                    TeacherAction('ACTION: RUN_TESTS'),
                    TeacherAction('ACTION: READ_FILE '+task.target_module),
                    TeacherAction('ACTION: EDIT_FILE '+task.target_module+'\n<<<TARGET\n'+stub+'\n===\n'+reference+'\n>>>')]
    else:
        actions.append(TeacherAction('ACTION: WRITE_FILE '+task.target_module+'\n'+reference))
    actions += [TeacherAction('ACTION: RUN_TESTS'),TeacherAction('ACTION: FINISH Implementation verified')]
    return {'source_id':task.task_id+':'+task.target_module,'domain':task.domain,
            'family_sha256':solution_family(task),'initial_prompt':prompt,
            'initial_files':{**task.initial_files,'test_public.py':public},
            'private_check':private,'target_module':task.target_module,
            'reference_solution':reference,'repair':repair,'actions':actions}


def partition_tasks(tasks, *, train_per_family=16, development_per_family=4):
    families={}
    for task in tasks:
        families.setdefault(solution_family(task),[]).append(task)
    if len(families)<10:
        raise ValueError('Too few distinct solution families')
    selected={'train':[],'development':[]}
    mapping={}
    for rank,key in enumerate(sorted(families)):
        partition='development' if rank%5==0 else 'train'
        count=development_per_family if partition=='development' else train_per_family
        available=sorted(families[key],key=lambda task:(task.target_module,task.target_function))
        if len(available)<count:
            raise ValueError('A solution family has insufficient candidates')
        mapping[key]={'partition':partition,'available':len(available),'selected':count}
        selected[partition].extend(repair_case(task,repair=bool(index%2)) for index,task in enumerate(available[:count]))
    if {row['family_sha256'] for row in selected['train']} & {row['family_sha256'] for row in selected['development']}:
        raise ValueError('Solution family leaked across partitions')
    return selected,mapping
