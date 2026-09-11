"""Audited tool-policy development scoring, not a frontier benchmark."""
from dataclasses import asdict
import hashlib
import re

from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment,ToolFeedback
from irene_brain.evaluation.workspace_sandbox import checked_path,run_workspace_tests


def target_digest(files,target):
    text=files.get(target)
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


def repair_evidence(events,episode_success):
    """Require observed execution failure, later target change, verified finish."""
    failed=None
    changed=None
    for index,event in enumerate(events):
        if not event['success'] and (event.get('tests_attempted') or event.get('validation_attempted')):
            if failed is None: failed=index
        if failed is not None and index>failed and event['success'] and event['action_type'] in ('WRITE_FILE','EDIT_FILE'):
            if event['target_before']!=event['target_after']: changed=index
        if episode_success and changed is not None and index>changed and event['action_type']=='FINISH' and event['verified_completion']:
            return {'failure_event':failed,'target_change_event':changed,'verified_finish_event':index}
    return None


class AuditedToolEnvironment(IsolatedSoftwareEnvironment):
    def __init__(self,files,*,target,private_check):
        self.target=target
        self.events=[]
        self.validation_runs=[]
        def validator(snapshot):
            snapshot=dict(snapshot)
            snapshot['__private_check.py']=private_check
            result=run_workspace_tests(snapshot,'__private_check.py')
            self.validation_runs.append(result)
            return result['passed'],'External checks passed' if result['passed'] else 'External checks failed'
        super().__init__(files,validator=validator)

    def execute_action(self,action):
        before=target_digest(self.files,self.target)
        validations=len(self.validation_runs)
        attempted=False
        match=re.fullmatch(r'ACTION: RUN_TESTS(?: ([^\n]*))?',action) if isinstance(action,str) and len(action.encode())<=70000 else None
        if match:
            try:
                result=run_workspace_tests(self.files,match.group(1) or '')
                attempted=bool(result.get('tests'))
                feedback=ToolFeedback('RUN_TESTS',result['passed'],result['output'])
            except (ValueError,RuntimeError,OSError) as exc:
                feedback=ToolFeedback('RUN_TESTS',False,str(exc))
        else:
            feedback=super().execute_action(action)
        self.events.append({'action_sha256':hashlib.sha256(action.encode()).hexdigest(),
            'action_type':feedback.action_type,'success':feedback.success,'verified_completion':feedback.verified_completion,
            'tests_attempted':attempted,'validation_attempted':len(self.validation_runs)>validations,
            'target_before':before,'target_after':target_digest(self.files,self.target)})
        return feedback


def run_tool_policy_case(agent,request,grader,*,mode):
    if set(request)!={'source_id','family_sha256','initial_prompt','initial_files'}:
        raise ValueError('Request contains unsupported or non-public fields')
    if request['source_id']!=grader['source_id'] or mode not in ('from_scratch','repair'):
        raise ValueError('Task binding or evaluation mode mismatch')
    targets=re.findall(r'^Target: (.+)$',request['initial_prompt'],re.MULTILINE)
    if len(targets)!=1 or targets[0]!=grader['target_module']:
        raise ValueError('Public target and private validator disagree')
    target=checked_path(targets[0])
    files=dict(request['initial_files'])
    prompt=request['initial_prompt']
    if mode=='repair':
        files[target]='raise RuntimeError("Injected evaluation fault")'
        prompt+='The existing target implementation is faulty. Diagnose and repair it.\n'
    good={**files,target:grader['reference_solution']}
    bad={**files,target:'raise RuntimeError("Injected evaluation fault")'}
    def private(snapshot):
        return run_workspace_tests({**snapshot,'__private_check.py':grader['private_check']},'__private_check.py')
    controls={'public_good':run_workspace_tests(good,'test_public.py'),'private_good':private(good),
              'public_bad':run_workspace_tests(bad,'test_public.py'),'private_bad':private(bad)}
    if not (controls['public_good']['passed'] and controls['private_good']['passed'] and
            all(not controls[name]['passed'] and 'Injected evaluation fault' in controls[name]['output'] for name in ('public_bad','private_bad'))):
        raise RuntimeError('Policy-evaluation reference/fault control failed')
    env=AuditedToolEnvironment(files,target=target,private_check=grader['private_check'])
    result=agent.execute_episode(prompt,env,max_cycles=8,max_action_tokens=1024,max_observation_tokens=4096,max_prompt_tokens=4096)
    executed=[entry for entry in result.trace if entry['executed']]
    if len(executed)!=len(env.events) or any(hashlib.sha256(entry['raw_action'].encode()).hexdigest()!=event['action_sha256'] for entry,event in zip(executed,env.events)):
        raise RuntimeError('Episode trace and actual environment executions disagree')
    if result.success and (not env.events or not env.events[-1]['verified_completion']):
        raise RuntimeError('Episode reported unverified success')
    proof=repair_evidence(env.events,result.success)
    return {'source_id':request['source_id'],'family_sha256':request['family_sha256'],'mode':mode,
            'success':result.success,'observed_repair':proof is not None,'repair_evidence':proof,
            'episode':asdict(result),'events':env.events,'final_files':env.files,
            'controls':{name:value['passed'] for name,value in controls.items()},
            'limit':'Procedural development only; same-interpreter tests are not adversarially tamper-proof.'}
