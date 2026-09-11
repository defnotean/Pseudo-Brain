"""Executed teacher traces and causal multi-turn action supervision.

These records are explicitly scripted demonstrations, never autonomous scores.
No canonical solution or hidden grader is added to the model's pointer input.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.agent.isolated_software_environment import ToolFeedback


@dataclass(frozen=True)
class TeacherAction:
    text: str
    kind: str = 'teacher'


def digest_record(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()


def record_executed_trajectory(initial_prompt, environment, actions, *, source_id, validator_id):
    """Execute a bounded teacher plan through the isolated environment once.

Injected faults enter context but are excluded from supervised targets. Teacher
actions may fail (e.g. RUN_TESTS discovers a bug); outcomes are actual feedback.
"""
    if environment.validator is None:
        raise ValueError('Executed training traces require external validation')
    if not all(isinstance(value,str) and value for value in (initial_prompt,source_id,validator_id)):
        raise ValueError('Prompt and provenance IDs must be nonempty strings')
    if not isinstance(actions,(list,tuple)) or not 1<=len(actions)<=64:
        raise ValueError('Teacher plan must contain 1 to 64 actions')
    if not all(isinstance(action,TeacherAction) and action.kind in ('teacher','injected_fault')
               and isinstance(action.text,str) and action.text for action in actions):
        raise ValueError('Invalid teacher action or supervision kind')
    record = {'version':'executed_trajectory_v1','policy_source':'scripted_teacher',
              'initial_prompt':initial_prompt,'source_id':source_id,'validator_id':validator_id,
              'initial_files':environment.files,'steps':[],'status':'incomplete'}
    for index,action in enumerate(actions):
        feedback = environment.execute_action(action.text)
        record['steps'].append({'action':action.text,'kind':action.kind,'feedback':asdict(feedback)})
        if feedback.action_type=='FINISH' and feedback.success and feedback.verified_completion:
            if index!=len(actions)-1:
                raise ValueError('Teacher plan contains actions after verified completion')
            record['status']='complete'
    record['final_files_sha256'] = digest_record(environment.files)
    record['record_sha256'] = digest_record(record)
    return record


def encode_executed_trajectory(record, tokenizer, *, max_tokens=32768):
    """Concatenate exactly the separately encoded frames consumed by the agent.

Labels predict teacher action tokens and their EOS, starting from RESP. Initial
prompt, observations, injected-fault actions, and EOS-to-observation transitions
are masked. There is one reset at task start, never a reset at each action.
"""
    if type(max_tokens) is not int or max_tokens<1:
        raise ValueError('max_tokens must be a positive integer')
    payload = {key:value for key,value in record.items() if key!='record_sha256'}
    if record.get('record_sha256')!=digest_record(payload):
        raise ValueError('Executed record digest mismatch')
    if record.get('version')!='executed_trajectory_v1' or record.get('policy_source')!='scripted_teacher' or record.get('status')!='complete':
        raise ValueError('Only externally completed executed teacher records are accepted')
    steps = record.get('steps')
    if not isinstance(steps,list) or not steps:
        raise ValueError('Executed record has no steps')
    if tokenizer.encode('[RESP]')!=[tokenizer.resp_id] or tokenizer.encode('[EOS]')!=[tokenizer.eos_id]:
        raise ValueError('Tokenizer control IDs do not match their encoded markers')
    prompt = tokenizer.encode(record['initial_prompt'])
    if not prompt:
        raise ValueError('Empty encoded task prompt')
    ids, labels = list(prompt), [-100]*len(prompt)
    spans = []
    for index,step in enumerate(steps):
        if step.get('kind') not in ('teacher','injected_fault'):
            raise ValueError('Unknown supervision kind')
        feedback = ToolFeedback(**step['feedback'])
        complete = feedback.action_type=='FINISH' and feedback.success and feedback.verified_completion
        if complete != (index==len(steps)-1):
            raise ValueError('Only the last action must have verified completion')
        action = tokenizer.encode(step['action'])
        if not action or any(token in (tokenizer.resp_id,tokenizer.eos_id,tokenizer.pad_id) for token in action):
            raise ValueError('Action contains an ambiguous or empty encoded control boundary')
        if step['kind']=='teacher' and feedback.action_type=='UNKNOWN':
            raise ValueError('Malformed actions cannot be supervised teacher targets')
        start = len(ids)
        segment = [tokenizer.resp_id]+action+[tokenizer.eos_id]
        ids.extend(segment)
        labels.extend(segment[1:]+[-100] if step['kind']=='teacher' else [-100]*len(segment))
        end = len(ids)
        observation = tokenizer.encode(observation_frame(feedback))
        if not observation:
            raise ValueError('Empty encoded observation')
        ids.extend(observation)
        labels.extend([-100]*len(observation))
        spans.append({'start':start,'eos_position':end-1,'observation_end':len(ids),'kind':step['kind']})
        if len(ids)>max_tokens:
            raise ValueError('Executed trajectory exceeds the token limit; no truncation allowed')
    if not any(label!=-100 for label in labels):
        raise ValueError('Executed trajectory has no supervised teacher tokens')
    reset = torch.zeros((1,len(ids)),dtype=torch.long)
    reset[0,0]=1
    return {'input_ids':torch.tensor([ids],dtype=torch.long),'labels':torch.tensor([labels],dtype=torch.long),
            'reset_mask':reset,'prompt_tokens':torch.tensor([prompt],dtype=torch.long),
            'action_spans':spans,'record_sha256':record['record_sha256']}
