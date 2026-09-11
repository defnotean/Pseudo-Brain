"""Exact model prefixes plus verified teacher continuations, separately versioned.

Actor tokens/observations are context only. This module does not generate model
actions, replay tests, change session state, or convert scripted v1 records.
"""
import copy
from dataclasses import asdict
import hashlib
import re

import torch

from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment, ToolFeedback
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.data.executed_trajectory import TeacherAction, digest_record, encode_executed_trajectory


VERSION = 'behavior_repair_trajectory_v2'
SOURCE = 'mixed_model_teacher'
VERBS = 'WRITE_FILE|READ_FILE|EDIT_FILE|RUN_TESTS|RETRIEVE_MEMORY|FINISH'


def _actor_metadata(actor):
    if not isinstance(actor, dict) or actor.get('architecture') not in ('parallel_depth_v1','comparison_transformer_v1'):
        raise ValueError('Unsupported actor architecture')
    if actor.get('provenance_kind') not in ('model_checkpoint','codec_fixture'):
        raise ValueError('Missing actor provenance kind')
    for key in ('checkpoint_sha256','tokenizer_sha256','source_manifest_sha256','initial_case_sha256'):
        if not isinstance(actor.get(key), str) or re.fullmatch('[0-9a-f]{64}',actor[key]) is None:
            raise ValueError('Missing or invalid actor provenance: '+key)
        if actor['provenance_kind']=='model_checkpoint' and actor[key]=='0'*64:
            raise ValueError('Fixture hash cannot identify a model collection')
    if not isinstance(actor.get('collection_protocol'),str) or not actor['collection_protocol']:
        raise ValueError('Missing collection protocol')


def _tokens(ids, tokenizer, *, empty=False):
    if not isinstance(ids,list) or (not ids and not empty) or any(type(value) is not int or not 0 <= value < tokenizer.vocab_size for value in ids):
        raise ValueError('Invalid action token IDs')
    if any(value in (tokenizer.resp_id,tokenizer.eos_id,tokenizer.pad_id) for value in ids):
        raise ValueError('Ambiguous control token inside action')


def _step(action, ids, kind, feedback, tokenizer):
    return {'action':action, 'action_token_ids':list(ids), 'kind':kind,
            'terminated_by_eos':True, 'feedback':asdict(feedback),
            'observation_token_ids':tokenizer.encode(observation_frame(feedback))}


def record_behavior_repair(initial_prompt, initial_files, episode, environment, teacher_actions,
                           tokenizer, *, actor, source_id, validator_id):
    """Continue the same environment after a completed, bounded actor prefix.

The caller preserves raw collection attempts before calling this function.
Token/context/observation caps and already-completed episodes are excluded.
File operations are checked on inert shadow text; tests and observations are
never replayed. Teacher execution begins only after prefix validation succeeds.
"""
    _actor_metadata(actor)
    if environment.validator is None:
        raise ValueError('Teacher continuation requires external validation')
    if not all(isinstance(value,str) and value for value in (initial_prompt,source_id,validator_id)):
        raise ValueError('Prompt and provenance IDs must be nonempty')
    if episode.success or episode.stop_reason != 'cycle_limit':
        raise ValueError('Only complete bounded actor prefixes may receive teacher continuation')
    if episode.initial_prompt_sha256 != hashlib.sha256(initial_prompt.encode()).hexdigest():
        raise ValueError('Actor initial prompt changed')
    if actor['architecture']=='parallel_depth_v1' and episode.state_bytes != 4096:
        raise ValueError('Actor recurrent state violates the 4096-byte contract')
    trace = episode.trace
    if not isinstance(trace,(list,tuple)) or not trace or episode.cycles != len(trace):
        raise ValueError('Invalid actor trace count')
    if not isinstance(teacher_actions,(list,tuple)) or not teacher_actions or len(trace)+len(teacher_actions)>64:
        raise ValueError('Invalid combined action count')
    if not all(isinstance(action,TeacherAction) and action.kind=='teacher' and isinstance(action.text,str) and action.text for action in teacher_actions):
        raise ValueError('Continuation accepts only teacher actions')
    prepared_teacher=[]
    for action in teacher_actions:
        action_ids=tokenizer.encode(action.text)
        _tokens(action_ids,tokenizer)
        prepared_teacher.append((action,action_ids))
    shadow = IsolatedSoftwareEnvironment(initial_files)
    steps = []
    prefix = tokenizer.encode(initial_prompt)
    for index, entry in enumerate(trace):
        if entry.get('cycle') != index+1 or entry.get('generation_stop')!='eos' or entry.get('executed') is not True or entry.get('observation_ingested') is not True:
            raise ValueError('Unfinished or unconsumed actor action cannot become a training prefix')
        if entry.get('state_bytes') != episode.state_bytes:
            raise ValueError('Actor per-action state accounting changed')
        ids = entry.get('token_ids')
        _tokens(ids,tokenizer,empty=True)
        action = entry.get('raw_action')
        if not isinstance(action,str) or tokenizer.decode(ids,skip_special=False) != action:
            raise ValueError('Actor token IDs disagree with decoded action')
        feedback = ToolFeedback(**entry['feedback'])
        if feedback.verified_completion:
            raise ValueError('Actor already verified completion')
        match = re.fullmatch('ACTION: ('+VERBS+')(?: (.*))?', action.partition('\n')[0])
        verb = match.group(1) if match else 'UNKNOWN'
        if verb != feedback.action_type:
            raise ValueError('Actor action and feedback verb disagree')
        if verb in ('WRITE_FILE','READ_FILE','EDIT_FILE','UNKNOWN'):
            # These operations manipulate/read inert text only, never run tests.
            observed = shadow.execute_action(action)
            if asdict(observed) != asdict(feedback):
                raise ValueError('Actor file-operation feedback does not match initial workspace')
        step = _step(action,ids,'behavior_model',feedback,tokenizer)
        if len(step['observation_token_ids']) != entry.get('observation_tokens'):
            raise ValueError('Actor observation framing changed')
        prefix += [tokenizer.resp_id]+ids+[tokenizer.eos_id]+step['observation_token_ids']
        steps.append(step)
    if actor['architecture']=='comparison_transformer_v1' and (episode.state_bytes is not None or episode.prefix_tokens!=len(prefix)):
        raise ValueError('Transformer consumed-prefix accounting changed')
    if actor['architecture']=='parallel_depth_v1' and episode.prefix_tokens is not None:
        raise ValueError('Recurrent actor must not retain a replay prefix')
    if shadow.files != environment.files:
        raise ValueError('Teacher environment differs from actor final workspace')
    if any(environment.files.get(path) != text for path,text in initial_files.items()):
        raise ValueError('Actor changed protected initial files')
    record = {'version':VERSION, 'policy_source':SOURCE, 'status':'incomplete',
              'initial_prompt':initial_prompt, 'initial_files':copy.deepcopy(initial_files),
              'source_id':source_id, 'validator_id':validator_id, 'actor':copy.deepcopy(actor),
              'behavior_steps':len(steps), 'behavior_prefix_sha256':digest_record(prefix),
              'behavior_final_files_sha256':digest_record(environment.files),
              'actor_trace':copy.deepcopy(list(trace)), 'actor_state_bytes':episode.state_bytes,
              'actor_prefix_tokens':episode.prefix_tokens, 'steps':steps}
    for index, (action, ids) in enumerate(prepared_teacher):
        feedback = environment.execute_action(action.text)
        steps.append(_step(action.text,ids,'teacher',feedback,tokenizer))
        if feedback.action_type=='FINISH' and feedback.success and feedback.verified_completion:
            if index != len(teacher_actions)-1:
                raise ValueError('Teacher actions follow verified completion')
            record['status']='complete'
    record['final_files_sha256']=digest_record(environment.files)
    record['record_sha256']=digest_record(record)
    return record


def encode_behavior_repair(record, tokenizer, *, max_tokens=8192, max_action_tokens=1024,
                           max_observation_tokens=4096, max_prompt_tokens=4096, allow_fixture=False):
    for value in (max_tokens,max_action_tokens,max_observation_tokens,max_prompt_tokens):
        if type(value) is not int or value<1:
            raise ValueError('Token limits must be positive integers')
    if record.get('record_sha256') != digest_record({k:v for k,v in record.items() if k!='record_sha256'}):
        raise ValueError('Mixed record digest mismatch')
    if record.get('version')!=VERSION or record.get('policy_source')!=SOURCE or record.get('status')!='complete':
        raise ValueError('Only externally completed mixed records are accepted')
    _actor_metadata(record.get('actor'))
    if record['actor']['provenance_kind']=='codec_fixture' and not allow_fixture:
        raise ValueError('Codec fixtures cannot enter ordinary training')
    if record['actor']['architecture']=='parallel_depth_v1' and record.get('actor_state_bytes')!=4096:
        raise ValueError('Recorded recurrent state violates the 4096-byte contract')
    if tokenizer.encode('[RESP]')!=[tokenizer.resp_id] or tokenizer.encode('[EOS]')!=[tokenizer.eos_id]:
        raise ValueError('Tokenizer marker IDs changed')
    prompt = tokenizer.encode(record['initial_prompt'])
    if not prompt or len(prompt)>max_prompt_tokens:
        raise ValueError('Prompt token limit exceeded')
    steps, count = record.get('steps'), record.get('behavior_steps')
    if not isinstance(steps,list) or type(count) is not int or not 1<=count<len(steps)<=64:
        raise ValueError('Invalid mixed action layout')
    actor_trace=record.get('actor_trace')
    if not isinstance(actor_trace,list) or len(actor_trace)!=count:
        raise ValueError('Actor trace count changed')
    ids, labels, spans = list(prompt), [-100]*len(prompt), []
    for index, step in enumerate(steps):
        expected = 'behavior_model' if index<count else 'teacher'
        if step.get('kind')!=expected or step.get('terminated_by_eos') is not True:
            raise ValueError('Mixed supervision order or real EOS is invalid')
        action_ids = step.get('action_token_ids')
        _tokens(action_ids,tokenizer,empty=expected=='behavior_model')
        if tokenizer.decode(action_ids,skip_special=False) != step.get('action'):
            raise ValueError('Stored action tokens disagree with action text')
        if expected=='behavior_model':
            trace=actor_trace[index]
            if (trace.get('cycle')!=index+1 or trace.get('token_ids')!=action_ids or
                trace.get('raw_action')!=step['action'] or trace.get('feedback')!=step['feedback'] or
                trace.get('generation_stop')!='eos' or trace.get('executed') is not True or
                trace.get('observation_ingested') is not True or trace.get('observation_tokens')!=len(step['observation_token_ids'])):
                raise ValueError('Stored behavior step disagrees with original actor trace')
        if expected=='teacher' and tokenizer.encode(step['action']) != action_ids:
            raise ValueError('Teacher action tokenization changed')
        if len(action_ids)+1>max_action_tokens:
            raise ValueError('Action token limit exceeded; no truncation allowed')
        feedback = ToolFeedback(**step['feedback'])
        complete = feedback.action_type=='FINISH' and feedback.success and feedback.verified_completion
        if complete != (index==len(steps)-1):
            raise ValueError('Only final teacher action must verify completion')
        if expected=='teacher' and feedback.action_type=='UNKNOWN':
            raise ValueError('Malformed teacher target')
        observation = tokenizer.encode(observation_frame(feedback))
        if observation != step.get('observation_token_ids') or not observation or len(observation)>max_observation_tokens:
            raise ValueError('Observation framing or limit changed')
        if index==count and digest_record(ids)!=record.get('behavior_prefix_sha256'):
            raise ValueError('Exact actor prefix identity changed')
        start = len(ids)
        segment = [tokenizer.resp_id]+action_ids+[tokenizer.eos_id]
        ids.extend(segment)
        labels.extend(segment[1:]+[-100] if expected=='teacher' else [-100]*len(segment))
        eos = len(ids)-1
        ids.extend(observation)
        labels.extend([-100]*len(observation))
        spans.append({'start':start,'eos_position':eos,'observation_end':len(ids),'kind':expected})
        if len(ids)>max_tokens:
            raise ValueError('Mixed trajectory exceeds token limit; no truncation allowed')
    reset = torch.zeros((1,len(ids)),dtype=torch.long)
    reset[0,0]=1
    return {'input_ids':torch.tensor([ids],dtype=torch.long),'labels':torch.tensor([labels],dtype=torch.long),
            'reset_mask':reset,'prompt_tokens':torch.tensor([prompt],dtype=torch.long),
            'action_spans':spans,'record_sha256':record['record_sha256']}


def encode_training_trajectory(record, tokenizer, *, max_tokens=8192):
    """Keep the legacy v1 encoder untouched and byte-compatible."""
    if record.get('version')=='executed_trajectory_v1':
        return encode_executed_trajectory(record,tokenizer,max_tokens=max_tokens)
    return encode_behavior_repair(record,tokenizer,max_tokens=max_tokens)
