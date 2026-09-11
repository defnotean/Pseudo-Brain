"""Frozen-bank byte compatibility, not training or a capability evaluation."""
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from irene_brain.data.behavior_repair_trajectory import encode_training_trajectory
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

reference=Path('/content/pb-procedural-repair-bank-v1/src/irene_brain/data/executed_trajectory.py')
spec=importlib.util.spec_from_file_location('frozen_v1_codec',reference)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
episode_reference=reference.parents[1]/'agent/parallel_depth_episode.py'
episode_spec=importlib.util.spec_from_file_location('frozen_v1_episode_frames',episode_reference)
episode_module=importlib.util.module_from_spec(episode_spec)
sys.modules[episode_spec.name]=episode_module
episode_spec.loader.exec_module(episode_module)
module.observation_frame=episode_module.observation_frame
tokenizer_path=Path(os.environ['PB_TOKENIZER_PATH'])
assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
tokenizer=BpeSemanticTokenizer(tokenizer_file=tokenizer_path,auto_build_if_missing=False)
banks=[('/content/pb-procedural-repair-bank-v1/corpus/train.jsonl.gz','896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0',352),
       ('/content/pb-procedural-repair-bank-v1/corpus/development.jsonl.gz','0381456052591928c18debafd5e86bf6dc1c304c1793bec2740e0c30f7cccae8',24),
       ('/content/pb-varied-repair-v2-pilot-r0/pilot/train-pilot.jsonl.gz','323aae211d6467e2805db728dc16492d5c48a662e822822579eb7406bd6158a2',220)]
digest=hashlib.sha256()
total=0
for path,expected,count in banks:
    raw=Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    records=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
    assert len(records)==count
    for record in records:
        old=module.encode_executed_trajectory(record,tokenizer,max_tokens=8192)
        new=encode_training_trajectory(record,tokenizer,max_tokens=8192)
        assert old.keys()==new.keys()
        for key in ('input_ids','labels','reset_mask','prompt_tokens'):
            assert old[key].dtype==new[key].dtype and old[key].shape==new[key].shape
            old_bytes=old[key].numpy().tobytes()
            assert old_bytes==new[key].numpy().tobytes()
            digest.update(record['record_sha256'].encode()+key.encode()+old_bytes)
        assert old['action_spans']==new['action_spans'] and old['record_sha256']==new['record_sha256']
        total+=1
report={'status':'complete','records':total,'tensor_sequence_sha256':digest.hexdigest(),
        'frozen_v1_source_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
        'frozen_v1_episode_source_sha256':hashlib.sha256(episode_reference.read_bytes()).hexdigest(),
        'model_updates':0,'model_generated_tokens':0,'development_use':'encoding compatibility only'}
Path('v1-compatibility.json').write_text(json.dumps(report,indent=2))
print('V1_COMPATIBILITY',json.dumps(report))
