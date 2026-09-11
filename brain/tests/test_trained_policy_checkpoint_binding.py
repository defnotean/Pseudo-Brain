import hashlib
import json

import pytest
from evaluate_trained_tool_policy import resolve_completed_checkpoint, SELECTION, TRAINING_SOURCE


def test_complete_pair_binding_rejects_partial_diagnostic_and_changed_weights(tmp_path):
    pair={'status':'complete','native_preflight':False,'source_manifest_sha256':TRAINING_SOURCE,'models':{}}
    for name in ('recurrent','transformer'):
        folder=tmp_path/name
        folder.mkdir()
        checkpoint=folder/'checkpoint-3520-trained.pt'
        # Inert bytes test artifact binding only; no model is loaded in this test.
        checkpoint.write_bytes(b'checkpoint fixture')
        child={'status':'complete','completed_updates':3520,'selection_sha256':SELECTION,'native_preflight':False,
               'parity':[{}, {}, {}] if name=='recurrent' else [],
               'checkpoint':{'checkpoint':checkpoint.name,'completed_updates':3520,
                             'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
        pair['models'][name]={'status':'complete','child_report':child}
    path=tmp_path/'report.json'
    def publish():
        path.write_text(json.dumps(pair))
        return hashlib.sha256(path.read_bytes()).hexdigest()
    checksum=publish()
    result=resolve_completed_checkpoint(tmp_path,checksum,'recurrent')
    assert result[0]==tmp_path/'recurrent/checkpoint-3520-trained.pt'
    with pytest.raises(ValueError,match='report hash'):
        resolve_completed_checkpoint(tmp_path,'0'*64,'recurrent')
    pair['models']['transformer']['child_report']['completed_updates']=880
    with pytest.raises(ValueError,match='partial training'):
        resolve_completed_checkpoint(tmp_path,publish(),'recurrent')
    pair['models']['transformer']['child_report']['completed_updates']=3520
    pair['native_preflight']=True
    with pytest.raises(ValueError,match='completed registered'):
        resolve_completed_checkpoint(tmp_path,publish(),'recurrent')
    pair['native_preflight']=False
    checksum=publish()
    (tmp_path/'recurrent/checkpoint-3520-trained.pt').write_bytes(b'changed bytes')
    with pytest.raises(ValueError,match='checkpoint hash'):
        resolve_completed_checkpoint(tmp_path,checksum,'recurrent')
