"""Reuse unchanged policy rollout/scoring, with strict v2 checkpoint binding."""
import evaluate_trained_tool_policy as evaluator
from resolve_broad_policy_v2 import resolve_completed_checkpoint,SELECTION

evaluator.resolve_completed_checkpoint=resolve_completed_checkpoint
evaluator.SELECTION=SELECTION
original_provenance=evaluator.provenance


def provenance(**kwargs):
    kwargs['train_seed']=1987
    return original_provenance(**kwargs)


evaluator.provenance=provenance

if __name__=='__main__':evaluator.main()
