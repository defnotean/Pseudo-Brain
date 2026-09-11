"""Experimental six-actuator environment with a virtual bounded workspace.

Writes change inert text only. Tests use a read-only snapshot inside Bubblewrap.
FINISH requires a separate supplied validator; a model's claim never suffices.
"""
from dataclasses import dataclass
import re

from irene_brain.evaluation.workspace_sandbox import checked_path, checked_snapshot, run_workspace_tests


@dataclass(frozen=True)
class ToolFeedback:
    action_type: str
    success: bool
    observation_text: str
    verified_completion: bool = False


class IsolatedSoftwareEnvironment:
    def __init__(self, files=None, *, validator=None, retrieve=None):
        self._files = checked_snapshot(dict(files or {}))
        self.validator = validator
        self.retrieve = retrieve

    @property
    def files(self):
        return dict(self._files)

    def execute_action(self, action):
        if not isinstance(action,str) or len(action.encode())>70000:
            return ToolFeedback('UNKNOWN',False,'Action must be text within 70,000 bytes')
        header, separator, body = action.partition('\n')
        match = re.fullmatch(r'ACTION: (WRITE_FILE|READ_FILE|EDIT_FILE|RUN_TESTS|RETRIEVE_MEMORY|FINISH)(?: (.*))?',header)
        if match is None:
            return ToolFeedback('UNKNOWN',False,'Unrecognized action format')
        verb, arg = match.group(1), match.group(2) or ''
        if verb not in ('WRITE_FILE','EDIT_FILE') and separator:
            return ToolFeedback(verb,False,'This action accepts one header line only')
        try:
            if verb in ('WRITE_FILE','READ_FILE','EDIT_FILE'):
                checked_path(arg)
            if verb == 'WRITE_FILE':
                if not separator:
                    raise ValueError('WRITE_FILE requires a newline before its body')
                updated = dict(self._files)
                updated[arg] = body
                self._files = checked_snapshot(updated)
                return ToolFeedback(verb,True,f'Wrote {len(body.encode())} bytes to {arg}')
            if verb == 'READ_FILE':
                if arg not in self._files:
                    raise ValueError('File does not exist')
                return ToolFeedback(verb,True,self._files[arg])
            if verb == 'EDIT_FILE':
                edit = re.fullmatch(r'<<<TARGET\n(.*?)\n===\n(.*?)\n>>>',body,re.DOTALL)
                if edit is None or not edit.group(1):
                    raise ValueError('EDIT_FILE requires a nonempty exact target and replacement block')
                if arg not in self._files or self._files[arg].count(edit.group(1))!=1:
                    raise ValueError('Edit target must occur exactly once in the existing file')
                updated = dict(self._files)
                updated[arg] = updated[arg].replace(edit.group(1),edit.group(2),1)
                self._files = checked_snapshot(updated)
                return ToolFeedback(verb,True,f'Edited {arg}')
            if verb == 'RUN_TESTS':
                result = run_workspace_tests(self._files,arg)
                return ToolFeedback(verb,result['passed'],result['output'])
            if verb == 'RETRIEVE_MEMORY':
                if self.retrieve is None:
                    return ToolFeedback(verb,False,'No retrieval provider configured')
                text = self.retrieve(arg)
                if not isinstance(text,str) or not text:
                    return ToolFeedback(verb,False,'No memory result')
                if len(text.encode())>65536:
                    raise ValueError('Memory result exceeds 64 KiB')
                return ToolFeedback(verb,True,text)
            if self.validator is None:
                return ToolFeedback(verb,False,'Completion requires an external validator')
            passed, detail = self.validator(self.files)
            if type(passed) is not bool or not isinstance(detail,str) or len(detail.encode())>65536:
                raise ValueError('Invalid validator result')
            return ToolFeedback(verb,passed,detail,verified_completion=passed)
        except (ValueError,RuntimeError,OSError) as exc:
            return ToolFeedback(verb,False,str(exc))
