# Pseudo-Brain

This is the canonical workspace for the persistent multi-thought sensorimotor
model research project.

The implementation, experiment configurations, tests, and research records are
under [brain](./brain/README.md). The Irene desktop/chat project is separate and
is not a dependency or training-data source for Pseudo-Brain.

The Python namespace `irene_brain` is retained temporarily for compatibility
with immutable historical checkpoints. New releases and runs are produced from
this project directory.

The checked-in `.pseudo-brain-workspace-v2` marker identifies this local
release source. DGX sync also verifies that this directory is the actual Git
top-level before packaging `brain/`.
