# Pseudo-Brain

This is the canonical workspace for the persistent multi-thought sensorimotor
model research project.

**Start here:** [CURRENT_WORK.md](./CURRENT_WORK.md). That file says what the
current experiment is, what a pass would mean, and which document to open next.
The step-by-step operator commands are in
[brain/docs/OPERATOR_GUIDE.md](./brain/docs/OPERATOR_GUIDE.md).

The implementation, experiment configurations, tests, and research records are
under [brain](./brain/README.md). The Irene desktop/chat project is separate and
is not a dependency or training-data source for Pseudo-Brain.

The Python namespace `irene_brain` is retained temporarily for compatibility
with immutable historical checkpoints. New releases and runs are produced from
this project directory.

The checked-in `.pseudo-brain-workspace-v2` marker identifies this local
release source. DGX sync also verifies that this directory is the actual Git
top-level before packaging `brain/`.
