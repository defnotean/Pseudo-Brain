import base64
from pathlib import Path

r = Path(__file__).resolve().parents[1]
z = r / "scripts" / "brain_src.zip"
b = base64.b64encode(z.read_bytes()).decode("ascii")
t = (r / "scripts" / "colab_train_a100.py").read_text("utf-8")

header = f"""import base64
from pathlib import Path
Path("/content/brain_src.zip").write_bytes(base64.b64decode('''{b}'''))
"""

out = r / "scripts" / "standalone_a100_train.py"
out.write_text(header + t, encoding="utf-8")
print(f"Created {out}, size {out.stat().st_size / (1024*1024):.2f} MB")
