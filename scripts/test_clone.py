import os
for root, dirs, files in os.walk("/content/Pseudo-Brain"):
    print(root, dirs[:5], files[:5])
    if len(root.split(os.sep)) > 5:
        break
