import re
import os

files = [
    "/Users/rtrevinnoc/maestria/tesis/research/experiments/exp_case1_full.py",
    "/Users/rtrevinnoc/maestria/tesis/research/experiments_alrawashdeh/exp_case1_full.py",
    "/Users/rtrevinnoc/maestria/tesis/research/nn/cnn/evaluate_model.py"
]

for f in files:
    if not os.path.exists(f): continue
    with open(f, "r") as file:
        content = file.read()
    
    # Remove suptitle calls
    content = re.sub(r'fig\d\.suptitle\([^)]+\)', '', content)
    # Also handle multiline suptitle calls
    content = re.sub(r'fig\d\.suptitle\([^)]+\n[^)]+\)', '', content)
    
    # Remove ax.set_title in evaluate_model
    content = re.sub(r"ax\.set_title\([^)]+\)", "", content)

    with open(f, "w") as file:
        file.write(content)
    print(f"Removed titles from {f}")
