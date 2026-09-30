import os

with open("zgraph/src/zgraph/ARCHITECTURE.md", "r") as f:
    existing = f.read()

# Replace mentions of __call__ with evaluate where appropriate in the existing text
existing = existing.replace("They execute the __call__() pass", "They execute the `evaluate()` pass")
existing = existing.replace("the partition function via `.__call__()`", "the partition function via `evaluate()` (wrapped by `__call__()`)")
existing = existing.replace("their `__call__()` passes", "their `evaluate()` passes")
existing = existing.replace("used in the __call__() pass", "used in the `evaluate()` pass")
existing = existing.replace("the `__call__` signature", "the `evaluate` signature")
existing = existing.replace("inside a `__call__()` pass", "inside an `evaluate()` pass")

with open("zgraph/ARCHITECTURE.md", "r") as f:
    new_rules = f.read()
    
# Extract the core sections from the new rules
sections_to_add = new_rules.split("## 1. The Functor Pattern (`ZGraphNode`)")[1]

merged = existing + "\n\n## 4. The Functor Pattern (`ZGraphNode`)\n" + sections_to_add
merged = merged.replace("## 2. Universal Uncertainty Quantification (UQ)", "## 5. Universal Uncertainty Quantification (UQ)")

# Remove the temporary file
os.remove("zgraph/ARCHITECTURE.md")

with open("zgraph/src/zgraph/ARCHITECTURE.md", "w") as f:
    f.write(merged)
