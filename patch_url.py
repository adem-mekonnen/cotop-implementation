import json

path = "d:/cotop-implementation/research_implementation.ipynb"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = "https://github.com/YOUR_GITHUB_USERNAME/cotop-implementation.git"
replacement = "https://github.com/adem-mekonnen/cotop-implementation.git"

if target in content:
    content = content.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("URL updated successfully.")
else:
    print("Target URL not found.")
