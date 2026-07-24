import os
import re

def main():
    root_dir = r"c:\Users\wellandm\Code\Zgraph"
    
    # 1. Find files and directories to rename
    dirs_to_rename = []
    files_to_rename = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".git" in dirpath:
            continue
            
        for d in dirnames:
            if "zgraph" in d.lower():
                dirs_to_rename.append(os.path.join(dirpath, d))
                
        for f in filenames:
            if "zgraph" in f.lower():
                files_to_rename.append(os.path.join(dirpath, f))

    # 2. Find files to modify (content)
    files_to_modify = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if ".git" in dirpath:
            continue
        for f in filenames:
            filepath = os.path.join(dirpath, f)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if re.search(r'zgraph', content, re.IGNORECASE):
                        files_to_modify.append(filepath)
            except Exception:
                pass # skip binary or unreadable files

    print("Directories to rename:")
    for d in dirs_to_rename:
        print(f"  - {d}")

    print("\nFiles to rename:")
    for f in files_to_rename:
        print(f"  - {f}")

    print(f"\nFiles to modify content ({len(files_to_modify)} files):")
    for f in files_to_modify:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
