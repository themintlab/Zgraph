import os
import re

def rename_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return False # skip unreadable/binary files

    new_content = content
    # Replace ZGraph -> ZGraph
    new_content = re.sub(r'\bZNet\b', 'ZGraph', new_content)
    # Replace Zgraph -> Zgraph
    new_content = re.sub(r'\bZnet\b', 'Zgraph', new_content)
    # Replace zgraph -> zgraph
    new_content = re.sub(r'\bznet\b', 'zgraph', new_content)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    root_dir = r"c:\Users\wellandm\Code\Zgraph"
    
    # 1. Modify file contents
    modified_files = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '.git' in dirpath or '__pycache__' in dirpath:
            continue
        for f in filenames:
            file_path = os.path.join(dirpath, f)
            if rename_content(file_path):
                print(f"Updated content in: {file_path}")
                modified_files += 1
                
    # 2. Rename files and directories (bottom-up to avoid path invalidation)
    renamed_items = 0
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if '.git' in dirpath or '__pycache__' in dirpath:
            continue
            
        # Rename files
        for f in filenames:
            if 'zgraph' in f.lower():
                old_path = os.path.join(dirpath, f)
                new_name = f.replace('zgraph', 'zgraph').replace('Zgraph', 'Zgraph').replace('ZGraph', 'ZGraph')
                new_path = os.path.join(dirpath, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")
                renamed_items += 1
                
        # Rename directories
        for d in dirnames:
            if 'zgraph' in d.lower():
                old_path = os.path.join(dirpath, d)
                new_name = d.replace('zgraph', 'zgraph').replace('Zgraph', 'Zgraph').replace('ZGraph', 'ZGraph')
                new_path = os.path.join(dirpath, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed directory: {old_path} -> {new_path}")
                renamed_items += 1

    print(f"\nDone! Modified contents of {modified_files} files and renamed {renamed_items} files/directories.")

if __name__ == "__main__":
    main()
