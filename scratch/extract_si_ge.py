def extract_functions(filename, prefixes):
    with open(filename, 'r') as f:
        content = f.read()
    
    blocks = content.split(" FUNCTION ")
    extracted = {}
    for block in blocks[1:]:
        func_name = block.split()[0].upper()
        match = any(func_name.startswith(pref) for pref in prefixes)
        if match:
            lines = []
            for line in block.split('\n'):
                if line.startswith('$'):
                    break
                lines.append(line.strip())
            extracted[func_name] = ' '.join(lines)
    return extracted

funcs = extract_functions("external_data/unary50.tdb", ['GHSERSI', 'GLIQSI', 'GHSERGE', 'GLIQGE'])
for k, v in funcs.items():
    print(f"{k}:")
    print(f"  {v.replace(';', ';\n  ')}\n")
