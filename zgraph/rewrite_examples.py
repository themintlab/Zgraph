import json
import glob
import re
import os

files = glob.glob('examples/*.ipynb')
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    changed = False
    for cell in data.get('cells', []):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell.get('source', []):
                new_line = line
                # Replace beta_factor = R with beta_node = RT_node
                if 'beta_factor = R' in line or 'beta_factor=R' in line:
                    if not any('RT_node = FactorNode' in src for src in cell.get('source', [])) and not any('RT_node = FactorNode' in src for src in new_source):
                        new_source.append("T_node = SignalNode(0)\n")
                        new_source.append("RT_node = FactorNode([[R]], [T_node])\n")
                    new_line = re.sub(r'beta_factor\s*=\s*R', 'beta_node=RT_node', new_line)
                
                # Replace beta_factor = 0 with beta_node = zero_node
                if 'beta_factor = 0' in line or 'beta_factor=0' in line:
                    if not any('zero_node = ConstantNode' in src for src in cell.get('source', [])) and not any('zero_node = ConstantNode' in src for src in new_source):
                        new_source.append("zero_node = ConstantNode(0)\n")
                    new_line = re.sub(r'beta_factor\s*=\s*0', 'beta_node=zero_node', new_line)
                    
                # Replace beta_factor = 8.314 with beta_node = RT_node
                if 'beta_factor = 8.314' in line or 'beta_factor=8.314' in line:
                    if '1e-8*8.314' not in line:
                        if not any('RT_node = FactorNode' in src for src in cell.get('source', [])) and not any('RT_node = FactorNode' in src for src in new_source):
                            new_source.append("T_node = SignalNode(0)\n")
                            new_source.append("RT_node = FactorNode([[8.314]], [T_node])\n")
                        new_line = re.sub(r'beta_factor\s*=\s*8\.314', 'beta_node=RT_node', new_line)
                        
                # Replace beta_factor = 1e-8*8.314
                if '1e-8*8.314' in line:
                    if not any('small_node = ConstantNode' in src for src in cell.get('source', [])) and not any('small_node = ConstantNode' in src for src in new_source):
                        new_source.append("small_node = ConstantNode(1e-8*8.314)\n")
                    new_line = re.sub(r'beta_factor\s*=\s*1e-8\*8\.314', 'beta_node=small_node', new_line)

                new_source.append(new_line)
                if new_line != line:
                    changed = True
            cell['source'] = new_source
            
    if changed:
        with open(f, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=1)
            file.write('\n')
        print(f"Updated {f}")
