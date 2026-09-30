import os
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from zgraph import save
from thermograph.nodes.sgte import SGTENode

class TDBParser:
    """
    A minimal CALPHAD TDB parser that extracts FUNCTION definitions and statically 
    evaluates nested dependencies into pure polynomial piecewise limits for ZGraph.
    """
    def __init__(self, tdb_text):
        self.functions = self._parse_functions(tdb_text)
        self.T = sympy.Symbol('T')
        self.terms = [1, self.T, self.T * sympy.log(self.T), self.T**2, self.T**-1, self.T**3, self.T**7, self.T**-9]

    def _parse_functions(self, tdb_text):
        functions = {}
        # Simple extraction of FUNCTION blocks (assuming they don't contain 'FUNCTION' internally)
        blocks = re.split(r'\bFUNCTION\b', tdb_text, flags=re.IGNORECASE)[1:]
        
        for block in blocks:
            block = block.split("!")[0]
            segments = block.split(";")
            
            first_seg = segments[0].strip()
            parts = first_seg.split(None, 2)
            if len(parts) < 2: continue
            name = parts[0].upper()
            low_T = float(parts[1])
            
            expr = parts[2] if len(parts) > 2 else ""
            
            pieces = []
            for i in range(len(segments) - 1):
                if i > 0:
                    expr = segments[i].strip()
                
                next_seg = segments[i+1].strip()
                if not next_seg: continue
                
                ns_parts = next_seg.split(None, 2)
                t_max = float(ns_parts[0])
                
                if len(ns_parts) > 1 and ns_parts[1].upper() in ['Y', 'N']:
                    segments[i+1] = ns_parts[2] if len(ns_parts) > 2 else ""
                else:
                    segments[i+1] = ns_parts[1] if len(ns_parts) > 1 else ""
                    
                pieces.append({'t_max': t_max, 'expr': expr.strip()})
                
            functions[name] = {
                'low_T': low_T,
                'pieces': pieces
            }
        return functions

    def _extract_coeffs(self, expr_str):
        expr_str = expr_str.replace('LN(T)', 'log(T)').replace('\n', '')
        expr = parse_expr(expr_str)
        expr = sympy.expand(expr)
        
        coeffs_dict = expr.as_coefficients_dict()
        coeffs = []
        for term in self.terms:
            c = coeffs_dict.get(term, 0.0)
            coeffs.append(float(c) if hasattr(c, '__float__') else float(c))
            
        deps = {}
        for term, c in coeffs_dict.items():
            if term not in self.terms and term != 1:
                deps[str(term).upper()] = float(c)
        return coeffs, deps

    def _get_piece(self, name, T_val):
        func = self.functions[name]
        for piece in func['pieces']:
            if T_val <= piece['t_max']:
                return piece
        return func['pieces'][-1]

    def build_phase(self, name):
        name = name.upper()
        if name not in self.functions:
            raise KeyError(f"Phase/Function '{name}' not found in TDB.")
            
        breakpoints = set()
        
        def collect_breakpoints(n):
            func = self.functions[n]
            breakpoints.add(func['low_T'])
            for piece in func['pieces']:
                breakpoints.add(piece['t_max'])
                coeffs, deps = self._extract_coeffs(piece['expr'])
                for dep in deps:
                    collect_breakpoints(dep)
        
        collect_breakpoints(name)
        breakpoints = sorted(list(breakpoints))
        
        compiled_pieces = []
        for i in range(len(breakpoints)-1):
            t_min = breakpoints[i]
            t_max = breakpoints[i+1]
            t_mid = (t_min + t_max) / 2.0
            
            def resolve_coeffs(n, t):
                piece = self._get_piece(n, t)
                base_coeffs, deps = self._extract_coeffs(piece['expr'])
                
                final_coeffs = list(base_coeffs)
                for dep_name, mult in deps.items():
                    dep_coeffs = resolve_coeffs(dep_name, t)
                    for j in range(8):
                        final_coeffs[j] += mult * dep_coeffs[j]
                return final_coeffs
                
            c = resolve_coeffs(name, t_mid)
            compiled_pieces.append((t_max, c))
            
        return compiled_pieces


def extract_sgte_library(tdb_path: str, output_dir: str, phases: list = None):
    """
    Reads a CALPHAD .tdb file, recursively resolves functional dependencies into 
    flat SGTE polynomials, constructs ZGraph SGTENodes, and serializes them to 
    the target output library directory.

    Args:
        tdb_path: Path to the .tdb file.
        output_dir: Path to the library folder to save .zg archives.
        phases: List of specific FUNCTION names to extract (e.g. ['GHSERSI', 'GLIQSI']). 
                If None, extracts ALL available functions.
    """
    with open(tdb_path, 'r') as f:
        text = f.read()
        
    parser = TDBParser(text)
    
    if phases is None:
        phases = list(parser.functions.keys())
        
    os.makedirs(output_dir, exist_ok=True)
    
    extracted = []
    for phase_name in phases:
        try:
            poly_data = parser.build_phase(phase_name)
            node = SGTENode(poly_data, T_index=0)
            
            filepath = os.path.join(output_dir, f"{phase_name}.zg")
            save(node, filepath)
            extracted.append(phase_name)
        except Exception as e:
            print(f"Warning: Failed to extract {phase_name} - {e}")
            
    print(f"Successfully serialized {len(extracted)} SGTE nodes to {output_dir}")
    return extracted
