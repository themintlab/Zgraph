import os
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from zgraph import save
from zgraph.core import TropicalPolynomialNode, PiecewiseNode
from thermograph.nodes.sgte import SGTESingleNode
from typing import Dict, Any, List, Optional
import jax.numpy as jnp

class TDBParser:
    """
    A minimal CALPHAD TDB parser that extracts FUNCTION definitions and 
    builds hierarchical ZGraph nodes, preserving all physical dependencies 
    (like GHSERSI) as exact subgraphs for perfect gradient flow.
    """
    def __init__(self, tdb_text):
        self.functions = self._parse_functions(tdb_text)
        self.T = sympy.Symbol('T')
        self.terms = [1, self.T, self.T * sympy.log(self.T), self.T**2, self.T**-1, self.T**3, self.T**7, self.T**-9]
        self.node_registry = {}

    def _parse_functions(self, tdb_text):
        functions = {}
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

    def get_node(self, name):
        name = name.upper()
        if name in self.node_registry:
            return self.node_registry[name]
            
        if name not in self.functions:
            raise KeyError(f"Phase/Function '{name}' not found in TDB.")
            
        func = self.functions[name]
        
        subgraphs = []
        bounds = []
        for piece in func['pieces']:
            coeffs, deps = self._extract_coeffs(piece['expr'])
            base = SGTESingleNode(coeffs, T_index=0)
            
            if not deps:
                sub = base
            else:
                subs = [base]
                weights = [1.0]
                for dep_name, mult in deps.items():
                    subs.append(self.get_node(dep_name))
                    weights.append(mult)
                sub = TropicalPolynomialNode(M_matrix=jnp.expand_dims(jnp.array(weights, dtype=jnp.float32), 0), subgraph_list=subs)
                
            subgraphs.append(sub)
            bounds.append(piece['t_max'])
            
        bounds = bounds[:-1] # PiecewiseNode takes N-1 bounds
        
        if len(bounds) == 0:
            node = subgraphs[0]
        else:
            node = PiecewiseNode(bounds, subgraphs, signal_index=0)
            
        self.node_registry[name] = node
        return node


def parse_tdb(tdb_path: str, phases: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Reads a CALPHAD .tdb file and parses functional dependencies as exact ZGraph
    hierarchical subgraphs. Returns a dictionary of compiled nodes.
    """
    with open(tdb_path, 'r') as f:
        text = f.read()
        
    parser = TDBParser(text)
    
    if phases is None:
        phases = list(parser.functions.keys())
        
    nodes = {}
    for phase_name in phases:
        try:
            node = parser.get_node(phase_name)
            nodes[phase_name] = node
        except Exception as e:
            print(f"Warning: Failed to extract {phase_name} - {e}")
            
    return nodes


def save_library(nodes: Dict[str, Any], output_dir: str = None, library_name: str = "library"):
    """
    Saves a dictionary of ZGraph nodes to the specified output directory.
    If output_dir is not provided, defaults to LIBRARY_DIR / library_name.
    """
    from thermograph.config import LIBRARY_DIR
    
    if output_dir is None:
        output_dir = os.path.join(LIBRARY_DIR, library_name)
        
    os.makedirs(output_dir, exist_ok=True)
    
    saved = []
    for name, node in nodes.items():
        filepath = os.path.join(output_dir, f"{name}.zg")
        save(node, filepath)
        saved.append(name)
        
    print(f"Successfully serialized {len(saved)} nodes to {output_dir}")
    return saved


# Backwards compatibility wrapper
def extract_sgte_library(tdb_path: str, output_dir: str = None, phases: list = None):
    nodes = parse_tdb(tdb_path, phases)
    tdb_name = os.path.splitext(os.path.basename(tdb_path))[0]
    return save_library(nodes, output_dir, library_name=tdb_name)
