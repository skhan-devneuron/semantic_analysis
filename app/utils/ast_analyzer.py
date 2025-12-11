"""AST-based code relationship analyzer for tracking function calls, imports, and data flow"""
import ast
import re
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeRelationshipAnalyzer:
    """Analyzes code structure and relationships using AST"""
    
    def __init__(self, codebase_path: Optional[Path] = None):
        """
        Initialize analyzer.
        
        Args:
            codebase_path: Root path of codebase (for resolving imports)
        """
        self.codebase_path = codebase_path
        self.function_map: Dict[str, Dict] = {}  # function_name -> {file, line_start, line_end, code, calls, called_by}
        self.class_map: Dict[str, Dict] = {}  # class_name -> {file, methods, line_start, line_end}
        self.import_map: Dict[str, List[str]] = {}  # file -> [imported_modules]
        self.call_graph: Dict[str, Set[str]] = {}  # function_name -> {called_functions}
        self.file_contents: Dict[str, str] = {}  # file_path -> file_content
        
    def analyze_codebase(self, code_files: List[Path]) -> None:
        """
        Analyze entire codebase to build relationship maps.
        
        Args:
            code_files: List of code file paths to analyze
        """
        logger.info(f"Analyzing {len(code_files)} files for code relationships")
        
        for file_path in code_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                self.file_contents[str(file_path)] = content
                self._analyze_file(file_path, content)
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")
                continue
        
        # Build call graph after all files are analyzed
        self._build_call_graph()
        logger.info(f"Built relationship map: {len(self.function_map)} functions, {len(self.class_map)} classes")
    
    def _analyze_file(self, file_path: Path, content: str) -> None:
        """Analyze a single file"""
        if file_path.suffix != ".py":
            return
        
        try:
            tree = ast.parse(content)
            visitor = CodeVisitor(file_path, content)
            visitor.visit(tree)
            
            # Store functions
            for func_name, func_info in visitor.functions.items():
                full_name = f"{file_path}:{func_name}"
                self.function_map[full_name] = {
                    **func_info,
                    "file": str(file_path),
                    "calls": set(),
                    "called_by": set()
                }
            
            # Store classes
            for class_name, class_info in visitor.classes.items():
                full_name = f"{file_path}:{class_name}"
                self.class_map[full_name] = {
                    **class_info,
                    "file": str(file_path)
                }
            
            # Store imports
            self.import_map[str(file_path)] = visitor.imports
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
    
    def _build_call_graph(self) -> None:
        """Build call graph by analyzing function calls"""
        for func_name, func_info in self.function_map.items():
            file_path = func_info["file"]
            code = func_info.get("code", "")
            
            # Find all function calls in this function
            try:
                tree = ast.parse(code)
                call_visitor = CallVisitor()
                call_visitor.visit(tree)
                
                # Match called functions to known functions
                for called_name in call_visitor.calls:
                    # Try to find matching function
                    matching_func = self._find_function(called_name, file_path)
                    if matching_func:
                        func_info["calls"].add(matching_func)
                        if matching_func in self.function_map:
                            self.function_map[matching_func]["called_by"].add(func_name)
            except Exception as e:
                logger.debug(f"Error building call graph for {func_name}: {e}")
    
    def _find_function(self, name: str, current_file: str) -> Optional[str]:
        """Find function by name, checking current file and imports"""
        # Check current file first
        for func_name in self.function_map:
            if func_name.endswith(f":{name}") and func_name.startswith(current_file):
                return func_name
        
        # Check imports
        if current_file in self.import_map:
            # For now, just check if name exists anywhere
            for func_name in self.function_map:
                if func_name.endswith(f":{name}"):
                    return func_name
        
        return None
    
    def get_callers(self, function_name: str) -> List[Dict]:
        """Get all functions that call this function"""
        if function_name not in self.function_map:
            return []
        
        callers = []
        for caller_name in self.function_map[function_name]["called_by"]:
            if caller_name in self.function_map:
                caller_info = self.function_map[caller_name]
                callers.append({
                    "name": caller_name.split(":")[-1],
                    "file_path": caller_info["file"],
                    "line_start": caller_info["line_start"],
                    "line_end": caller_info["line_end"],
                    "code": caller_info.get("code", "")
                })
        
        return callers
    
    def get_callees(self, function_name: str) -> List[Dict]:
        """Get all functions called by this function"""
        if function_name not in self.function_map:
            return []
        
        callees = []
        for callee_name in self.function_map[function_name]["calls"]:
            if callee_name in self.function_map:
                callee_info = self.function_map[callee_name]
                callees.append({
                    "name": callee_name.split(":")[-1],
                    "file_path": callee_info["file"],
                    "line_start": callee_info["line_start"],
                    "line_end": callee_info["line_end"],
                    "code": callee_info.get("code", "")
                })
        
        return callees
    
    def get_related_functions(self, function_name: str, same_file: bool = True) -> List[Dict]:
        """Get related functions (same file, same class, etc.)"""
        if function_name not in self.function_map:
            return []
        
        func_info = self.function_map[function_name]
        file_path = func_info["file"]
        related = []
        
        for other_name, other_info in self.function_map.items():
            if other_name == function_name:
                continue
            
            if same_file and other_info["file"] != file_path:
                continue
            
            # Check if in same class (if function names suggest class membership)
            func_short = function_name.split(":")[-1]
            other_short = other_name.split(":")[-1]
            
            # Simple heuristic: functions in same file are related
            if other_info["file"] == file_path:
                related.append({
                    "name": other_short,
                    "file_path": other_info["file"],
                    "line_start": other_info["line_start"],
                    "line_end": other_info["line_end"],
                    "code": other_info.get("code", "")
                })
        
        return related
    
    def get_surrounding_context(self, file_path: str, line_start: int, line_end: int, context_lines: int = 10) -> str:
        """Get surrounding code context"""
        if file_path not in self.file_contents:
            return ""
        
        lines = self.file_contents[file_path].split('\n')
        start = max(0, line_start - context_lines - 1)
        end = min(len(lines), line_end + context_lines)
        
        return '\n'.join(lines[start:end])


class CodeVisitor(ast.NodeVisitor):
    """AST visitor to extract functions, classes, and imports"""
    
    def __init__(self, file_path: Path, content: str):
        self.file_path = file_path
        self.content = content
        self.functions: Dict[str, Dict] = {}
        self.classes: Dict[str, Dict] = {}
        self.imports: List[str] = []
    
    def visit_FunctionDef(self, node):
        """Extract function definition"""
        code = ast.get_source_segment(self.content, node) or ""
        self.functions[node.name] = {
            "line_start": node.lineno,
            "line_end": node.end_lineno or node.lineno,
            "code": code,
            "is_async": False
        }
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Extract async function definition"""
        code = ast.get_source_segment(self.content, node) or ""
        self.functions[node.name] = {
            "line_start": node.lineno,
            "line_end": node.end_lineno or node.lineno,
            "code": code,
            "is_async": True
        }
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        """Extract class definition"""
        code = ast.get_source_segment(self.content, node) or ""
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.classes[node.name] = {
            "line_start": node.lineno,
            "line_end": node.end_lineno or node.lineno,
            "code": code,
            "methods": methods
        }
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """Extract import statements"""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Extract from ... import statements"""
        module = node.module or ""
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}" if module else alias.name)
        self.generic_visit(node)


class CallVisitor(ast.NodeVisitor):
    """AST visitor to extract function calls"""
    
    def __init__(self):
        self.calls: Set[str] = set()
    
    def visit_Call(self, node):
        """Extract function call"""
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            self.calls.add(node.func.attr)
        self.generic_visit(node)


