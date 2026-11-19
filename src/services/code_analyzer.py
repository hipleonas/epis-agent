"""Code analyzer for extracting functions and classes from repository structure."""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# File extensions that contain parseable code
CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.java', '.cs', '.go', '.rb', '.php',
    '.cpp', '.c', '.h', '.hpp', '.kt', '.swift', '.vue',
}

# Exact filenames to skip entirely (no content to parse)
SKIP_FILENAMES = {
    'package.json', 'package-lock.json', 'yarn.lock',
    'pnpm-lock.yaml', 'pnpm-lock.yml', 'composer.lock',
    'Gemfile.lock', 'poetry.lock', 'Pipfile.lock',
    '.eslintrc', '.eslintrc.js', '.eslintrc.json', '.eslintrc.yml',
    '.prettierrc', '.prettierrc.js', '.prettierrc.json',
    '.babelrc', 'babel.config.js', 'babel.config.json',
    '.gitignore', '.gitattributes', '.editorconfig',
    'tsconfig.json', 'jsconfig.json', 'webpack.config.js',
    'vite.config.js', 'vite.config.ts', 'rollup.config.js',
    'jest.config.js', 'jest.config.ts', 'vitest.config.ts',
    'tailwind.config.js', 'tailwind.config.ts',
    'postcss.config.js', 'next.config.js', 'next.config.ts',
    'nuxt.config.js', 'nuxt.config.ts',
    'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
    '.env', '.env.example', '.env.local',
    'Makefile', 'README.md', 'CHANGELOG.md', 'LICENSE',
}

# File extensions to skip
SKIP_EXTENSIONS = {
    '.svg', '.ico', '.png', '.jpg', '.jpeg', '.gif', '.webp',
    '.mp4', '.mp3', '.wav', '.ogg',
    '.pdf', '.doc', '.docx',
    '.zip', '.tar', '.gz', '.rar',
    '.lock',       # any *.lock file
    '.map',        # source maps
    '.min',        # minified (catch .min.js via name check below)
    '.css', '.scss', '.sass', '.less',  # pure style files (no logic)
    '.json',       # config/data files (not code)
    '.yaml', '.yml',
    '.toml', '.ini', '.cfg', '.conf',
    '.txt', '.md', '.rst',
    '.d.ts',       # TypeScript declaration files
    '.snap',       # Jest snapshot files
}


def _should_skip(filename: str) -> bool:
    """Return True for files that should not be parsed for functions."""
    if filename in SKIP_FILENAMES:
        return True
    # .min.js, .min.ts, etc.
    if '.min.' in filename:
        return True
    # Check extension chain (handles .d.ts, .test.js, etc. partially)
    if '.' in filename:
        ext = '.' + filename.rsplit('.', 1)[-1].lower()
        if ext in SKIP_EXTENSIONS:
            return True
        # Double extension: .d.ts
        if filename.endswith('.d.ts') or filename.endswith('.spec.ts') or filename.endswith('.test.ts') \
                or filename.endswith('.spec.js') or filename.endswith('.test.js'):
            return True
    return False


def analyze_code_structure(repo_structure: Dict) -> Tuple[str, List[Dict]]:
    """Phân tích cấu trúc code để hiểu từng function/class làm gì.
    
    Args:
        repo_structure: Repository structure từ GitHub API.
        
    Returns:
        Tuple of (formatted_string, structured_functions_list)
        - formatted_string: Chuỗi mô tả chi tiết để hiển thị
        - structured_functions_list: List các dict chứa thông tin function
    """
    if not repo_structure or not isinstance(repo_structure, dict):
        return ("Không thể phân tích code", [])
    
    analysis = []
    functions_list = []  # List các function với metadata
    analysis.append("=== PHÂN TÍCH CHI TIẾT CODE ===")
    
    def analyze_file(node: Dict, path: str = "") -> None:
        """Phân tích từng file để tìm functions/classes."""
        if not isinstance(node, dict) or "name" not in node:
            return
        
        current_path = f"{path}/{node['name']}" if path else node['name']
        
        if node.get("type") == "folder":
            for child in node.get("children", []):
                analyze_file(child, current_path)
        
        elif node.get("type") == "file" or node.get("type") == "blob":
            name = node.get("name", "")

            # Skip noise / config / asset files
            if _should_skip(name):
                return

            # Only parse recognised code extensions
            ext = '.' + name.rsplit('.', 1)[-1].lower() if '.' in name else ''
            if ext not in CODE_EXTENSIONS:
                return

            content = node.get("content", "")
            if not content:
                return
            
            analysis.append(f"\n📄 File: {current_path}")
            
            # Phân tích Python
            if name.endswith('.py'):
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # Tìm class definitions
                    if line_stripped.startswith('class '):
                        class_name = line_stripped.split('(')[0].replace('class ', '').strip().rstrip(':')
                        analysis.append(f"  • Class '{class_name}' (dòng {i})")
                        
                        # Add to structured list
                        functions_list.append({
                            "type": "class",
                            "name": class_name,
                            "file": current_path,
                            "line": i,
                            "params": "",
                            "description": ""
                        })
                    
                    # Tìm function/method definitions
                    elif line_stripped.startswith('def '):
                        func_name = line_stripped.split('(')[0].replace('def ', '').strip()
                        params = line_stripped.split('(', 1)[1].split(')')[0] if '(' in line_stripped else ""
                        
                        # Tìm docstring của function
                        docstring = ""
                        if i < len(lines):
                            for j in range(i, min(i+5, len(lines))):
                                next_line = lines[j].strip()
                                if '"""' in next_line or "'''" in next_line:
                                    docstring = next_line.replace('"""', '').replace("'''", '').strip()
                                    break
                        
                        func_info = f"  • Hàm '{func_name}({params})' (dòng {i})"
                        if docstring:
                            func_info += f"\n    → Mục đích: {docstring[:100]}"
                        analysis.append(func_info)
                        
                        # Add to structured list
                        functions_list.append({
                            "type": "function",
                            "name": func_name,
                            "file": current_path,
                            "line": i,
                            "params": params,
                            "description": docstring[:100] if docstring else ""
                        })
                    
                    # Tìm route decorators (Flask/FastAPI)
                    elif '@app.route' in line_stripped or '@router.' in line_stripped:
                        route = line_stripped.split('(')[1].split(')')[0] if '(' in line_stripped else ""
                        analysis.append(f"  • Endpoint: {route} (dòng {i})")
                        
                        # Add to structured list
                        functions_list.append({
                            "type": "route",
                            "name": route,
                            "file": current_path,
                            "line": i,
                            "params": "",
                            "description": f"Flask endpoint {route}"
                        })
            
            # Phân tích JavaScript / TypeScript / JSX / TSX / Vue
            elif any(name.endswith(e) for e in ('.js', '.ts', '.jsx', '.tsx', '.vue')):
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # Tìm function declarations
                    if line_stripped.startswith('function '):
                        func_name = line_stripped.split('(')[0].replace('function ', '').strip()
                        params = line_stripped.split('(', 1)[1].split(')')[0] if '(' in line_stripped else ""
                        analysis.append(f"  • Hàm '{func_name}' (dòng {i})")
                        
                        # Add to structured list
                        functions_list.append({
                            "type": "function",
                            "name": func_name,
                            "file": current_path,
                            "line": i,
                            "params": params,
                            "description": ""
                        })
                    
                    # Tìm arrow functions và const functions
                    elif ' = ' in line_stripped and ('=>' in line_stripped or 'function(' in line_stripped):
                        func_name = line_stripped.split('=')[0].replace('const', '').replace('let', '').replace('var', '').strip()
                        analysis.append(f"  • Hàm '{func_name}' (dòng {i})")
                        
                        # Add to structured list
                        functions_list.append({
                            "type": "function",
                            "name": func_name,
                            "file": current_path,
                            "line": i,
                            "params": "",
                            "description": ""
                        })
            
            # Phân tích C/C++
            elif name.endswith('.cpp') or name.endswith('.c') or name.endswith('.h') or name.endswith('.hpp'):
                lines = content.split('\n')
                in_comment = False
                
                for i, line in enumerate(lines, 1):
                    line_stripped = line.strip()
                    
                    # Bỏ qua comments
                    if '/*' in line_stripped:
                        in_comment = True
                    if '*/' in line_stripped:
                        in_comment = False
                        continue
                    if in_comment or line_stripped.startswith('//'):
                        continue
                    
                    # Tìm function declarations/definitions
                    # Pattern: return_type function_name(params) or return_type function_name(params) {
                    if ('(' in line_stripped and 
                        ')' in line_stripped and 
                        not line_stripped.startswith('#') and
                        not line_stripped.startswith('if') and
                        not line_stripped.startswith('while') and
                        not line_stripped.startswith('for') and
                        not line_stripped.startswith('switch') and
                        ('void ' in line_stripped or 'int ' in line_stripped or 'double ' in line_stripped or 
                         'float ' in line_stripped or 'bool ' in line_stripped or 'char ' in line_stripped or
                         'long ' in line_stripped or 'string ' in line_stripped)):
                        
                        # Trích xuất tên hàm
                        try:
                            # Tìm phần trước dấu (
                            before_paren = line_stripped.split('(')[0]
                            # Lấy từ cuối cùng (tên hàm)
                            parts = before_paren.split()
                            if len(parts) >= 2:
                                func_name = parts[-1].replace('*', '').replace('&', '').strip()
                                params = line_stripped.split('(', 1)[1].split(')')[0] if '(' in line_stripped else ""
                                
                                # Tìm comment mô tả hàm (trước hàm)
                                description = ""
                                if i > 1:
                                    prev_line = lines[i-2].strip()
                                    if prev_line.startswith('//'):
                                        description = prev_line.replace('//', '').strip()
                                
                                func_info = f"  • Hàm '{func_name}({params})' (dòng {i})"
                                if description:
                                    func_info += f"\n    → {description}"
                                analysis.append(func_info)
                                
                                # Add to structured list
                                functions_list.append({
                                    "type": "function",
                                    "name": func_name,
                                    "file": current_path,
                                    "line": i,
                                    "params": params,
                                    "description": description
                                })
                        except:
                            pass
    
    try:
        analyze_file(repo_structure)
    except Exception as e:
        logger.error(f"Lỗi phân tích code: {e}")
        analysis.append(f"\n(Lỗi phân tích: {e})")
    
    if len(analysis) <= 1:  # Chỉ có header
        return ("Không tìm thấy functions/classes để phân tích", [])
    
    return ("\n".join(analysis), functions_list)
