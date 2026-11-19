"""Analyze Agent - Phân tích đề bài và code, map requirements với functions."""
import os
import json
import logging
from typing import List, Dict, Any
from openai import OpenAI
from src.services.llm_client_factory import build_llm_client

logger = logging.getLogger(__name__)


class AnalyzeAgent:
    """Agent phân tích đề bài và code, map requirements với functions."""
    
    def __init__(self, llm_provider: str = "open_ai"):
        """Initialize AnalyzeAgent with LLM client."""
        self.llm_client, self.model = build_llm_client(llm_provider)
        self.temperature = 0.3  # Lower for consistent analysis
    
    def analyze(
        self,
        assignment: str,
        functions_list: List[Dict],
        repository_structure: Dict
    ) -> Dict[str, Any]:
        """Phân tích và map requirements với functions.
        
        Args:
            assignment: Nội dung đề bài
            functions_list: List functions từ code analysis
            repository_structure: Full repository structure với content
            
        Returns:
            {
                "requirements": [
                    {
                        "requirement_text": "...",
                        "functions": [func_dict1, func_dict2, ...],
                        "idea": "Mô tả cách implement",
                        "inappropriate_code": "Code sai/chưa đủ và lý do",
                        "update_suggestion": "Cách cải thiện code",
                        "importance": "high/medium/low"
                    }
                ],
                "unmatched_functions": [func_dict, ...],
                "metadata": {
                    "total_requirements": 5,
                    "total_functions": 20,
                    "analysis_version": "2.0"
                }
            }
        """
        logger.info("🔍 AnalyzeAgent: Starting requirement-function mapping...")

        # Prepare functions summary for LLM
        if functions_list:
            functions_summary = "\n".join([
                f"- `{f['name']}({f['params']})` in {f['file']} line {f['line']}"
                for f in functions_list
            ])
        else:
            # No functions found — list files so the LLM can still reason about the project
            file_names = self._collect_file_names(repository_structure)
            functions_summary = (
                "(Không tìm thấy function definitions. Các file trong repo:)\n"
                + "\n".join(f"- {f}" for f in file_names[:30])
            )

        # Extract code snippets for inappropriate code detection
        code_snippets = self._extract_code_snippets(repository_structure, functions_list)
        
        prompt = f"""Bạn là chuyên gia phân tích code và đề bài. Phân tích code của sinh viên và map với requirements.

=== ĐỀ BÀI ===
{assignment[:4000]}

=== DANH SÁCH FUNCTIONS ===
{functions_summary}

=== CODE SNIPPETS (để phát hiện lỗi) ===
{code_snippets[:2000]}

=== YÊU CẦU OUTPUT ===
Trả về JSON với format:
{{
    "requirements": [
        {{
            "requirement_text": "Requirement từ đề bài (ngắn gọn)",
            "function_names": ["func1", "func2"],
            "idea": "Cách implement (1-2 câu)",
            "inappropriate_code": "Code sai/chưa tối ưu và LÝ DO (nếu có). Để trống nếu code OK",
            "update_suggestion": "Cách cải thiện: fix lỗi logic/syntax/convention/optimization",
            "importance": "high"
        }},
        ...
    ],
    "unmatched_functions": ["func_name1", ...]
}}

QUY TẮC PHÂN TÍCH:
1. **Requirement mapping**: Chỉ chọn functions CHÍNH liên quan đến requirement
2. **Importance**: 
   - "high" = yêu cầu bắt buộc/chính
   - "medium" = yêu cầu phụ
   - "low" = optional/bonus
3. **Inappropriate_code**: Kiểm tra:
   - ❌ Lỗi logic (sai thuật toán, sai điều kiện)
   - ❌ Lỗi syntax (compile error)
   - ❌ Vi phạm convention (naming, formatting)
   - ❌ Chưa tối ưu (complexity cao, redundant code)
   - ✅ Để TRỐNG nếu code OK
   Format: "Dòng X: [mô tả lỗi]. Lý do: [giải thích]"
4. **Update_suggestion**: Đưa ra cách FIX cụ thể:
   - Code example nếu cần
   - Best practices
   - Optimization techniques
5. **Unmatched functions**: Helper functions không thuộc requirement nào

Trả về ĐÚNG JSON format, không thêm text hay markdown."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a code analysis expert. Return only valid JSON without markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=3000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            if result_text.startswith("```json"):
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif result_text.startswith("```"):
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            mapping = json.loads(result_text)
            
            # Convert function names to full function objects
            func_dict = {f["name"]: f for f in functions_list}
            
            # Build result with new schema
            requirements_list = []
            for req_data in mapping.get("requirements", []):
                func_names = req_data.get("function_names", [])
                requirements_list.append({
                    "requirement_text": req_data.get("requirement_text", ""),
                    "functions": [func_dict[name] for name in func_names if name in func_dict],
                    "idea": req_data.get("idea", ""),
                    "inappropriate_code": req_data.get("inappropriate_code", ""),
                    "update_suggestion": req_data.get("update_suggestion", ""),
                    "importance": req_data.get("importance", "medium")
                })
            
            # Unmatched functions
            matched_names = set()
            for req in requirements_list:
                matched_names.update(f["name"] for f in req["functions"])
            
            unmatched_names = mapping.get("unmatched_functions", [])
            unmatched_functions = [
                func_dict[name] for name in unmatched_names if name in func_dict
            ]
            
            result = {
                "requirements": requirements_list,
                "unmatched_functions": unmatched_functions,
                "metadata": {
                    "total_requirements": len(requirements_list),
                    "total_functions": len(functions_list),
                    "matched_functions": len(matched_names),
                    "analysis_version": "2.0"
                }
            }
            
            logger.info(f"✅ AnalyzeAgent: Mapped {len(requirements_list)} requirements, {len(unmatched_functions)} unmatched")
            return result
            
        except Exception as e:
            logger.error(f"❌ AnalyzeAgent failed: {e}")
            # Fallback: return basic structure
            return {
                "requirements": [],
                "unmatched_functions": functions_list,
                "metadata": {
                    "total_requirements": 0,
                    "total_functions": len(functions_list),
                    "matched_functions": 0,
                    "analysis_version": "2.0",
                    "error": str(e)
                }
            }
    
    def _collect_file_names(self, repo_structure: Dict) -> List[str]:
        """Recursively collect all file paths from the repository tree."""
        files: List[str] = []

        def walk(node: Dict, path: str = "") -> None:
            if not isinstance(node, dict):
                return
            name = node.get("name", "")
            current = f"{path}/{name}" if path else name
            if node.get("type") in ("file", "blob"):
                files.append(current)
            for child in node.get("children", []):
                walk(child, current)

        walk(repo_structure)
        return files

    def _extract_code_snippets(self, repo_structure: Dict, functions_list: List[Dict]) -> str:
        """Extract code snippets around functions for error detection.
        
        Args:
            repo_structure: Repository structure with file contents
            functions_list: List of functions to extract
            
        Returns:
            Formatted string with code snippets
        """
        snippets = []
        
        # Group functions by file
        functions_by_file = {}
        for func in functions_list:
            file_path = func["file"]
            if file_path not in functions_by_file:
                functions_by_file[file_path] = []
            functions_by_file[file_path].append(func)
        
        # Extract snippets from each file
        def find_file_content(node: Dict, target_path: str, current_path: str = "") -> str:
            """Recursively find file content."""
            if not isinstance(node, dict):
                return ""
            
            node_name = node.get("name", "")
            full_path = f"{current_path}/{node_name}" if current_path else node_name
            
            if node.get("type") == "file" and full_path == target_path:
                return node.get("content", "")
            
            if node.get("type") == "folder":
                for child in node.get("children", []):
                    result = find_file_content(child, target_path, full_path)
                    if result:
                        return result
            
            return ""
        
        for file_path, funcs in functions_by_file.items():
            content = find_file_content(repo_structure, file_path)
            if not content:
                continue
            
            lines = content.split('\n')
            
            for func in funcs[:3]:  # Limit to 3 functions per file
                line_num = func["line"]
                start = max(0, line_num - 1)
                end = min(len(lines), line_num + 10)  # 10 lines of code
                
                snippet = '\n'.join(lines[start:end])
                snippets.append(f"// {func['name']} in {file_path}:{line_num}\n{snippet}\n")
        
        return '\n'.join(snippets[:5])  # Max 5 snippets total
