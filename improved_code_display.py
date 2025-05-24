import streamlit as st
import re
from typing import List, Dict, Any
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter
from pygments.util import ClassNotFound

def get_improved_css_styles():
    """Enhanced CSS with better syntax highlighting and readability"""
    return """
    <style>
        /* Enhanced code block container */
        .enhanced-code-block {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-radius: 12px;
            padding: 0;
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid #404040;
            overflow: hidden;
        }
        
        /* Code header with language and actions */
        .code-header {
            background: linear-gradient(90deg, #404040 0%, #505050 100%);
            padding: 12px 20px;
            border-bottom: 1px solid #555;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .code-language {
            color: #00d4ff;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .code-actions {
            display: flex;
            gap: 10px;
        }
        
        /* Enhanced code content */
        .code-content {
            padding: 20px;
            background: #1e1e1e;
            overflow-x: auto;
            # color: white;
        }
        
        .code-content pre {
            margin: 0 !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            font-family: 'JetBrains Mono', 'Fira Code', 'Monaco', 'Consolas', monospace !important;
            font-size: 14px !important;
            line-height: 1.6 !important;
            color: #d4d4d4 !important;
        }
        
        /* Line numbers */
        .code-with-lines {
            display: table;
            width: 100%;
        }
        
        .code-line {
            display: table-row;
        }
        
        .line-number {
            display: table-cell;
            color: #666;
            text-align: right;
            padding-right: 15px;
            user-select: none;
            width: 40px;
            border-right: 1px solid #333;
            margin-right: 15px;
        }
        
        .line-content {
            display: table-cell;
            padding-left: 15px;
            width: 100%;
        }
        
        /* Syntax highlighting improvements */
        .highlight {
            background: transparent !important;
        }
        
        /* Keywords */
        .highlight .k, .highlight .kc, .highlight .kd, .highlight .kn, 
        .highlight .kp, .highlight .kr, .highlight .kt { 
            color: #569cd6 !important; 
            font-weight: normal !important;
        }
        
        /* Strings */
        .highlight .s, .highlight .s1, .highlight .s2, .highlight .sa,
        .highlight .sb, .highlight .sc, .highlight .sd, .highlight .se,
        .highlight .sh, .highlight .si, .highlight .sx, .highlight .sr,
        .highlight .ss { 
            color: #ce9178 !important; 
        }
        
        /* Comments */
        .highlight .c, .highlight .c1, .highlight .cm, .highlight .cp,
        .highlight .cs { 
            color: #6a9955 !important; 
            font-style: italic !important;
        }
        
        /* Functions */
        .highlight .nf, .highlight .fm { 
            color: #dcdcaa !important; 
        }
        
        /* Classes */
        .highlight .nc, .highlight .nn { 
            color: #4ec9b0 !important; 
        }
        
        /* Numbers */
        .highlight .m, .highlight .mi, .highlight .mf, .highlight .mh,
        .highlight .mo, .highlight .mb, .highlight .il { 
            color: #b5cea8 !important; 
        }
        
        /* Variables and identifiers */
        .highlight .n, .highlight .na, .highlight .nb, .highlight .nc,
        .highlight .no, .highlight .nd, .highlight .ni, .highlight .ne,
        .highlight .nv, .highlight .vc, .highlight .vg, .highlight .vi,
        .highlight .vm { 
            color: #9cdcfe !important; 
        }
        
        /* Operators */
        .highlight .o, .highlight .ow { 
            color: #d4d4d4 !important; 
        }
        
        /* Punctuation */
        .highlight .p { 
            color: #d4d4d4 !important; 
        }
        
        /* Error highlighting */
        .highlight .err { 
            color: #f44747 !important; 
            background: transparent !important;
        }
        
        /* Copy button styling */
        .copy-button {
            background: #007acc;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        
        .copy-button:hover {
            background: #005a9e;
            transform: translateY(-1px);
        }
        
        /* Scrollbar styling */
        .code-content::-webkit-scrollbar {
            height: 8px;
        }
        
        .code-content::-webkit-scrollbar-track {
            background: #2d2d2d;
        }
        
        .code-content::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        
        .code-content::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
    </style>
    """

def detect_language_better(code: str, declared_language: str = None) -> str:
    """Improved language detection with fallbacks"""
    if declared_language and declared_language.lower() != "text":
        return declared_language.lower()
    
    # Language detection patterns
    patterns = {
        'python': [r'def\s+\w+\(', r'import\s+\w+', r'from\s+\w+\s+import', r'if\s+__name__\s*==\s*["\']__main__["\']'],
        'javascript': [r'function\s+\w+\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'var\s+\w+\s*=', r'=>'],
        'java': [r'public\s+class\s+\w+', r'public\s+static\s+void\s+main', r'System\.out\.print'],
        'cpp': [r'#include\s*<', r'int\s+main\s*\(', r'std::', r'cout\s*<<'],
        'c': [r'#include\s*<stdio\.h>', r'int\s+main\s*\(', r'printf\s*\('],
        'go': [r'package\s+main', r'func\s+main\s*\(', r'fmt\.Print', r'import\s*\('],
        'rust': [r'fn\s+main\s*\(', r'let\s+mut\s+', r'println!'],
        'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO'],
        'html': [r'<html', r'<div', r'<script', r'<!DOCTYPE'],
        'css': [r'\{[^}]*\}', r'@media', r'\.[\w-]+\s*\{'],
        'bash': [r'#!/bin/bash', r'\$\w+', r'echo\s+'],
        'json': [r'^\s*\{', r'^\s*\[', r'"\w+":\s*'],
        'yaml': [r'^\s*\w+:\s*', r'^\s*-\s+'],
        'xml': [r'<\?xml', r'<\w+.*?>', r'</\w+>']
    }
    
    code_lower = code.lower()
    for lang, lang_patterns in patterns.items():
        if any(re.search(pattern, code_lower, re.MULTILINE) for pattern in lang_patterns):
            return lang
    
    # Fallback to pygments detection
    try:
        lexer = guess_lexer(code)
        return lexer.aliases[0] if lexer.aliases else 'text'
    except ClassNotFound:
        return 'text'

def add_line_numbers(code: str) -> str:
    """Add line numbers to code"""
    lines = code.split('\n')
    numbered_lines = []
    
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f'<div class="code-line">'
                            f'<span class="line-number">{i:2d}</span>'
                            f'<span class="line-content">{line}</span>'
                            f'</div>')
    
    return f'<div class="code-with-lines">{"".join(numbered_lines)}</div>'

def highlight_code_with_pygments(code: str, language: str) -> str:
    """Use Pygments for better syntax highlighting"""
    try:
        lexer = get_lexer_by_name(language, stripnl=False)
        formatter = HtmlFormatter(
            style='monokai',
            noclasses=False,
            cssclass='highlight',
            linenos=False
        )
        return highlight(code, lexer, formatter)
    except ClassNotFound:
        # Fallback to simple HTML escaping
        return f'<pre><code>{code}</code></pre>'

def create_enhanced_code_block(code: str, language: str = None, title: str = None, 
                             show_line_numbers: bool = True, collapsible: bool = False) -> str:
    """Create an enhanced, highly readable code block"""
    
    # Detect language if not provided
    detected_language = detect_language_better(code, language)
    display_language = detected_language.upper()
    
    # Generate syntax highlighted code
    highlighted_code = highlight_code_with_pygments(code, detected_language)
    
    # Add line numbers if requested
    if show_line_numbers:
        # Extract the code content from the highlighted HTML
        code_content = re.search(r'<pre[^>]*>(.*?)</pre>', highlighted_code, re.DOTALL)
        if code_content:
            clean_code = re.sub(r'<[^>]+>', '', code_content.group(1))
            numbered_code = add_line_numbers(clean_code)
            highlighted_code = highlighted_code.replace(code_content.group(1), numbered_code)
    
    # Create the complete code block HTML
    block_title = title or f"{display_language} Code"
    
    html_content = f"""
    <div class="enhanced-code-block">
        <div class="code-header">
            <span class="code-language">{display_language}</span>
            <div class="code-actions">
                <button class="copy-button" onclick="copyCodeToClipboard(this)">📋 Copy</button>
            </div>
        </div>
        <div class="code-content">
            {highlighted_code}
        </div>
    </div>
    
    <script>
    function copyCodeToClipboard(button) {{
        const codeBlock = button.closest('.enhanced-code-block');
        const codeContent = codeBlock.querySelector('.code-content');
        const code = codeContent.innerText || codeContent.textContent;
        
        navigator.clipboard.writeText(code).then(function() {{
            const originalText = button.innerHTML;
            button.innerHTML = '✅ Copied!';
            button.style.background = '#28a745';
            setTimeout(function() {{
                button.innerHTML = originalText;
                button.style.background = '#007acc';
            }}, 2000);
        }}).catch(function(err) {{
            console.error('Failed to copy: ', err);
            button.innerHTML = '❌ Failed';
            setTimeout(function() {{
                button.innerHTML = '📋 Copy';
            }}, 2000);
        }});
    }}
    </script>
    """
    
    return html_content

def display_code_with_explanation(code: str, language: str = None, 
                                explanation: str = None, title: str = None):
    """Display code with optional explanation in Streamlit"""
    
    # Apply enhanced CSS
    st.markdown(get_improved_css_styles(), unsafe_allow_html=True)
    
    # Show explanation first if provided
    if explanation:
        st.markdown(f"**💡 Explanation:** {explanation}")
        st.markdown("---")
    
    # Display the enhanced code block
    enhanced_html = create_enhanced_code_block(
        code=code,
        language=language,
        title=title,
        show_line_numbers=True
    )
    
    st.markdown(enhanced_html, unsafe_allow_html=True)

def create_code_comparison(code1: str, code2: str, lang1: str = None, lang2: str = None,
                          title1: str = "Before", title2: str = "After"):
    """Display two code blocks side by side for comparison"""
    
    st.markdown(get_improved_css_styles(), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {title1}")
        enhanced_html1 = create_enhanced_code_block(code1, lang1, title1, show_line_numbers=True)
        st.markdown(enhanced_html1, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"### {title2}")
        enhanced_html2 = create_enhanced_code_block(code2, lang2, title2, show_line_numbers=True)
        st.markdown(enhanced_html2, unsafe_allow_html=True)

def create_interactive_code_demo(code: str, language: str, description: str = None):
    """Create an interactive code demonstration with execution capability"""
    
    st.markdown(get_improved_css_styles(), unsafe_allow_html=True)
    
    if description:
        st.markdown(f"**📖 Description:** {description}")
    
    # Display the code
    enhanced_html = create_enhanced_code_block(code, language, show_line_numbers=True)
    st.markdown(enhanced_html, unsafe_allow_html=True)
    
    # Add execution button for Python
    if language.lower() == 'python':
        if st.button("▶️ Run This Code", key=f"run_{hash(code)}"):
            try:
                # Capture output
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()
                
                # Execute the code
                exec(code)
                
                # Get the output
                sys.stdout = old_stdout
                output = captured_output.getvalue()
                
                if output:
                    st.success("**Output:**")
                    st.code(output, language="text")
                else:
                    st.success("Code executed successfully (no output)")
                    
            except Exception as e:
                st.error(f"**Error:** {str(e)}")

# Example usage functions
def enhanced_display_example():
    """Example of how to use the enhanced code display"""
    
    sample_code = '''def fibonacci(n):
    """Generate Fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

# Generate first 10 Fibonacci numbers
fib_numbers = fibonacci(10)
print("First 10 Fibonacci numbers:", fib_numbers)'''
    
    display_code_with_explanation(
        code=sample_code,
        language="python",
        explanation="This function generates the Fibonacci sequence using an iterative approach for better performance.",
        title="Fibonacci Generator"
    )

def get_file_extension(language: str) -> str:
    """Get appropriate file extension for a programming language"""
    extensions = {
        'python': 'py',
        'javascript': 'js',
        'typescript': 'ts',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'csharp': 'cs',
        'go': 'go',
        'rust': 'rs',
        'html': 'html',
        'css': 'css',
        'sql': 'sql',
        'bash': 'sh',
        'powershell': 'ps1',
        'yaml': 'yml',
        'json': 'json',
        'xml': 'xml',
        'markdown': 'md'
    }
    return extensions.get(language.lower(), 'txt')
