import re
import base64
import requests
from io import BytesIO
import matplotlib.pyplot as plt  # type: ignore
# import seaborn as sns
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import streamlit as st
import numpy as np
import pandas as pd
import time
from improved_code_display import get_improved_css_styles, create_enhanced_code_block, get_file_extension

# Enhanced output parser for code and images


class CodeBlock(BaseModel):
    language: str = Field(description="Programming language of the code")
    code: str = Field(description="The actual code content")
    explanation: Optional[str] = Field(
        description="Optional explanation of the code")


class ImageRequest(BaseModel):
    prompt: str = Field(description="Image generation prompt")
    style: Optional[str] = Field(
        description="Image style (realistic, cartoon, etc.)")
    size: Optional[str] = Field(
        description="Image size (256x256, 512x512, 1024x1024)")


class EnhancedResponse(BaseModel):
    text_content: str = Field(description="Main text response")
    code_blocks: List[CodeBlock] = Field(
        default=[], description="Code blocks in the response")
    image_requests: List[ImageRequest] = Field(
        default=[], description="Image generation requests")
    has_visualization: bool = Field(
        default=False, description="Whether response includes data visualization")


def parse_code_blocks(text: str) -> List[CodeBlock]:
    """Extract code blocks from markdown text"""
    code_blocks = []
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)

    for language, code in matches:
        if not language:
            language = "text"
        code_blocks.append(CodeBlock(
            language=language.lower(),
            code=code.strip(),
            explanation=None
        ))

    return code_blocks


def detect_image_requests(text: str) -> List[ImageRequest]:
    """Detect image generation requests in text"""
    image_requests = []

    # Common patterns for image requests
    image_patterns = [
        r"generate an? image of (.*?)(?:\.|$|,)",
        r"create an? image showing (.*?)(?:\.|$|,)",
        r"draw (.*?)(?:\.|$|,)",
        r"visualize (.*?)(?:\.|$|,)",
        r"show me an? image of (.*?)(?:\.|$|,)"
    ]

    for pattern in image_patterns:
        matches = re.findall(pattern, text.lower(), re.IGNORECASE)
        for match in matches:
            image_requests.append(ImageRequest(
                prompt=match.strip(),
                style="realistic",
                size="512x512"
            ))

    return image_requests


def generate_image_with_api(prompt: str, api_key: str = None) -> Optional[str]:
    """Generate image using external API (placeholder for your preferred service)"""
    try:
        # Example using DALL-E API (you'll need to implement based on your chosen service)
        # This is a placeholder - replace with your actual image generation service

        # For now, we'll create a simple matplotlib visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Generated Image:\n{prompt}",
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def create_data_visualization(data: Dict[str, Any], chart_type: str = "bar") -> Optional[str]:
    """Create data visualizations from structured data"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar" and "x" in data and "y" in data:
            ax.bar(data["x"], data["y"])
            ax.set_xlabel(data.get("xlabel", "X-axis"))
            ax.set_ylabel(data.get("ylabel", "Y-axis"))
            ax.set_title(data.get("title", "Bar Chart"))

        elif chart_type == "line" and "x" in data and "y" in data:
            ax.plot(data["x"], data["y"], marker='o')
            ax.set_xlabel(data.get("xlabel", "X-axis"))
            ax.set_ylabel(data.get("ylabel", "Y-axis"))
            ax.set_title(data.get("title", "Line Chart"))

        elif chart_type == "scatter" and "x" in data and "y" in data:
            ax.scatter(data["x"], data["y"])
            ax.set_xlabel(data.get("xlabel", "X-axis"))
            ax.set_ylabel(data.get("ylabel", "Y-axis"))
            ax.set_title(data.get("title", "Scatter Plot"))

        plt.tight_layout()

        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return None


def enhanced_response_parser(response_text: str, include_images: bool = True) -> EnhancedResponse:
    """Parse response text for code blocks, images, and visualizations"""

    # Extract code blocks
    code_blocks = parse_code_blocks(response_text)

    # Detect image requests
    image_requests = []
    if include_images:
        image_requests = detect_image_requests(response_text)

    # Check for data visualization needs
    has_visualization = any([
        "chart" in response_text.lower(),
        "graph" in response_text.lower(),
        "plot" in response_text.lower(),
        "visualization" in response_text.lower()
    ])

    return EnhancedResponse(
        text_content=response_text,
        code_blocks=code_blocks,
        image_requests=image_requests,
        has_visualization=has_visualization
    )


def format_code_for_display(code_block: CodeBlock) -> str:
    """Format code block for better display in Streamlit"""
    return f"""
### {code_block.language.upper()} Code:
```{code_block.language}
{code_block.code}
```
{code_block.explanation if code_block.explanation else ""}
"""


def display_enhanced_response(response_text: str, include_images: bool = True, api_key: str = None, message_id: str = None):
    """Display enhanced response with code highlighting and image generation"""

    # Add custom CSS for better code block styling
    # st.markdown(get_improved_css_styles(), unsafe_allow_html=True)
    st.markdown("""
    <style>
        /* Code block container */
        .stCodeBlock {
            background-color: #1E1E1E !important;
            border-radius: 8px !important;
            padding: 1rem !important;
            margin: 1rem 0 !important;
            border: 1px solid #333 !important;
        }
        .stCodeBlock pre {
            background-color: #1E1E1E !important;
            color: #D4D4D4 !important;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            padding: 0.5rem !important;
            margin: 0 !important;
            overflow-x: auto !important;
        }
        .stCodeBlock code {
            color: #D4D4D4 !important;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
        }
                
        /* Syntax highlighting colors */
        .stCodeBlock .k { color: #569CD6 !important; } /* Keywords */
        .stCodeBlock .n { color: #9CDCFE !important; } /* Names */
        .stCodeBlock .s { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .c { color: #6A9955 !important; } /* Comments */
        .stCodeBlock .o { color: #D4D4D4 !important; } /* Operators */
        .stCodeBlock .p { color: #D4D4D4 !important; } /* Punctuation */
        .stCodeBlock .mi { color: #B5CEA8 !important; } /* Numbers */
        .stCodeBlock .kc { color: #569CD6 !important; } /* Constants */
        .stCodeBlock .kd { color: #569CD6 !important; } /* Declarations */
        .stCodeBlock .kn { color: #569CD6 !important; } /* Keywords */
        .stCodeBlock .kp { color: #569CD6 !important; } /* Keywords */
        .stCodeBlock .kr { color: #569CD6 !important; } /* Keywords */
        .stCodeBlock .kt { color: #4EC9B0 !important; } /* Types */
        .stCodeBlock .nc { color: #4EC9B0 !important; } /* Classes */
        .stCodeBlock .no { color: #569CD6 !important; } /* Constants */
        .stCodeBlock .nd { color: #DCDCAA !important; } /* Decorators */
        .stCodeBlock .ni { color: #9CDCFE !important; } /* Names */
        .stCodeBlock .ne { color: #D7BA7D !important; } /* Exceptions */
        .stCodeBlock .nf { color: #DCDCAA !important; } /* Functions */
        .stCodeBlock .nl { color: #9CDCFE !important; } /* Names */
        .stCodeBlock .nn { color: #4EC9B0 !important; } /* Names */
        .stCodeBlock .nx { color: #9CDCFE !important; } /* Names */
        .stCodeBlock .py { color: #9CDCFE !important; } /* Python */
        .stCodeBlock .nt { color: #569CD6 !important; } /* Tags */
        .stCodeBlock .nv { color: #9CDCFE !important; } /* Variables */
        .stCodeBlock .ow { color: #D4D4D4 !important; } /* Operators */
        .stCodeBlock .w { color: #D4D4D4 !important; } /* Whitespace */
        .stCodeBlock .mb { color: #B5CEA8 !important; } /* Numbers */
        .stCodeBlock .mf { color: #B5CEA8 !important; } /* Numbers */
        .stCodeBlock .mh { color: #B5CEA8 !important; } /* Numbers */
        .stCodeBlock .mi { color: #B5CEA8 !important; } /* Numbers */
        .stCodeBlock .mo { color: #B5CEA8 !important; } /* Numbers */
        .stCodeBlock .sa { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .sb { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .sc { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .dl { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .sd { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .s2 { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .se { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .sh { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .si { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .sx { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .sr { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .s1 { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .ss { color: #CE9178 !important; } /* Strings */
        .stCodeBlock .bp { color: #9CDCFE !important; } /* Built-in */
        .stCodeBlock .fm { color: #DCDCAA !important; } /* Functions */
        .stCodeBlock .vc { color: #9CDCFE !important; } /* Variables */
        .stCodeBlock .vg { color: #9CDCFE !important; } /* Variables */
        .stCodeBlock .vi { color: #9CDCFE !important; } /* Variables */
        .stCodeBlock .vm { color: #9CDCFE !important; } /* Variables */
        .stCodeBlock .il { color: #B5CEA8 !important; } /* Numbers */
        
        /* Code block hover effect */
        .stCodeBlock:hover {
            box-shadow: 0 0 10px rgba(0,0,0,0.2) !important;
        }
        
        /* Code block scrollbar styling */
        .stCodeBlock pre::-webkit-scrollbar {
            height: 8px !important;
            width: 8px !important;
        }
        
        .stCodeBlock pre::-webkit-scrollbar-track {
            background: #1E1E1E !important;
        }
        
        .stCodeBlock pre::-webkit-scrollbar-thumb {
            background: #555 !important;
            border-radius: 4px !important;
        }
        
        .stCodeBlock pre::-webkit-scrollbar-thumb:hover {
            background: #666 !important;
        }
    </style>
    
    <script>
        function copyToClipboard(text) {
            navigator.clipboard.writeText(text).then(function() {
                // Success callback
                const button = document.querySelector('.copy-button');
                if (button) {
                    const originalText = button.textContent;
                    button.textContent = '✓ Copied!';
                    button.style.backgroundColor = '#4CAF50';
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.style.backgroundColor = '';
                    }, 2000);
                }
            }).catch(function(err) {
                console.error('Failed to copy text: ', err);
            });
        }
    </script>
    """, unsafe_allow_html=True)

    # Parse the response
    parsed_response = enhanced_response_parser(response_text, include_images)

    # Generate a unique timestamp for this display
    timestamp = str(int(time.time() * 1000))

    # Display main text content
    clean_text = response_text
    for code_block in parsed_response.code_blocks:
        clean_text = clean_text.replace(
            f"```{code_block.language}\n{code_block.code}\n```", "")

    st.markdown(clean_text)

    # Display code blocks with syntax highlighting
    if parsed_response.code_blocks:
        st.markdown("---")
        st.markdown("### 📝 Code Examples:")

        for i, code_block in enumerate(parsed_response.code_blocks):
            # Create unique keys using message_id, block index, and timestamp
            block_id = f"{message_id}_{i}_{timestamp}" if message_id else f"block_{i}_{timestamp}"

            # Use the enhanced code display
            # enhanced_html = create_enhanced_code_block(
            #     code=code_block.code,
            #     language=code_block.language,
            #     show_line_numbers=True
            # )

            with st.expander(f"{code_block.language.upper()} Code", expanded=True):
                # Display the code with syntax highlighting
                st.code(code_block.code, language=code_block.language)

                # Create columns for buttons
                col1, col2 = st.columns([1, 2])

                # Add copy button with unique key
                # with col1:
                #     if st.button(f"📋 Copy Code", key=f"copy_{block_id}", help="Click to copy code to clipboard"):
                #         # Use JavaScript to copy the code
                #         escaped_code = code_block.code.replace("`", "\\`")
                #         js_code = f"""
                #         <script>
                #             document.querySelector('[data-testid="stButton"]').classList.add('copy-button');
                #             copyToClipboard(`{escaped_code}`);
                #         </script>
                #         """
                #         st.markdown(js_code, unsafe_allow_html=True)

                # Add download button
                with col1:
                    st.download_button(
                        label=f"💾 Download",
                        data=code_block.code,
                        file_name=f"code.{get_file_extension(code_block.language)}",
                        mime="text/plain",
                        key=f"download_{block_id}"
                    )

                if code_block.explanation:
                    st.markdown(f"**💡 Explanation:** {code_block.explanation}")

    # Generate and display images
    if parsed_response.image_requests and include_images:
        st.markdown("---")
        st.markdown("### 🖼️ Generated Images:")

        for i, img_request in enumerate(parsed_response.image_requests):
            # Create unique keys for image expanders using timestamp
            img_id = f"{message_id}_{i}_{timestamp}" if message_id else f"img_{i}_{timestamp}"
            with st.expander(f"Image: {img_request.prompt[:50]}...", expanded=True):
                with st.spinner(f"Generating image..."):
                    image_base64 = generate_image_with_api(
                        img_request.prompt, api_key)

                    if image_base64:
                        st.image(
                            f"data:image/png;base64,{image_base64}",
                            caption=img_request.prompt,
                            use_column_width=True,
                            key=f"image_{img_id}"
                        )
                    else:
                        st.warning(
                            f"Could not generate image for: {img_request.prompt}")

    # Handle data visualizations
    if parsed_response.has_visualization:
        st.markdown("---")
        st.markdown("### 📊 Data Visualization:")
        st.info(
            "Visualization capabilities detected. You can implement custom charts here.")

# Example usage function for your agent


def process_agent_response(response_text: str, user_question: str) -> str:
    """Process agent response with enhanced parsing"""

    # Check if response contains code or image requests
    if any(marker in response_text for marker in ["```", "generate image", "create image", "visualize"]):
        # Store the enhanced response in session state
        if "enhanced_responses" not in st.session_state:
            st.session_state.enhanced_responses = {}

        # Generate a unique key for this response
        response_key = f"response_{len(st.session_state.enhanced_responses)}"
        st.session_state.enhanced_responses[response_key] = response_text

        # Use enhanced display with unique message ID
        display_enhanced_response(
            response_text, include_images=True, message_id=response_key)

        # Return a special marker that won't be displayed
        return f"__ENHANCED_RESPONSE__{response_key}"
    else:
        # Regular text response
        return response_text


def get_enhanced_response(key: str) -> str:
    """Retrieve a stored enhanced response"""
    if "enhanced_responses" in st.session_state and key in st.session_state.enhanced_responses:
        return st.session_state.enhanced_responses[key]
    return "Response not found"
