import re
import json
from bs4 import BeautifulSoup

class MarkdownFilter:
    '''
    Class to filter Markdown content by removing HTML tags, code blocks, and links, and extracting plain text.

    Attributes:
        markdown_text (str): The original Markdown text to be filtered.
        links (list): List of extracted links from the Markdown text.
        filtered_text (str): The resulting plain text after filtering.

    Methods:
        __init__(markdown_text):
            初始化MarkdownFilter类，保存原始Markdown文本。

        extract_links():
            从Markdown文本中提取所有链接，并保存到self.links列表。

        remove_html_tags(html_content):
            移除HTML标签，仅保留纯文本内容。

        remove_code_blocks(text):
            移除Markdown中的代码块（```或~~~包裹的内容）。

        remove_markdown_links(text):
            移除Markdown格式的链接（[text](url)）。

        filter():
            执行所有过滤操作，提取链接并生成最终的纯文本内容，返回包含链接和过滤后文本的字典。
    '''
    '''Class to filter Markdown content by removing HTML tags, code blocks, and links, and extracting plain text.'''
    def __init__(self, markdown_text):
        self.markdown_text = markdown_text
        self.links = []
        self.filtered_text = ""

    def extract_links(self):
        link_pattern = r'\[.*?\]\((.*?)\)'
        self.links = re.findall(link_pattern, self.markdown_text)

    def remove_html_tags(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        cleaned_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        return cleaned_text

    def remove_code_blocks(self, text):
        code_block_pattern = r'```.*?```|~~~.*?~~~'
        return re.sub(code_block_pattern, '', text, flags=re.DOTALL)

    def remove_markdown_links(self, text):
        link_pattern = r'\[.*?\]\(.*?\)'
        return re.sub(link_pattern, '', text)

    def filter(self):
        self.extract_links()
        text_no_code = self.remove_code_blocks(self.markdown_text)
        text_no_links = self.remove_markdown_links(text_no_code)
        text_clean = self.remove_html_tags(text_no_links)
        text_clean = re.sub(r'[\n]+', '', text_clean)
        text_clean = re.sub(r'[\*\-\+#]', '', text_clean)
        text_clean = re.sub(r'\[[xX]\]', '', text_clean)
        text_clean = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1', text_clean)
        text_clean = re.sub(r'https?://[^\s]+', '', text_clean)
        # Remove image links (e.g., ![alt](url.png), ![alt](url.jpg), etc.)
        text_clean = re.sub(r'!\[.*?\]\((.*?\.(png|jpg|jpeg|gif|bmp|svg)[^\)]*)\)', '', text_clean, flags=re.IGNORECASE)
        text_clean = text_clean.strip()


        self.filtered_text = text_clean
        return {
            'links': self.links,
            'filtered_text': self.filtered_text
        }

if __name__ == "__main__":
    with open(r'3dcitydb-web-map-v2.0.0.json', 'r', encoding='utf-8') as f:
        md_content = json.load(f).get("folder_document_details", "")
    
    doc = ""
    for item in md_content:
        doc += item.get("content", "")
    md_filter = MarkdownFilter(doc)
    result = md_filter.filter()

    with open(r"/home/zyx/open_insight/Scripts/doc_extract/filtered_document.txt", "w+", encoding="utf-8") as f:
        f.write(result['filtered_text'])
