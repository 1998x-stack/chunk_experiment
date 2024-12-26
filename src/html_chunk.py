from typing import List, Tuple, Optional, Any, Iterable, Dict, cast
import pathlib
import copy
from io import StringIO

# Placeholder implementations. Replace with actual implementations.
class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

class RecursiveCharacterTextSplitter:
    def __init__(self, **kwargs: Any):
        # Initialize with possible customization parameters
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        # Implement actual splitting logic here
        # For demonstration, return the documents as-is
        return documents

class HTMLSectionSplitter:
    """
    Splits HTML documents into sections based on specified header tags and manages metadata.
    
    Dependencies:
        - lxml
        - BeautifulSoup (bs4)
    """

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        xslt_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the HTMLSectionSplitter.

        Args:
            headers_to_split_on (List[Tuple[str, str]]): 
                A list of tuples where each tuple contains:
                    - The header tag to split on (e.g., 'h1', 'h2').
                    - The key name to associate with the header's text in metadata.
                Example: [("h1", "Title"), ("h2", "Section")]
            
            xslt_path (Optional[str]): 
                Path to an XSLT file for transforming HTML content.
                If not provided, a default XSLT file located at "xsl/converting_to_header.xslt" relative to this script is used.
            
            **kwargs (Any): 
                Additional keyword arguments for customizing the text splitter.
        """
        self.headers_to_split_on = dict(headers_to_split_on)

        if xslt_path is None:
            default_xslt = pathlib.Path(__file__).parent / "xsl" / "converting_to_header.xslt"
            self.xslt_path = default_xslt.absolute()
        else:
            self.xslt_path = pathlib.Path(xslt_path).absolute()

        self.kwargs = kwargs

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """
        Splits multiple documents into sections based on headers.

        Args:
            documents (Iterable[Document]): 
                An iterable of Document instances to be split.

        Returns:
            List[Document]: 
                A list of Document instances representing the split sections.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        split_docs = self.create_documents(texts, metadatas=metadatas)

        text_splitter = RecursiveCharacterTextSplitter(**self.kwargs)
        final_docs = text_splitter.split_documents(split_docs)

        return final_docs

    def split_text(self, text: str) -> List[Document]:
        """
        Splits a single HTML text string into sections.

        Args:
            text (str): The HTML content to be split.

        Returns:
            List[Document]: 
                A list of Document instances representing the split sections.
        """
        with StringIO(text) as file_like:
            return self.split_text_from_file(file_like)

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Creates Document instances from texts and their corresponding metadata.

        Args:
            texts (List[str]): 
                A list of HTML content strings to be split.
            
            metadatas (Optional[List[Dict[str, Any]]]): 
                A list of metadata dictionaries corresponding to each text.

        Returns:
            List[Document]: 
                A list of Document instances representing the split sections.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        else:
            # Ensure metadatas list matches texts list
            if len(metadatas) != len(texts):
                raise ValueError("Length of metadatas must match length of texts.")

        documents = []
        for text, metadata in zip(texts, metadatas):
            sections = self.split_text(text)
            for section in sections:
                section_metadata = copy.deepcopy(metadata)
                # Replace placeholder headers with actual metadata
                for key, value in section.metadata.items():
                    if value == "#TITLE#" and "Title" in metadata:
                        section.metadata[key] = metadata["Title"]
                combined_metadata = {**section_metadata, **section.metadata}
                new_doc = Document(
                    page_content=section.page_content,
                    metadata=combined_metadata
                )
                documents.append(new_doc)
        return documents

    def split_html_by_headers(self, html_doc: str) -> List[Dict[str, Optional[str]]]:
        """
        Splits an HTML document into sections based on specified header tags.

        Args:
            html_doc (str): The HTML content to be split.

        Returns:
            List[Dict[str, Optional[str]]]: 
                A list of dictionaries, each representing a section with:
                    - 'header': The header text or a default title for the first section.
                    - 'content': The content under the header.
                    - 'tag_name': The name of the header tag (e.g., 'h1', 'h2').
        """
        try:
            from bs4 import BeautifulSoup, Tag
        except ImportError:
            raise ImportError(
                "BeautifulSoup is not installed. Please install it using 'pip install beautifulsoup4'."
            )

        soup = BeautifulSoup(html_doc, "html.parser")
        header_tags = list(self.headers_to_split_on.keys())
        sections: List[Dict[str, Optional[str]]] = []

        # Find all headers and body tag
        headers = soup.find_all(header_tags + ["body"])

        for i, header in enumerate(headers):
            if isinstance(header, Tag):
                if i == 0:
                    current_header = "#TITLE#"
                    current_tag = "h1"
                else:
                    current_header = header.get_text(strip=True)
                    current_tag = header.name

                # Collect content until the next header
                content_elements = []
                for sibling in header.next_siblings:
                    if isinstance(sibling, Tag) and sibling.name in header_tags:
                        break
                    if isinstance(sibling, Tag):
                        content_elements.append(sibling.get_text(separator=" ", strip=True))
                    elif isinstance(sibling, str):
                        content_elements.append(sibling.strip())

                content = " ".join(filter(None, content_elements)).strip()

                if content:
                    sections.append({
                        "header": current_header,
                        "content": content,
                        "tag_name": current_tag,
                    })

        return sections

    def convert_possible_tags_to_header(self, html_content: str) -> str:
        """
        Converts specific HTML tags to headers using an XSLT transformation.

        Args:
            html_content (str): The HTML content to be transformed.

        Returns:
            str: The transformed HTML content.
        """
        try:
            from lxml import etree
        except ImportError:
            raise ImportError(
                "lxml is not installed. Please install it using 'pip install lxml'."
            )

        if not self.xslt_path.exists():
            raise FileNotFoundError(f"XSLT file not found at {self.xslt_path}")

        try:
            parser = etree.HTMLParser()
            tree = etree.parse(StringIO(html_content), parser)

            xslt_tree = etree.parse(str(self.xslt_path))
            transform = etree.XSLT(xslt_tree)
            result_tree = transform(tree)
            transformed_html = etree.tostring(result_tree, pretty_print=True, method="html").decode()
            return transformed_html
        except etree.XMLSyntaxError as e:
            raise ValueError(f"Invalid XML/HTML content: {e}") from e
        except etree.XSLTApplyError as e:
            raise ValueError(f"XSLT transformation failed: {e}") from e

    def split_text_from_file(self, file: Any) -> List[Document]:
        """
        Splits HTML content from a file-like object into Document sections.

        Args:
            file (Any): A file-like object containing HTML content.

        Returns:
            List[Document]: 
                A list of Document instances representing the split sections.
        """
        file_content = file.read()
        transformed_content = self.convert_possible_tags_to_header(file_content)
        sections = self.split_html_by_headers(transformed_content)

        documents = []
        for section in sections:
            metadata_key = self.headers_to_split_on.get(section["tag_name"], "Unknown")
            metadata_value = section["header"]
            document = Document(
                page_content=section["content"],
                metadata={
                    metadata_key: metadata_value
                }
            )
            documents.append(document)

        return documents

# Example Usage
if __name__ == "__main__":
    # Define headers to split on with their corresponding metadata keys
    headers = [("h1", "Title"), ("h2", "Section")]

    # Initialize the splitter
    splitter = HTMLSectionSplitter(headers_to_split_on=headers)

    # Example HTML content
    html_content = """
    <html>
        <body>
            <h1>Introduction</h1>
            <p>This is the introduction.</p>
            <h2>Background</h2>
            <p>Some background information.</p>
            <h2>Objectives</h2>
            <p>The objectives are listed here.</p>
        </body>
    </html>
    """

    # Create a Document instance
    doc = Document(page_content=html_content, metadata={"Title": "Sample Document"})

    # Split the document
    split_docs = splitter.split_documents([doc])

    # Print the split documents
    for idx, split_doc in enumerate(split_docs, 1):
        print(f"Document {idx}:")
        print(f"Metadata: {split_doc.metadata}")
        print(f"Content: {split_doc.page_content}\n")