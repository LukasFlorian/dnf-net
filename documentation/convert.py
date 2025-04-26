from spire.doc import *
from spire.doc.common import *

# Load the Markdown file
doc = Document()
doc.LoadFromFile("your_markdown_file.md")

# Save as PDF
doc.SaveToFile("output.pdf", FileFormat.PDF)