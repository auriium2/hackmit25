from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer
from pprint import pprint



artifacts_path = "models"

pipeline_options = PdfPipelineOptions() #artifacts_path=artifacts_path
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

source = "https://arxiv.org/pdf/2509.08461"  # file path or URL
doc = doc_converter.convert(source).document

serializer = MarkdownDocSerializer(doc=doc)
ser_result = serializer.serialize()
ser_text = ser_result.text

with open("output.md", "w", encoding="utf-8") as f:
    f.write(ser_text)
