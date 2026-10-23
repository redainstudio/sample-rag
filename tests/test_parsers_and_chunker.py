import pytest
from app.ingestion.text import TextParser
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.service import DocumentIngestionService
from app.schemas.document import IngestionRequest, ParsedSection


def test_text_parser_markdown():
    parser = TextParser()
    md_content = """# Company Overview
Botzi is a next-gen AI bot platform for websites.
It empowers teams to create grounded assistants.

## Shipping & Returns
Standard delivery takes 2 to 4 business days.
Returns are accepted within 30 days of purchase.
"""
    sections = parser.parse_text(md_content, filename="overview.md")
    assert len(sections) == 2
    assert sections[0].title == "Company Overview"
    assert "Botzi is a next-gen" in sections[0].content
    assert sections[1].title == "Shipping & Returns"
    assert "Standard delivery takes 2 to 4" in sections[1].content


def test_structure_aware_chunker():
    chunker = StructureAwareChunker(target_tokens=50, overlap_tokens=10)
    sections = [
        ParsedSection(
            title="Refund Policy",
            content="Customers can request a refund within 30 days of purchase. The item must be in original condition with tags attached. Please contact support with order number.",
            page_number=3,
            section_type="text"
        )
    ]
    chunks = chunker.chunk_sections(
        sections=sections,
        organization_id="org_test",
        bot_id="bot_test",
        document_id="doc_100",
        source_type="md"
    )
    assert len(chunks) >= 1
    first_chunk = chunks[0]
    assert first_chunk.organization_id == "org_test"
    assert first_chunk.bot_id == "bot_test"
    assert first_chunk.document_id == "doc_100"
    assert first_chunk.section_title == "Refund Policy"
    assert first_chunk.page_number == 3
    assert "Refund Policy" in first_chunk.text


def test_document_ingestion_service_text():
    service = DocumentIngestionService()
    req = IngestionRequest(
        organization_id="org_abc",
        bot_id="bot_xyz",
        document_id="doc_001",
        filename="faq.txt",
        content_text="Q: What are your operating hours?\nA: We are open Monday through Friday 9 AM to 6 PM."
    )
    chunks, progress = service.process_document(req)
    assert progress.status.value == "READY"
    assert len(chunks) == 1
    assert chunks[0].document_id == "doc_001"
    assert "operating hours" in chunks[0].text
