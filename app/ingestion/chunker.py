import re
from typing import List, Optional
from app.core.config import settings
from app.schemas.document import ParsedSection, DocumentChunk


class StructureAwareChunker:
    """
    Structure-aware semantic chunker implementing Section 8 of the Architecture Blueprint.
    Splits text along logical boundaries (paragraphs, sentences) while retaining section context,
    page numbers, and multi-tenant metadata.
    """

    def __init__(
        self,
        target_tokens: int = settings.CHUNK_SIZE_TOKENS,
        overlap_tokens: int = settings.CHUNK_OVERLAP_TOKENS,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimator (average 1 token ~ 4 chars / 0.75 words)."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def chunk_sections(
        self,
        sections: List[ParsedSection],
        organization_id: str,
        bot_id: str,
        document_id: str,
        source_type: str = "text"
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for section in sections:
            section_title = section.title or "General"
            page_num = section.page_number or 1
            content = section.content.strip()

            if not content:
                continue

            est_tokens = self.estimate_tokens(content)

            # If section fits within target token limit, keep it intact
            if est_tokens <= self.target_tokens:
                chunk_text = f"[{section_title}]\n{content}" if section_title != "General" else content
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}_chk_{global_chunk_idx:04d}",
                        document_id=document_id,
                        organization_id=organization_id,
                        bot_id=bot_id,
                        text=chunk_text,
                        page_number=page_num,
                        section_title=section_title,
                        chunk_index=global_chunk_idx,
                        token_count=self.estimate_tokens(chunk_text),
                        metadata={
                            **section.metadata,
                            "source_type": source_type,
                            "section_type": section.section_type
                        }
                    )
                )
                global_chunk_idx += 1
                continue

            # Larger section: split into paragraphs first
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
            current_chunk_paras: List[str] = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self.estimate_tokens(para)

                # If paragraph itself is excessively large, split by sentences
                if para_tokens > self.target_tokens:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    for sentence in sentences:
                        s_tokens = self.estimate_tokens(sentence)
                        if current_tokens + s_tokens > self.target_tokens and current_chunk_paras:
                            # Flush current chunk
                            raw_text = "\n\n".join(current_chunk_paras)
                            chunk_text = f"[{section_title}]\n{raw_text}" if section_title != "General" else raw_text
                            chunks.append(
                                DocumentChunk(
                                    chunk_id=f"{document_id}_chk_{global_chunk_idx:04d}",
                                    document_id=document_id,
                                    organization_id=organization_id,
                                    bot_id=bot_id,
                                    text=chunk_text,
                                    page_number=page_num,
                                    section_title=section_title,
                                    chunk_index=global_chunk_idx,
                                    token_count=self.estimate_tokens(chunk_text),
                                    metadata={
                                        **section.metadata,
                                        "source_type": source_type,
                                        "section_type": section.section_type
                                    }
                                )
                            )
                            global_chunk_idx += 1
                            # Retain overlap if possible
                            current_chunk_paras = [sentence]
                            current_tokens = s_tokens
                        else:
                            current_chunk_paras.append(sentence)
                            current_tokens += s_tokens
                    continue

                if current_tokens + para_tokens > self.target_tokens and current_chunk_paras:
                    # Flush current chunk
                    raw_text = "\n\n".join(current_chunk_paras)
                    chunk_text = f"[{section_title}]\n{raw_text}" if section_title != "General" else raw_text
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{document_id}_chk_{global_chunk_idx:04d}",
                            document_id=document_id,
                            organization_id=organization_id,
                            bot_id=bot_id,
                            text=chunk_text,
                            page_number=page_num,
                            section_title=section_title,
                            chunk_index=global_chunk_idx,
                            token_count=self.estimate_tokens(chunk_text),
                            metadata={
                                **section.metadata,
                                "source_type": source_type,
                                "section_type": section.section_type
                            }
                        )
                    )
                    global_chunk_idx += 1
                    current_chunk_paras = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk_paras.append(para)
                    current_tokens += para_tokens

            # Flush any remaining content
            if current_chunk_paras:
                raw_text = "\n\n".join(current_chunk_paras)
                chunk_text = f"[{section_title}]\n{raw_text}" if section_title != "General" else raw_text
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}_chk_{global_chunk_idx:04d}",
                        document_id=document_id,
                        organization_id=organization_id,
                        bot_id=bot_id,
                        text=chunk_text,
                        page_number=page_num,
                        section_title=section_title,
                        chunk_index=global_chunk_idx,
                        token_count=self.estimate_tokens(chunk_text),
                        metadata={
                            **section.metadata,
                            "source_type": source_type,
                            "section_type": section.section_type
                        }
                    )
                )
                global_chunk_idx += 1

        return chunks
