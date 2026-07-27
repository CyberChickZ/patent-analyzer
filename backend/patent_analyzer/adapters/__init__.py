"""Input adapters: paper text / manuscript / disclosure fields -> Doc.

Doc = {title, abstract, sections: [{title, paragraphs: [str], subsections: [...]}], kind}
"""

from patent_analyzer.adapters.paper import doc_from_sections, doc_from_text, iter_paragraphs, locate_marker, render_doc

__all__ = ["doc_from_text", "doc_from_sections", "render_doc", "locate_marker", "iter_paragraphs"]
