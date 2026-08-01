import re
from typing import Dict, Any, List

# Regex matching bracket citation markers like [S1:02], [S2:00], stacked [S1:02][S3:01]
CITATION_REGEX = re.compile(r'\[S\d+:\d{2}\]')

class CitationVerifier:
    """
    Step 6 Post-Processor: Verifies every factual sentence in model output has a bracket citation marker.
    Measures marker drop rate and provides feedback for repair pass.
    """
    def __init__(self, citation_pattern: str = r'\[S\d+:\d{2}\]'):
        self.pattern = re.compile(citation_pattern)

    def extract_factual_sentences(self, output_text: str) -> List[str]:
        # Exclude metadata sections: "Confidence:", "Sources used:", "The provided sources do not address..."
        lines = output_text.split('\n')
        content_lines = []
        in_sources_section = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Confidence:") or stripped.startswith("Sources used:"):
                in_sources_section = True
                continue
            if in_sources_section:
                continue
            content_lines.append(stripped)

        full_content = " ".join(content_lines)
        sentences = re.split(r'(?<=[.!?])\s+', full_content)
        
        # Filter out empty or pure header lines
        factual_sentences = [
            s.strip() for s in sentences 
            if len(s.strip().split()) > 3 and not s.strip().startswith('#')
        ]
        return factual_sentences

    def verify_citations(self, output_text: str) -> Dict[str, Any]:
        factual_sentences = self.extract_factual_sentences(output_text)
        
        if not factual_sentences:
            return {
                "is_verified": True,
                "citation_density": 1.0,
                "drop_rate": 0.0,
                "total_sentences": 0,
                "cited_sentences_count": 0,
                "uncited_sentences": [],
                "all_markers_found": list(set(self.pattern.findall(output_text))),
                "status": "VERIFIED (NO FAILS)"
            }

        uncited_sentences = []
        cited_count = 0

        for sent in factual_sentences:
            # General connecting reasoning like "Therefore, X follows from Y" or explicit refusal messages
            # can be excluded if they don't contain factual claims, but strict rule checks for markers
            if self.pattern.search(sent):
                cited_count += 1
            else:
                # Check if it's explicit refusal or transition sentence
                if "do not address" in sent.lower() or "do not contain" in sent.lower() or "sources used" in sent.lower():
                    cited_count += 1
                else:
                    uncited_sentences.append(sent)

        total = len(factual_sentences)
        density = cited_count / total if total > 0 else 1.0
        drop_rate = (total - cited_count) / total if total > 0 else 0.0
        is_verified = (drop_rate == 0.0)

        all_markers = list(set(self.pattern.findall(output_text)))

        return {
            "is_verified": is_verified,
            "citation_density": round(density, 4),
            "drop_rate": round(drop_rate, 4),
            "total_sentences": total,
            "cited_sentences_count": cited_count,
            "uncited_sentences": uncited_sentences,
            "all_markers_found": all_markers,
            "status": "VERIFIED" if is_verified else ("PARTIAL" if cited_count > 0 else "UNVERIFIED")
        }

if __name__ == "__main__":
    sample_text = """
    Revenue grew 12% YoY [S1:00]. Operating margin reached 21.5% [S1:00].
    Apex spent $45 million on R&D.
    Confidence: [Fully supported]
    Sources used:
    - [S1] Apex Global Q3 Report
    """
    verifier = CitationVerifier()
    result = verifier.verify_citations(sample_text)
    print("Verification Result:", result)
