"""
Indian Legal Regex Patterns
============================
Regex patterns for extracting legal entities, citations, and statutory references
from Indian court judgments.

Based on OpenNyAI specifications for Indian Legal NLP.
"""

import re
from typing import Dict, List, Pattern, Tuple


class IndianLegalPatterns:
    """
    Collection of regex patterns for Indian legal text processing.
    
    Covers:
    - Case citations (AIR, SCC, SCR, SCALE, etc.)
    - Statutory references (IPC, CrPC, Acts, Sections)
    - Party name extraction
    - Case number formats
    - Date patterns in Indian format
    """
    
    # =========================================
    # CITATION PATTERNS
    # =========================================
    
    # AIR Citations: AIR 1973 SC 1461, 1973 AIR SC 1461
    AIR_CITATION = re.compile(
        r'(?:AIR\s+)?(\d{4})\s+AIR\s+(SC|[A-Z]{2,3})\s+(\d+)|'
        r'AIR\s+(\d{4})\s+(SC|[A-Z]{2,3})\s+(\d+)',
        re.IGNORECASE
    )
    
    # SCC Citations: (2019) 5 SCC 123, 2019 (5) SCC 123
    SCC_CITATION = re.compile(
        r'\(?\s*(\d{4})\s*\)?\s*\(?\s*(\d+)\s*\)?\s*SCC\s+(\d+)',
        re.IGNORECASE
    )
    
    # SCR Citations: 1973 SCR (1) 461
    SCR_CITATION = re.compile(
        r'(\d{4})\s+SCR\s*\(\s*(\d+)\s*\)\s*(\d+)',
        re.IGNORECASE
    )
    
    # SCALE Citations: (2020) 3 SCALE 456
    SCALE_CITATION = re.compile(
        r'\(?\s*(\d{4})\s*\)?\s*\(?\s*(\d+)\s*\)?\s*SCALE\s+(\d+)',
        re.IGNORECASE
    )
    
    # Criminal Law Journal: 2019 Cri.L.J. 123
    CRILJ_CITATION = re.compile(
        r'(\d{4})\s*Cri\.?\s*L\.?J\.?\s*(\d+)',
        re.IGNORECASE
    )
    
    # Indian Law Reports: ILR 2019 Kar 456
    ILR_CITATION = re.compile(
        r'ILR\s+(\d{4})\s+([A-Za-z]+)\s+(\d+)',
        re.IGNORECASE
    )
    
    # All citations combined
    ALL_CITATIONS = re.compile(
        r'(?:' +
        r'(?:AIR\s+)?(\d{4})\s+AIR\s+(?:SC|[A-Z]{2,3})\s+\d+|' +
        r'\(?\s*\d{4}\s*\)?\s*\(?\s*\d+\s*\)?\s*SCC\s+\d+|' +
        r'\d{4}\s+SCR\s*\(\s*\d+\s*\)\s*\d+|' +
        r'\(?\s*\d{4}\s*\)?\s*\(?\s*\d+\s*\)?\s*SCALE\s+\d+|' +
        r'\d{4}\s*Cri\.?\s*L\.?J\.?\s*\d+|' +
        r'ILR\s+\d{4}\s+[A-Za-z]+\s+\d+' +
        r')',
        re.IGNORECASE
    )
    
    # =========================================
    # STATUTORY REFERENCE PATTERNS
    # =========================================
    
    # Section references: Section 302, Sec. 34, u/s 420, s. 376
    SECTION_PATTERN = re.compile(
        r'(?i)(?:section|sec\.?|u/s|s\.)\s+(\d+[A-Z]?)',
        re.IGNORECASE
    )
    
    # Section with Act: Section 302 of the Indian Penal Code
    SECTION_WITH_ACT = re.compile(
        r'(?i)(?:section|sec\.?|u/s|s\.)\s+(\d+[A-Z]*)\s+(?:of\s+)?(?:the\s+)?([A-Za-z\s]+(?:Act|Code))',
        re.IGNORECASE
    )
    
    # Article references (Constitution): Article 21, Art. 14
    ARTICLE_PATTERN = re.compile(
        r'(?i)(?:article|art\.?)\s+(\d+[A-Z]?)',
        re.IGNORECASE
    )
    
    # Order and Rule (CPC): Order XXI Rule 97
    ORDER_RULE_PATTERN = re.compile(
        r'(?i)order\s+([IVXLCDM]+|\d+)\s*(?:rule|r\.?)?\s*(\d+)?',
        re.IGNORECASE
    )
    
    # Common Indian Acts
    ACT_NAMES = re.compile(
        r'(?i)(?:the\s+)?(' +
        r'Indian Penal Code|IPC|' +
        r'Code of Criminal Procedure|CrPC|Cr\.P\.C\.|' +
        r'Code of Civil Procedure|CPC|C\.P\.C\.|' +
        r'Indian Evidence Act|' +
        r'Constitution of India|' +
        r'Prevention of Corruption Act|' +
        r'Negotiable Instruments Act|' +
        r'Companies Act|' +
        r'Income Tax Act|' +
        r'POCSO Act|' +
        r'NDPS Act|' +
        r'Motor Vehicles Act|' +
        r'Consumer Protection Act|' +
        r'Arbitration and Conciliation Act|' +
        r'Specific Relief Act|' +
        r'Transfer of Property Act|' +
        r'Indian Contract Act|' +
        r'Hindu Marriage Act|' +
        r'Muslim Personal Law|' +
        r'Special Marriage Act|' +
        r'Domestic Violence Act|' +
        r'Information Technology Act|' +
        r'Right to Information Act|RTI Act' +
        r')(?:\s*,?\s*\d{4})?',
        re.IGNORECASE
    )
    
    # =========================================
    # PARTY NAME PATTERNS
    # =========================================
    
    # Case title: Petitioner vs. Respondent
    CASE_TITLE = re.compile(
        r'(.+?)\s+(?:vs\.?|versus|v\.)\s+(.+?)(?:\s+(?:and|&)\s+(?:others|ors\.?|anr\.?))?',
        re.IGNORECASE
    )
    
    # State as party
    STATE_PARTY = re.compile(
        r'(?:State\s+of\s+|Union\s+of\s+)([A-Za-z\s]+)',
        re.IGNORECASE
    )
    
    # =========================================
    # CASE NUMBER PATTERNS
    # =========================================
    
    # Writ Petition: Writ Petition (Civil) No. 135/2019
    WRIT_PETITION = re.compile(
        r'(?:Writ\s+Petition|W\.?P\.?)\s*\(?\s*(?:Civil|Criminal|C|Crl)\.?\s*\)?\s*(?:No\.?)?\s*(\d+)\s*/\s*(\d{4})',
        re.IGNORECASE
    )
    
    # Criminal Appeal: Criminal Appeal No. 123/2020
    CRIMINAL_APPEAL = re.compile(
        r'(?:Criminal\s+Appeal|Crl\.?\s*A\.?)\s*(?:No\.?)?\s*(\d+)\s*/\s*(\d{4})',
        re.IGNORECASE
    )
    
    # Civil Appeal: Civil Appeal No. 456/2021
    CIVIL_APPEAL = re.compile(
        r'(?:Civil\s+Appeal|C\.?A\.?)\s*(?:No\.?)?\s*(\d+)\s*/\s*(\d{4})',
        re.IGNORECASE
    )
    
    # Special Leave Petition: SLP (C) No. 789/2022
    SLP_PATTERN = re.compile(
        r'(?:S\.?L\.?P\.?|Special\s+Leave\s+Petition)\s*\(?\s*(?:Civil|Criminal|C|Crl)\.?\s*\)?\s*(?:No\.?)?\s*(\d+)\s*/\s*(\d{4})',
        re.IGNORECASE
    )
    
    # FIR Number: FIR No. 123/2020
    FIR_PATTERN = re.compile(
        r'(?:F\.?I\.?R\.?|First\s+Information\s+Report)\s*(?:No\.?)?\s*(\d+)\s*/\s*(\d{4})',
        re.IGNORECASE
    )
    
    # =========================================
    # DATE PATTERNS
    # =========================================
    
    # Indian date format: 14th February 2024, 14-02-2024, 14.02.2024
    DATE_PATTERNS = re.compile(
        r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?'
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*'
        r'(\d{4})|'
        r'(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})',
        re.IGNORECASE
    )
    
    # =========================================
    # LEGAL TERMINOLOGY
    # =========================================
    
    # Latin legal maxims
    LATIN_MAXIMS = re.compile(
        r'\b(suo\s*moto|res\s*judicata|ratio\s*decidendi|obiter\s*dicta|'
        r'stare\s*decisis|prima\s*facie|mens\s*rea|actus\s*reus|'
        r'ex\s*parte|inter\s*alia|in\s*toto|per\s*incuriam|'
        r'de\s*novo|ultra\s*vires|intra\s*vires|locus\s*standi|'
        r'ad\s*hoc|ab\s*initio|ipso\s*facto|mutatis\s*mutandis)\b',
        re.IGNORECASE
    )
    
    # Court names
    COURT_NAMES = re.compile(
        r'(?:the\s+)?(?:Hon\'?ble\s+)?(' +
        r'Supreme\s+Court(?:\s+of\s+India)?|' +
        r'High\s+Court(?:\s+of\s+[A-Za-z\s]+)?|' +
        r'District\s+Court(?:\s+of\s+[A-Za-z\s]+)?|' +
        r'Sessions\s+Court|' +
        r'Magistrate(?:\'s)?\s+Court|' +
        r'Family\s+Court|' +
        r'Labour\s+Court|' +
        r'Consumer\s+(?:Dispute\s+)?(?:Redressal\s+)?(?:Commission|Forum)|' +
        r'Tribunal|' +
        r'NCLT|NCLAT|ITAT|DRT|DRAT|NGT|CAT|' +
        r'Apex\s+Court' +
        r')',
        re.IGNORECASE
    )
    
    # =========================================
    # PII PATTERNS (for anonymization)
    # =========================================
    
    # Phone numbers (Indian format)
    PHONE_PATTERN = re.compile(
        r'(?:\+91[-\s]?)?[6-9]\d{9}|\d{2,4}[-\s]?\d{6,8}'
    )
    
    # Email addresses
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    
    # Aadhaar number
    AADHAAR_PATTERN = re.compile(
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    )
    
    # PAN number
    PAN_PATTERN = re.compile(
        r'\b[A-Z]{5}\d{4}[A-Z]\b'
    )
    
    @classmethod
    def extract_all_citations(cls, text: str) -> List[str]:
        """Extract all legal citations from text."""
        return cls.ALL_CITATIONS.findall(text)
    
    @classmethod
    def extract_sections(cls, text: str) -> List[Tuple[str, str]]:
        """Extract section references with their associated Acts."""
        matches = cls.SECTION_WITH_ACT.findall(text)
        return [(section, act.strip()) for section, act in matches]
    
    @classmethod
    def extract_case_parties(cls, text: str) -> Dict[str, str]:
        """Extract petitioner and respondent from case title."""
        match = cls.CASE_TITLE.search(text)
        if match:
            return {
                'petitioner': match.group(1).strip(),
                'respondent': match.group(2).strip()
            }
        return {}
    
    @classmethod
    def normalize_citation(cls, citation: str) -> str:
        """Normalize citation to canonical format."""
        # Clean whitespace
        citation = re.sub(r'\s+', ' ', citation.strip())
        
        # Normalize AIR citations
        air_match = cls.AIR_CITATION.search(citation)
        if air_match:
            groups = [g for g in air_match.groups() if g]
            if len(groups) >= 3:
                return f"AIR {groups[0]} {groups[1]} {groups[2]}"
        
        # Normalize SCC citations
        scc_match = cls.SCC_CITATION.search(citation)
        if scc_match:
            return f"({scc_match.group(1)}) {scc_match.group(2)} SCC {scc_match.group(3)}"
        
        return citation
    
    @classmethod
    def anonymize_pii(cls, text: str) -> str:
        """Redact PII from text."""
        text = cls.PHONE_PATTERN.sub('[PHONE_REDACTED]', text)
        text = cls.EMAIL_PATTERN.sub('[EMAIL_REDACTED]', text)
        text = cls.AADHAAR_PATTERN.sub('[AADHAAR_REDACTED]', text)
        text = cls.PAN_PATTERN.sub('[PAN_REDACTED]', text)
        return text
    
    @classmethod
    def extract_legal_terms(cls, text: str) -> Dict[str, List[str]]:
        """Extract all legal terms from text."""
        return {
            'citations': [m[0] if m[0] else ''.join(m) for m in cls.ALL_CITATIONS.findall(text)],
            'sections': cls.SECTION_PATTERN.findall(text),
            'articles': cls.ARTICLE_PATTERN.findall(text),
            'acts': cls.ACT_NAMES.findall(text),
            'courts': cls.COURT_NAMES.findall(text),
            'latin_maxims': cls.LATIN_MAXIMS.findall(text),
        }


# Pre-compiled patterns for performance
CLEAN_HTML = re.compile(r'<[^>]+>')
CLEAN_WHITESPACE = re.compile(r'\s+')
CLEAN_SPECIAL = re.compile(r'[^\w\s\-.,;:()\'\"§¶/\\]')


def clean_legal_text(text: str) -> str:
    """
    Clean legal text by removing HTML, normalizing whitespace.
    
    Args:
        text: Raw legal text.
        
    Returns:
        Cleaned text.
    """
    # Remove HTML tags
    text = CLEAN_HTML.sub('', text)
    
    # Normalize whitespace
    text = CLEAN_WHITESPACE.sub(' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    
    return text.strip()


if __name__ == "__main__":
    # Test patterns
    sample_text = """
    In the case of Kesavananda Bharati v. State of Kerala (AIR 1973 SC 1461),
    the Supreme Court examined Article 368 of the Constitution of India.
    The petitioner argued under Section 302 of the Indian Penal Code.
    This case was also cited as (1973) 4 SCC 225.
    FIR No. 123/2020 was registered at the local police station.
    Contact: 9876543210, email: test@example.com
    """
    
    patterns = IndianLegalPatterns()
    
    print("=" * 60)
    print("Indian Legal Pattern Extraction Test")
    print("=" * 60)
    
    terms = patterns.extract_legal_terms(sample_text)
    for category, items in terms.items():
        if items:
            print(f"\n{category.upper()}:")
            for item in items:
                print(f"  - {item}")
    
    print("\n" + "=" * 60)
    print("PII Anonymization Test")
    print("=" * 60)
    print(patterns.anonymize_pii(sample_text))
