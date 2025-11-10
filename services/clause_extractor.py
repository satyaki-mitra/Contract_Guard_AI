"""
Advanced Clause Extractor using Legal-BERT + Structural Patterns
Uses nlpaueb/legal-bert-base-uncased for semantic clause understanding
"""

import torch
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from sentence_transformers import util

# Import utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import ContractAnalyzerLogger, log_info, log_error
from utils.text_processor import TextProcessor


@dataclass
class ExtractedClause:
    """
    Extracted clause with comprehensive metadata
    """
    text: str
    reference: str  # e.g., "Section 5.2", "Clause 11.1"
    category: str  # e.g., "termination", "compensation", "indemnification"
    confidence: float  # 0.0-1.0
    start_pos: int
    end_pos: int
    extraction_method: str  # "structural", "semantic", "hybrid"
    risk_indicators: List[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    subclauses: List[str] = field(default_factory=list)
    legal_bert_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "text": self.text,
            "reference": self.reference,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "extraction_method": self.extraction_method,
            "risk_indicators": self.risk_indicators,
            "subclauses": self.subclauses,
            "legal_bert_score": round(self.legal_bert_score, 3)
        }


class ClauseExtractor:
    """
    Advanced clause extraction using Legal-BERT + structural patterns
    
    Process:
    1. Structural extraction (numbered sections like "5.2", "Article III")
    2. Semantic chunking (for unstructured contracts)
    3. Legal-BERT embeddings for semantic similarity
    4. Category classification using Legal-BERT + keyword matching
    5. Deduplication and ranking
    """
    
    # =========================================================================
    # CLAUSE CATEGORY DEFINITIONS WITH REPRESENTATIVE TEXTS
    # =========================================================================
    
    CLAUSE_CATEGORIES = {
        'compensation': {
            'keywords': ['salary', 'wage', 'compensation', 'pay', 'payment', 'bonus', 
                        'commission', 'remuneration', 'fee', 'rate', 'benefits'],
            'representative_text': (
                "The Employee shall receive an annual base salary of One Hundred Thousand Dollars "
                "payable in accordance with the Company's standard payroll practices. "
                "Additional compensation may include performance bonuses and stock options."
            ),
            'weight': 1.0
        },
        'termination': {
            'keywords': ['termination', 'terminate', 'notice period', 'resignation', 
                        'dismissal', 'severance', 'end of employment', 'cessation', 'notice'],
            'representative_text': (
                "Either party may terminate this Agreement upon thirty days written notice. "
                "The Company may terminate for cause immediately upon written notice to Employee. "
                "Upon termination, Employee shall receive severance compensation."
            ),
            'weight': 1.2
        },
        'non_compete': {
            'keywords': ['non-compete', 'non-solicit', 'non-solicitation', 'restrictive covenant',
                        'competitive', 'competition', 'competing business', 'competitive activities'],
            'representative_text': (
                "Employee agrees not to engage in any competitive business activities for a period "
                "of twelve months following termination within a fifty-mile radius. "
                "Employee shall not solicit Company clients or employees during this period."
            ),
            'weight': 1.5
        },
        'confidentiality': {
            'keywords': ['confidential', 'proprietary', 'trade secret', 'disclosure',
                        'confidentiality', 'secret', 'private', 'non-disclosure'],
            'representative_text': (
                "Employee shall maintain the confidentiality of all proprietary information "
                "and trade secrets of the Company. Confidential Information includes business plans, "
                "customer lists, and technical data. These obligations survive termination."
            ),
            'weight': 1.1
        },
        'indemnification': {
            'keywords': ['indemnify', 'indemnification', 'hold harmless', 'defend',
                        'liability', 'claims', 'losses', 'damages'],
            'representative_text': (
                "Party A shall indemnify and hold harmless Party B from any claims, losses, "
                "or damages arising from Party A's breach or negligence. This indemnification "
                "includes reasonable attorneys' fees and costs of defense."
            ),
            'weight': 1.3
        },
        'intellectual_property': {
            'keywords': ['intellectual property', 'ip', 'copyright', 'patent', 'trademark',
                        'work product', 'inventions', 'creation', 'ownership', 'ip rights'],
            'representative_text': (
                "All work product and inventions created by Employee during employment shall be "
                "the exclusive property of the Company. Employee assigns all intellectual property "
                "rights including patents, copyrights, and trade secrets to the Company."
            ),
            'weight': 1.2
        },
        'liability': {
            'keywords': ['liable', 'liability', 'damages', 'limitation', 'consequential',
                        'indirect', 'punitive', 'cap', 'limited liability'],
            'representative_text': (
                "In no event shall either party be liable for indirect, incidental, or consequential "
                "damages. Total liability under this Agreement shall not exceed the amounts paid "
                "in the twelve months preceding the claim."
            ),
            'weight': 1.2
        },
        'warranty': {
            'keywords': ['warranty', 'warrant', 'representation', 'guarantee',
                        'assurance', 'promise', 'warranties'],
            'representative_text': (
                "Company warrants that the Services will be performed in a professional manner. "
                "EXCEPT AS EXPRESSLY PROVIDED, COMPANY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, "
                "INCLUDING WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE."
            ),
            'weight': 0.9
        },
        'dispute_resolution': {
            'keywords': ['arbitration', 'mediation', 'dispute', 'jurisdiction',
                        'governing law', 'venue', 'forum', 'resolution'],
            'representative_text': (
                "Any disputes arising under this Agreement shall be resolved through binding arbitration "
                "in accordance with the rules of the American Arbitration Association. "
                "This Agreement shall be governed by the laws of the State of California."
            ),
            'weight': 0.9
        },
        'insurance': {
            'keywords': ['insurance', 'coverage', 'insured', 'policy', 'premium', 'insurer'],
            'representative_text': (
                "Contractor shall maintain general liability insurance with minimum coverage of "
                "one million dollars per occurrence. Proof of insurance shall be provided to Client. "
                "Company shall be named as additional insured on all policies."
            ),
            'weight': 0.8
        },
        'assignment': {
            'keywords': ['assignment', 'assign', 'transfer', 'successor', 'binding', 'assignee'],
            'representative_text': (
                "This Agreement may not be assigned by either party without the prior written consent "
                "of the other party. This Agreement shall be binding upon and inure to the benefit "
                "of the parties' successors and permitted assigns."
            ),
            'weight': 0.8
        },
        'amendment': {
            'keywords': ['amendment', 'modify', 'modification', 'change', 'alteration', 'waiver'],
            'representative_text': (
                "This Agreement may not be amended or modified except by written instrument signed "
                "by both parties. No waiver of any provision shall be effective unless in writing. "
                "All modifications must be mutually agreed upon."
            ),
            'weight': 0.7
        },
        'force_majeure': {
            'keywords': ['force majeure', 'act of god', 'unforeseeable', 'beyond control', 'natural disaster'],
            'representative_text': (
                "Neither party shall be liable for failure to perform due to causes beyond its reasonable "
                "control including acts of God, war, strikes, or natural disasters. "
                "Performance shall be suspended during the force majeure event."
            ),
            'weight': 0.7
        },
        'entire_agreement': {
            'keywords': ['entire agreement', 'integration', 'supersedes', 'prior agreements', 'complete agreement'],
            'representative_text': (
                "This Agreement constitutes the entire agreement between the parties and supersedes "
                "all prior agreements, whether written or oral. No other representations or warranties "
                "shall be binding unless incorporated herein."
            ),
            'weight': 0.6
        },
        'general': {
            'keywords': ['provision', 'term', 'condition', 'obligation', 'requirement'],
            'representative_text': (
                "The parties agree to the following terms and conditions governing their relationship. "
                "Each party shall perform its obligations in good faith and in accordance with "
                "industry standards and applicable law."
            ),
            'weight': 0.5
        }
    }
    
    # =========================================================================
    # RISK INDICATOR PATTERNS
    # =========================================================================
    
    RISK_INDICATORS = {
        'critical': [
            'unlimited liability', 'perpetual', 'irrevocable', 'forfeit',
            'liquidated damages', 'wage withholding', 'joint and several'
        ],
        'high': [
            'non-compete', 'non-solicit', 'penalty', 'without cause',
            'sole discretion', 'immediate termination', 'at-will'
        ],
        'medium': [
            'indemnify', 'hold harmless', 'confidential', 'proprietary',
            'exclusive', 'terminate', 'default', 'breach'
        ]
    }
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(self, model_loader, contract_category: Optional[str] = None):
        """
        Initialize clause extractor with Legal-BERT
        
        Args:
            model_loader: ModelLoader instance for accessing Legal-BERT
            contract_category: Optional contract category for context-aware extraction
        """
        self.model_loader = model_loader
        self.contract_category = contract_category
        
        # Models (lazy loaded)
        self.legal_bert_model = None
        self.legal_bert_tokenizer = None
        self.embedding_model = None
        self.device = None
        
        # Category embeddings (computed from representative texts)
        self.category_embeddings = {}
        
        # Text processor
        self.text_processor = TextProcessor(use_spacy=False)
        
        # Logger
        self.logger = ContractAnalyzerLogger.get_logger()
        
        # Lazy load
        self._lazy_load()
    
    def _lazy_load(self):
        """Lazy load Legal-BERT and embedding models"""
        if self.legal_bert_model is None:
            try:
                log_info("Loading Legal-BERT for clause extraction...")
                
                # Load Legal-BERT (nlpaueb/legal-bert-base-uncased)
                self.legal_bert_model, self.legal_bert_tokenizer = self.model_loader.load_legal_bert()
                self.device = self.model_loader.device
                
                # Load sentence transformer for embeddings
                self.embedding_model = self.model_loader.load_embedding_model()
                
                # Prepare category embeddings using Legal-BERT
                self._prepare_category_embeddings()
                
                log_info("Clause extractor models loaded successfully")
                
            except Exception as e:
                log_error(e, context={"component": "ClauseExtractor", "operation": "model_loading"})
                raise
    
    def _prepare_category_embeddings(self):
        """
        Pre-compute Legal-BERT embeddings for category representative texts
        This enables semantic similarity matching for clause classification
        """
        log_info("Computing Legal-BERT embeddings for clause categories...")
        
        for category, config in self.CLAUSE_CATEGORIES.items():
            representative_text = config['representative_text']
            
            # Get Legal-BERT embedding (using [CLS] token)
            embedding = self._get_legal_bert_embedding(representative_text)
            self.category_embeddings[category] = embedding
        
        log_info(f"Prepared Legal-BERT embeddings for {len(self.category_embeddings)} categories")
    
    def _get_legal_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get Legal-BERT embedding for text using [CLS] token
        
        Args:
            text: Input text
        
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize
        inputs = self.legal_bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.legal_bert_model(**inputs)
            # Use [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return cls_embedding
    
    # =========================================================================
    # MAIN EXTRACTION METHOD
    # =========================================================================
    
    @ContractAnalyzerLogger.log_execution_time("extract_clauses")
    def extract_clauses(self, contract_text: str, 
                       max_clauses: int = 15) -> List[ExtractedClause]:
        """
        Extract and classify clauses from contract using hybrid approach
        
        Process:
        1. Structural extraction (numbered sections)
        2. Semantic chunking (for unstructured text)
        3. Legal-BERT classification
        4. Deduplicate and rank by confidence
        
        Args:
            contract_text: Full contract text
            max_clauses: Maximum number of clauses to return
        
        Returns:
            List of ExtractedClause objects sorted by confidence
        """
        
        log_info("Starting clause extraction", 
                text_length=len(contract_text),
                contract_category=self.contract_category,
                max_clauses=max_clauses)
        
        # Step 1: Extract using structural patterns
        structural_clauses = self._extract_structural_clauses(contract_text)
        log_info(f"Extracted {len(structural_clauses)} structural clauses")
        
        # Step 2: Semantic chunking for unstructured parts
        semantic_chunks = self._semantic_chunking(contract_text, structural_clauses)
        log_info(f"Created {len(semantic_chunks)} semantic chunks")
        
        # Step 3: Combine all candidates
        all_candidates = structural_clauses + semantic_chunks
        log_info(f"Total candidates: {len(all_candidates)}")
        
        # Step 4: Classify with Legal-BERT
        classified_clauses = self._classify_clauses_with_legal_bert(all_candidates)
        log_info(f"Classified {len(classified_clauses)} clauses")
        
        # Step 5: Deduplicate and rank
        final_clauses = self._deduplicate_and_rank(classified_clauses, max_clauses)
        log_info(f"Final output: {len(final_clauses)} clauses")
        
        return final_clauses
    
    # =========================================================================
    # STEP 1: STRUCTURAL EXTRACTION
    # =========================================================================
    
    def _extract_structural_clauses(self, text: str) -> List[Dict]:
        """
        Extract clauses using structural numbering patterns
        
        Detects patterns like:
        - "1.1. Text"
        - "Section 5.2. Text"
        - "Article III. Text"
        - "Clause 11. Text"
        """
        candidates = []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Patterns for legal numbering
        patterns = [
            # Match: "1.1. Text" or "1.1 Text"
            (r'(\d+\.\d+(?:\.\d+)*)\.\s*([^\n]{30,800}?)(?=\d+\.\d+(?:\.\d+)*\.|$)', 'numbered'),
            # Match: "Article 1.1. Text" or "Article III. Text"
            (r'(Article\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+))\.\s*([^\n]{30,800}?)(?=Article\s+(?:\d+|[IVXLCDM]+)|$)', 'article'),
            # Match: "Section 1.1. Text"
            (r'(Section\s+\d+(?:\.\d+)*)\.\s*([^\n]{30,800}?)(?=Section\s+\d+|$)', 'section'),
            # Match: "Clause 1.1. Text"
            (r'(Clause\s+\d+(?:\.\d+)*)\.\s*([^\n]{30,800}?)(?=Clause\s+\d+|$)', 'clause'),
            # Match: "(a) Text", "(i) Text" - sub-clauses
            (r'\(([a-z]|[ivxlcdm]+)\)\s*([^\n]{30,500}?)(?=\([a-z]|[ivxlcdm]+\)|\n\n|$)', 'subclause')
        ]
        
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                clause_text = match.group(2).strip()
                
                # Filter out boilerplate/definitions
                if not self._is_boilerplate(clause_text):
                    # Check for meaningful content
                    if self._has_meaningful_content(clause_text):
                        candidates.append({
                            'text': clause_text,
                            'reference': match.group(1).strip(),
                            'start': match.start(),
                            'end': match.end(),
                            'type': 'structural',
                            'ref_type': ref_type
                        })
        
        # Remove overlapping clauses
        candidates = self._remove_overlapping(candidates)
        
        return candidates
    
    def _is_boilerplate(self, text: str) -> bool:
        """Check if text is boilerplate/definitional rather than substantive"""
        boilerplate_indicators = [
            'shall mean', 'means and includes', 'defined as', 'definition of',
            'hereinafter referred to', 'for purposes of this', 'interpretation of',
            'as used in this', 'the term', 'shall include', 'includes but not limited'
        ]
        
        text_lower = text.lower()
        # Must have at least one strong indicator AND be definition-heavy
        has_indicator = any(indicator in text_lower for indicator in boilerplate_indicators)
        is_short_definition = len(text.split()) < 50 and '"' in text
        
        return has_indicator or is_short_definition
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if text has meaningful legal content"""
        # Must have minimum length
        if len(text.split()) < 15:
            return False
        
        # Check for legal action verbs
        action_verbs = [
            'shall', 'must', 'will', 'may', 'agrees', 'undertakes',
            'covenants', 'warrants', 'represents', 'acknowledges',
            'certifies', 'indemnifies', 'waives', 'terminates'
        ]
        
        text_lower = text.lower()
        has_action = any(verb in text_lower for verb in action_verbs)
        
        # Check for legal subjects
        legal_subjects = [
            'party', 'parties', 'employee', 'employer', 'company',
            'contractor', 'consultant', 'client', 'vendor', 'buyer',
            'seller', 'landlord', 'tenant', 'licensor', 'licensee'
        ]
        
        has_subject = any(subj in text_lower for subj in legal_subjects)
        
        return has_action or has_subject
    
    def _remove_overlapping(self, candidates: List[Dict]) -> List[Dict]:
        """Remove overlapping clause extractions"""
        if not candidates:
            return []
        
        # Sort by start position
        candidates.sort(key=lambda x: x['start'])
        
        non_overlapping = [candidates[0]]
        
        for candidate in candidates[1:]:
            last = non_overlapping[-1]
            
            # Check if overlaps
            if candidate['start'] >= last['end']:
                non_overlapping.append(candidate)
            elif len(candidate['text']) > len(last['text']):
                # Keep longer clause if overlapping
                non_overlapping[-1] = candidate
        
        return non_overlapping
    
    # =========================================================================
    # STEP 2: SEMANTIC CHUNKING
    # =========================================================================
    
    def _semantic_chunking(self, text: str, 
                          structural_clauses: List[Dict],
                          chunk_size: int = 200) -> List[Dict]:
        """
        Chunk unstructured text semantically
        Uses sentence boundaries to find natural clause boundaries
        """
        
        # Get covered ranges from structural clauses
        covered_ranges = [(c['start'], c['end']) for c in structural_clauses]
        
        # Split into sentences
        sentences = self.text_processor.extract_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        current_start = 0
        
        for sentence in sentences:
            # Check if sentence is already covered by structural extraction
            sentence_start = text.find(sentence, current_start)
            if sentence_start == -1:
                continue
                
            if self._is_in_range(sentence_start, covered_ranges):
                current_start = sentence_start + len(sentence)
                continue
            
            current_chunk.append(sentence)
            current_length += len(sentence.split())
            
            # Create chunk when reaching size limit
            if current_length >= chunk_size:
                chunk_text = ' '.join(current_chunk).strip()
                
                if len(chunk_text) >= 50 and not self._is_boilerplate(chunk_text):
                    if self._has_meaningful_content(chunk_text):
                        chunks.append({
                            'text': chunk_text,
                            'reference': f'Semantic-{len(chunks)+1}',
                            'start': sentence_start,
                            'end': sentence_start + len(chunk_text),
                            'type': 'semantic',
                            'ref_type': 'semantic'
                        })
                
                current_chunk = []
                current_length = 0
            
            current_start = sentence_start + len(sentence)
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if len(chunk_text) >= 50 and not self._is_boilerplate(chunk_text):
                if self._has_meaningful_content(chunk_text):
                    sentence_start = text.find(current_chunk[0])
                    chunks.append({
                        'text': chunk_text,
                        'reference': f'Semantic-{len(chunks)+1}',
                        'start': sentence_start,
                        'end': sentence_start + len(chunk_text),
                        'type': 'semantic',
                        'ref_type': 'semantic'
                    })
        
        return chunks
    
    def _is_in_range(self, position: int, ranges: List[Tuple[int, int]]) -> bool:
        """Check if position is within any of the ranges"""
        return any(start <= position <= end for start, end in ranges)
    
    # =========================================================================
    # STEP 3: LEGAL-BERT CLASSIFICATION
    # =========================================================================
    
    def _classify_clauses_with_legal_bert(self, candidates: List[Dict]) -> List[ExtractedClause]:
        """
        Classify clauses using Legal-BERT embeddings + keyword matching
        """
        classified = []
        
        for candidate in candidates:
            # Get Legal-BERT embedding for clause
            clause_embedding = self._get_legal_bert_embedding(candidate['text'])
            
            # Classify using hybrid approach
            category, confidence, legal_bert_score = self._classify_single_clause(
                candidate['text'], 
                clause_embedding
            )
            
            # Extract risk indicators
            risk_indicators = self._extract_risk_indicators(candidate['text'])
            
            # Extract sub-clauses if any
            subclauses = self._extract_subclauses(candidate['text'])
            
            classified.append(ExtractedClause(
                text=candidate['text'],
                reference=candidate['reference'],
                category=category,
                confidence=confidence,
                start_pos=candidate['start'],
                end_pos=candidate['end'],
                extraction_method=candidate['type'],
                risk_indicators=risk_indicators,
                embeddings=clause_embedding,
                subclauses=subclauses,
                legal_bert_score=legal_bert_score
            ))
        
        return classified
    
    def _classify_single_clause(self, text: str, 
                               clause_embedding: np.ndarray) -> Tuple[str, float, float]:
        """
        Classify single clause using Legal-BERT + keyword matching
        
        Returns:
            (category, confidence, legal_bert_score)
        """
        text_lower = text.lower()
        
        # Method 1: Keyword matching
        keyword_scores = {}
        for category, config in self.CLAUSE_CATEGORIES.items():
            keywords = config['keywords']
            weight = config['weight']
            
            keyword_count = sum(1 for kw in keywords if kw in text_lower)
            keyword_scores[category] = (keyword_count / len(keywords)) * weight
        
        # Method 2: Legal-BERT semantic similarity
        semantic_scores = {}
        clause_embedding_tensor = torch.tensor(clause_embedding).unsqueeze(0)
        
        for category, cat_embedding in self.category_embeddings.items():
            cat_embedding_tensor = torch.tensor(cat_embedding).unsqueeze(0)
            similarity = torch.nn.functional.cosine_similarity(
                clause_embedding_tensor, 
                cat_embedding_tensor
            ).item()
            semantic_scores[category] = similarity
        
        # Combine scores (70% semantic, 30% keyword)
        combined_scores = {}
        for category in self.CLAUSE_CATEGORIES.keys():
            combined = (
                semantic_scores.get(category, 0) * 0.70 +
                keyword_scores.get(category, 0) * 0.30
            )
            combined_scores[category] = combined
        
        # Get best category
        best_category = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[best_category]
        legal_bert_score = semantic_scores[best_category]
        
        return best_category, confidence, legal_bert_score
    
    def _extract_risk_indicators(self, text: str) -> List[str]:
        """Extract risk indicator keywords from clause text"""
        text_lower = text.lower()
        found_indicators = []
        
        for severity, indicators in self.RISK_INDICATORS.items():
            for indicator in indicators:
                if indicator in text_lower:
                    found_indicators.append(indicator)
        
        return found_indicators[:5]  # Top 5 risk indicators
    
    def _extract_subclauses(self, text: str) -> List[str]:
        """Extract sub-clauses from main clause (e.g., (a), (b), (i), (ii))"""
        # Pattern for sub-clauses: (a), (i), etc.
        subclause_pattern = r'\(([a-z]|[ivxlcdm]+)\)\s*([^()]{20,200}?)(?=\([a-z]|[ivxlcdm]+\)|$)'
        matches = re.findall(subclause_pattern, text, re.IGNORECASE)
        
        subclauses = []
        for ref, subtext in matches:
            clean_text = subtext.strip()
            if len(clean_text) >= 20:
                subclauses.append(f"({ref}) {clean_text}")
        
        return subclauses[:5]  # Max 5 sub-clauses
    
    # =========================================================================
    # STEP 4: DEDUPLICATION AND RANKING
    # =========================================================================
    
    def _deduplicate_and_rank(self, clauses: List[ExtractedClause],
                             max_clauses: int) -> List[ExtractedClause]:
        """
        Remove duplicates and rank by confidence + legal_bert_score
        """
        if not clauses:
            return []
        
        # Sort by combined score (confidence * 0.6 + legal_bert_score * 0.4)
        clauses.sort(
            key=lambda x: (x.confidence * 0.6 + x.legal_bert_score * 0.4), 
            reverse=True
        )
        
        # Deduplicate by text similarity
        unique_clauses = []
        seen_texts = set()
        
        for clause in clauses:
            # Simple deduplication by first 100 chars
            text_key = clause.text[:100].lower().strip()
            
            # Also check similarity to already added clauses
            is_duplicate = False
            for existing in unique_clauses:
                similarity = self._text_similarity(clause.text, existing.text)
                if similarity > 0.85:
                    is_duplicate = True
                    break
            
            if text_key not in seen_texts and not is_duplicate:
                unique_clauses.append(clause)
                seen_texts.add(text_key)
                
                if len(unique_clauses) >= max_clauses:
                    break
        
        return unique_clauses
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple Jaccard similarity)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_category_distribution(self, clauses: List[ExtractedClause]) -> Dict[str, int]:
        """Get distribution of clause categories"""
        distribution = defaultdict(int)
        for clause in clauses:
            distribution[clause.category] += 1
        
        log_info("Clause category distribution", distribution=dict(distribution))
        
        return dict(distribution)
    
    def get_high_risk_clauses(self, clauses: List[ExtractedClause]) -> List[ExtractedClause]:
        """Get clauses with risk indicators"""
        risky = [c for c in clauses if c.risk_indicators]
        risky.sort(key=lambda x: len(x.risk_indicators), reverse=True)
        
        log_info(f"Found {len(risky)} clauses with risk indicators")
        
        return risky
    
    def filter_by_category(self, clauses: List[ExtractedClause], 
                          category: str) -> List[ExtractedClause]:
        """Filter clauses by specific category"""
        filtered = [c for c in clauses if c.category == category]
        
        log_info(f"Filtered clauses by category '{category}'", count=len(filtered))
        
        return filtered
    
    def get_extraction_stats(self, clauses: List[ExtractedClause]) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        stats = {
            "total_clauses": len(clauses),
            "extraction_methods": defaultdict(int),
            "categories": defaultdict(int),
            "avg_confidence": 0.0,
            "avg_legal_bert_score": 0.0,
            "clauses_with_risk_indicators": 0,
            "clauses_with_subclauses": 0,
            "avg_clause_length": 0
        }
        
        if not clauses:
            return stats
        
        # Calculate statistics
        total_confidence = 0
        total_bert_score = 0
        total_length = 0
        
        for clause in clauses:
            stats["extraction_methods"][clause.extraction_method] += 1
            stats["categories"][clause.category] += 1
            
            total_confidence += clause.confidence
            total_bert_score += clause.legal_bert_score
            total_length += len(clause.text.split())
            
            if clause.risk_indicators:
                stats["clauses_with_risk_indicators"] += 1
            
            if clause.subclauses:
                stats["clauses_with_subclauses"] += 1
        
        stats["avg_confidence"] = round(total_confidence / len(clauses), 3)
        stats["avg_legal_bert_score"] = round(total_bert_score / len(clauses), 3)
        stats["avg_clause_length"] = round(total_length / len(clauses), 1)
        
        # Convert defaultdicts to regular dicts
        stats["extraction_methods"] = dict(stats["extraction_methods"])
        stats["categories"] = dict(stats["categories"])
        
        log_info("Extraction statistics calculated", **stats)
        
        return stats
    
    def export_clauses_to_dict(self, clauses: List[ExtractedClause]) -> List[Dict[str, Any]]:
        """Export clauses to list of dictionaries for serialization"""
        return [clause.to_dict() for clause in clauses]
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def extract_clauses_batch(self, contracts: List[str], 
                             max_clauses: int = 15) -> List[List[ExtractedClause]]:
        """
        Extract clauses from multiple contracts (batch processing)
        
        Args:
            contracts: List of contract texts
            max_clauses: Maximum clauses per contract
        
        Returns:
            List of clause lists (one per contract)
        """
        log_info("Starting batch clause extraction", batch_size=len(contracts))
        
        all_results = []
        
        for i, contract_text in enumerate(contracts):
            log_info(f"Processing contract {i+1}/{len(contracts)}")
            
            try:
                clauses = self.extract_clauses(contract_text, max_clauses)
                all_results.append(clauses)
            except Exception as e:
                log_error(e, context={
                    "component": "ClauseExtractor",
                    "operation": "batch_extraction",
                    "contract_index": i
                })
                all_results.append([])  # Empty list on error
        
        successful = sum(1 for r in all_results if r)
        log_info("Batch extraction completed",
                total=len(contracts),
                successful=successful,
                failed=len(contracts) - successful)
        
        return all_results
    
    # =========================================================================
    # CONTEXT-AWARE EXTRACTION (for specific contract types)
    # =========================================================================
    
    def extract_category_specific_clauses(self, contract_text: str,
                                         target_categories: List[str],
                                         max_clauses: int = 10) -> List[ExtractedClause]:
        """
        Extract clauses focusing on specific categories
        (e.g., only extract termination + compensation clauses)
        
        Args:
            contract_text: Contract text
            target_categories: List of target categories to focus on
            max_clauses: Maximum clauses to return
        
        Returns:
            List of clauses filtered to target categories
        """
        log_info("Extracting category-specific clauses",
                target_categories=target_categories)
        
        # Extract all clauses
        all_clauses = self.extract_clauses(contract_text, max_clauses=50)
        
        # Filter to target categories
        filtered = [c for c in all_clauses if c.category in target_categories]
        
        # Sort by confidence and limit
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        filtered = filtered[:max_clauses]
        
        log_info(f"Extracted {len(filtered)} category-specific clauses")
        
        return filtered
    
    def extract_risky_clauses_only(self, contract_text: str,
                                   max_clauses: int = 10) -> List[ExtractedClause]:
        """
        Extract only clauses with risk indicators
        
        Args:
            contract_text: Contract text
            max_clauses: Maximum clauses to return
        
        Returns:
            List of high-risk clauses
        """
        log_info("Extracting risky clauses only")
        
        # Extract all clauses
        all_clauses = self.extract_clauses(contract_text, max_clauses=50)
        
        # Filter to clauses with risk indicators
        risky = [c for c in all_clauses if c.risk_indicators]
        
        # Sort by number of risk indicators + confidence
        risky.sort(
            key=lambda x: (len(x.risk_indicators) * 0.6 + x.confidence * 0.4),
            reverse=True
        )
        
        risky = risky[:max_clauses]
        
        log_info(f"Extracted {len(risky)} risky clauses")
        
        return risky
    
    # =========================================================================
    # CLAUSE COMPARISON
    # =========================================================================
    
    def compare_clauses(self, clause1: ExtractedClause, 
                       clause2: ExtractedClause) -> Dict[str, Any]:
        """
        Compare two clauses for similarity and differences
        
        Returns:
            Dictionary with comparison results
        """
        # Text similarity
        text_sim = self._text_similarity(clause1.text, clause2.text)
        
        # Embedding similarity (if available)
        embedding_sim = 0.0
        if clause1.embeddings is not None and clause2.embeddings is not None:
            emb1 = torch.tensor(clause1.embeddings).unsqueeze(0)
            emb2 = torch.tensor(clause2.embeddings).unsqueeze(0)
            embedding_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        
        # Category match
        category_match = clause1.category == clause2.category
        
        # Risk indicator overlap
        risk_overlap = set(clause1.risk_indicators) & set(clause2.risk_indicators)
        
        comparison = {
            "text_similarity": round(text_sim, 3),
            "embedding_similarity": round(embedding_sim, 3),
            "category_match": category_match,
            "risk_indicator_overlap": list(risk_overlap),
            "confidence_diff": round(abs(clause1.confidence - clause2.confidence), 3),
            "same_extraction_method": clause1.extraction_method == clause2.extraction_method
        }
        
        log_info("Clause comparison completed", **comparison)
        
        return comparison
    
    def find_similar_clauses(self, target_clause: ExtractedClause,
                            clause_pool: List[ExtractedClause],
                            similarity_threshold: float = 0.70) -> List[Tuple[ExtractedClause, float]]:
        """
        Find clauses similar to target clause
        
        Args:
            target_clause: Target clause to match
            clause_pool: Pool of clauses to search
            similarity_threshold: Minimum similarity (0-1)
        
        Returns:
            List of (clause, similarity_score) tuples
        """
        similar = []
        
        for candidate in clause_pool:
            if candidate.reference == target_clause.reference:
                continue  # Skip same clause
            
            # Calculate similarity
            if target_clause.embeddings is not None and candidate.embeddings is not None:
                emb1 = torch.tensor(target_clause.embeddings).unsqueeze(0)
                emb2 = torch.tensor(candidate.embeddings).unsqueeze(0)
                similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            else:
                similarity = self._text_similarity(target_clause.text, candidate.text)
            
            if similarity >= similarity_threshold:
                similar.append((candidate, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        
        log_info(f"Found {len(similar)} similar clauses",
                threshold=similarity_threshold)
        
        return similar
    
    # =========================================================================
    # CLAUSE MERGING (for related clauses)
    # =========================================================================
    
    def merge_related_clauses(self, clauses: List[ExtractedClause],
                             similarity_threshold: float = 0.80) -> List[ExtractedClause]:
        """
        Merge related clauses that are highly similar
        (useful for consolidating subclauses or redundant extractions)
        
        Args:
            clauses: List of clauses
            similarity_threshold: Minimum similarity for merging
        
        Returns:
            List of merged clauses
        """
        if not clauses:
            return []
        
        merged = []
        used_indices = set()
        
        for i, clause1 in enumerate(clauses):
            if i in used_indices:
                continue
            
            # Find similar clauses
            similar_group = [clause1]
            
            for j, clause2 in enumerate(clauses[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                similarity = self._text_similarity(clause1.text, clause2.text)
                
                if similarity >= similarity_threshold and clause1.category == clause2.category:
                    similar_group.append(clause2)
                    used_indices.add(j)
            
            # Merge group
            if len(similar_group) > 1:
                merged_clause = self._merge_clause_group(similar_group)
                merged.append(merged_clause)
            else:
                merged.append(clause1)
            
            used_indices.add(i)
        
        log_info(f"Merged {len(clauses)} clauses into {len(merged)} clauses")
        
        return merged
    
    def _merge_clause_group(self, clause_group: List[ExtractedClause]) -> ExtractedClause:
        """Merge a group of similar clauses into one"""
        # Use the clause with highest confidence as base
        base_clause = max(clause_group, key=lambda x: x.confidence)
        
        # Combine risk indicators
        all_risk_indicators = set()
        for clause in clause_group:
            all_risk_indicators.update(clause.risk_indicators)
        
        # Combine subclauses
        all_subclauses = []
        for clause in clause_group:
            all_subclauses.extend(clause.subclauses)
        
        # Average confidence and legal_bert_score
        avg_confidence = sum(c.confidence for c in clause_group) / len(clause_group)
        avg_bert_score = sum(c.legal_bert_score for c in clause_group) / len(clause_group)
        
        # Create merged clause
        merged = ExtractedClause(
            text=base_clause.text,
            reference=f"{base_clause.reference} (merged from {len(clause_group)} clauses)",
            category=base_clause.category,
            confidence=avg_confidence,
            start_pos=base_clause.start_pos,
            end_pos=base_clause.end_pos,
            extraction_method="merged",
            risk_indicators=list(all_risk_indicators),
            embeddings=base_clause.embeddings,
            subclauses=all_subclauses,
            legal_bert_score=avg_bert_score
        )
        
        return merged
    
    # =========================================================================
    # ADVANCED: CLAUSE RELATIONSHIP DETECTION
    # =========================================================================
    
    def detect_clause_relationships(self, clauses: List[ExtractedClause]) -> List[Dict[str, Any]]:
        """
        Detect relationships between clauses (e.g., one clause references another)
        
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        for i, clause1 in enumerate(clauses):
            for j, clause2 in enumerate(clauses[i+1:], start=i+1):
                # Check for explicit references
                if self._has_reference(clause1.text, clause2.reference):
                    relationships.append({
                        "from": clause1.reference,
                        "to": clause2.reference,
                        "type": "explicit_reference",
                        "description": f"{clause1.reference} references {clause2.reference}"
                    })
                
                # Check for thematic relationship (same category + high similarity)
                if clause1.category == clause2.category:
                    similarity = self._text_similarity(clause1.text, clause2.text)
                    if 0.40 <= similarity < 0.70:  # Related but not duplicate
                        relationships.append({
                            "from": clause1.reference,
                            "to": clause2.reference,
                            "type": "thematic_relationship",
                            "similarity": round(similarity, 3),
                            "description": f"Related {clause1.category} clauses"
                        })
        
        log_info(f"Detected {len(relationships)} clause relationships")
        
        return relationships
    
    def _has_reference(self, text: str, reference: str) -> bool:
        """Check if text contains reference to another clause"""
        # Common reference patterns
        patterns = [
            rf'\b{re.escape(reference)}\b',
            rf'pursuant to {re.escape(reference)}',
            rf'as defined in {re.escape(reference)}',
            rf'see {re.escape(reference)}'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)