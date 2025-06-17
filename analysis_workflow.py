"""
Optimized Analysis Workflow for Cyber Assessment Reviewer
Implements systematic evidence evaluation and control analysis workflow
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from models import AssessmentResult, ControlData, EvidenceFile
from ai_backend import CyberAssessmentReviewer
from file_processors import CybersecurityTextProcessor

logger = logging.getLogger(__name__)

@dataclass
class EvidenceMapping:
    """Maps evidence to control requirements"""
    control_id: str
    relevant_evidence: Dict[str, List[str]]  # filename -> relevant_sections
    evidence_quality_score: float
    coverage_score: float  # How well evidence covers control requirements

@dataclass
class AnalysisContext:
    """Context for systematic analysis"""
    framework: str
    control_count: int
    evidence_files: List[str]
    analysis_depth: str  # 'basic', 'standard', 'comprehensive'

class OptimizedAnalysisWorkflow:
    """Optimized workflow for systematic cybersecurity control analysis"""
    
    def __init__(self, reviewer: CyberAssessmentReviewer):
        self.reviewer = reviewer
        self.text_processor = CybersecurityTextProcessor()
    
    def analyze_controls_systematically(self, controls: List[ControlData], 
                                      evidence_texts: Dict[str, Dict[str, str]], 
                                      framework: str = "NIST",
                                      analysis_depth: str = "standard") -> List[AssessmentResult]:
        """
        Systematic analysis of controls with optimized evidence correlation
        
        Args:
            controls: List of controls to analyze
            evidence_texts: Evidence organized by filename and section
            framework: Cybersecurity framework to use
            analysis_depth: Level of analysis detail
        
        Returns:
            List of validated assessment results
        """
        logger.info(f"Starting systematic analysis of {len(controls)} controls")
        
        # Create analysis context
        context = AnalysisContext(
            framework=framework,
            control_count=len(controls),
            evidence_files=list(evidence_texts.keys()),
            analysis_depth=analysis_depth
        )
        
        # Step 1: Pre-process and categorize evidence
        categorized_evidence = self._categorize_evidence_by_domain(evidence_texts)
        
        # Step 2: Map evidence to controls
        evidence_mappings = self._map_evidence_to_controls(controls, categorized_evidence)
        
        # Step 3: Prioritize controls for analysis
        prioritized_controls = self._prioritize_controls(controls, evidence_mappings, framework)
        
        # Step 4: Analyze controls with optimized evidence correlation
        results = []
        for i, control in enumerate(prioritized_controls):
            logger.info(f"Analyzing control {i+1}/{len(prioritized_controls)}: {control.control_id}")
            
            # Get relevant evidence for this control
            relevant_evidence = self._get_relevant_evidence_for_control(
                control, evidence_mappings, categorized_evidence
            )
            
            # Perform analysis with enhanced context
            result = self._analyze_control_with_context(
                control, relevant_evidence, context, evidence_mappings
            )
            
            results.append(result)
        
        logger.info(f"Completed systematic analysis of {len(results)} controls")
        return results
    
    def _categorize_evidence_by_domain(self, evidence_texts: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, List[str]]]:
        """Categorize evidence by cybersecurity domains"""
        categorized = {}
        
        for filename, sections in evidence_texts.items():
            categorized[filename] = {}
            
            for section_name, text in sections.items():
                # Use the text processor to categorize by security domain
                domain_categories = self.text_processor.categorize_text_by_security_domain(text)
                
                for domain, sentences in domain_categories.items():
                    if domain not in categorized[filename]:
                        categorized[filename][domain] = []
                    categorized[filename][domain].extend(sentences)
        
        return categorized
    
    def _map_evidence_to_controls(self, controls: List[ControlData], 
                                 categorized_evidence: Dict[str, Dict[str, List[str]]]) -> Dict[str, EvidenceMapping]:
        """Map evidence to specific controls based on relevance"""
        mappings = {}
        
        for control in controls:
            relevant_evidence = {}
            total_relevance_score = 0
            coverage_indicators = 0
            
            # Extract key terms from control requirement
            control_keywords = self._extract_control_keywords(control)
            
            for filename, domains in categorized_evidence.items():
                file_relevance = {}
                
                for domain, sentences in domains.items():
                    # Calculate relevance score for this domain
                    relevance_score = self._calculate_relevance_score(
                        control_keywords, sentences, domain
                    )
                    
                    if relevance_score > 0.3:  # Threshold for relevance
                        file_relevance[domain] = sentences
                        total_relevance_score += relevance_score
                        coverage_indicators += 1
                
                if file_relevance:
                    relevant_evidence[filename] = file_relevance
            
            # Calculate overall evidence quality and coverage
            evidence_quality = min(1.0, total_relevance_score / len(categorized_evidence))
            coverage_score = min(1.0, coverage_indicators / 5.0)  # Normalize to 5 domains
            
            mappings[control.control_id] = EvidenceMapping(
                control_id=control.control_id,
                relevant_evidence=relevant_evidence,
                evidence_quality_score=evidence_quality,
                coverage_score=coverage_score
            )
        
        return mappings
    
    def _prioritize_controls(self, controls: List[ControlData], 
                           evidence_mappings: Dict[str, EvidenceMapping], 
                           framework: str) -> List[ControlData]:
        """Prioritize controls for analysis based on evidence quality and criticality"""
        
        # Define framework-specific critical control patterns
        critical_patterns = {
            'NIST': ['access', 'authentication', 'encryption', 'monitoring', 'incident'],
            'ISO27001': ['access', 'information', 'security', 'risk', 'management'],
            'SOC2': ['security', 'availability', 'confidentiality', 'processing', 'privacy'],
            'CIS': ['inventory', 'configuration', 'access', 'vulnerability', 'monitoring'],
            'PCI-DSS': ['cardholder', 'encryption', 'access', 'network', 'monitoring']
        }
        
        framework_patterns = critical_patterns.get(framework, critical_patterns['NIST'])
        
        def priority_score(control: ControlData) -> float:
            mapping = evidence_mappings.get(control.control_id)
            if not mapping:
                return 0.0
            
            # Base score from evidence quality
            score = mapping.evidence_quality_score * 0.4 + mapping.coverage_score * 0.3
            
            # Boost score for critical controls
            control_text = f"{control.control_name} {control.requirement}".lower()
            criticality_boost = sum(0.1 for pattern in framework_patterns if pattern in control_text)
            score += min(0.3, criticality_boost)
            
            return score
        
        # Sort controls by priority score (highest first)
        return sorted(controls, key=priority_score, reverse=True)
    
    def _get_relevant_evidence_for_control(self, control: ControlData, 
                                         evidence_mappings: Dict[str, EvidenceMapping],
                                         categorized_evidence: Dict[str, Dict[str, List[str]]]) -> str:
        """Get the most relevant evidence text for a specific control"""
        mapping = evidence_mappings.get(control.control_id)
        if not mapping or not mapping.relevant_evidence:
            # Fallback to all evidence if no specific mapping
            all_evidence = []
            for filename, sections in categorized_evidence.items():
                for domain, sentences in sections.items():
                    all_evidence.extend(sentences[:2])  # Limit sentences per domain
            return '. '.join(all_evidence[:10])  # Limit total sentences
        
        # Compile relevant evidence
        relevant_text = []
        for filename, domains in mapping.relevant_evidence.items():
            for domain, sentences in domains.items():
                # Add domain context
                relevant_text.append(f"[{filename} - {domain.replace('_', ' ').title()}]: {'. '.join(sentences[:3])}")
        
        return '\n\n'.join(relevant_text[:8])  # Limit to 8 evidence sections
    
    def _analyze_control_with_context(self, control: ControlData, 
                                    relevant_evidence: str,
                                    context: AnalysisContext,
                                    evidence_mappings: Dict[str, EvidenceMapping]) -> AssessmentResult:
        """Analyze control with enhanced context and evidence correlation"""
        
        # Get evidence mapping for this control
        mapping = evidence_mappings.get(control.control_id)
        
        # Enhance the evidence text with context
        enhanced_evidence = self._enhance_evidence_with_context(
            relevant_evidence, control, mapping, context
        )
        
        # Perform the analysis using the reviewer
        result = self.reviewer.analyze_control_point(
            control, 
            {f"Enhanced_Evidence_{control.control_id}": {"Relevant_Evidence": enhanced_evidence}}, 
            context.framework
        )
        
        # Add workflow-specific enhancements
        if mapping:
            result.technical_details.append(f"Evidence Quality Score: {mapping.evidence_quality_score:.2f}")
            result.technical_details.append(f"Coverage Score: {mapping.coverage_score:.2f}")
            
            if mapping.evidence_quality_score < 0.5:
                result.evidence_gaps.append("Low evidence quality detected")
            
            if mapping.coverage_score < 0.4:
                result.evidence_gaps.append("Insufficient evidence coverage of control requirements")
        
        return result
    
    def _enhance_evidence_with_context(self, evidence_text: str, control: ControlData,
                                     mapping: Optional[EvidenceMapping], 
                                     context: AnalysisContext) -> str:
        """Enhance evidence text with additional context for better AI analysis"""
        
        enhanced = f"""
CONTROL ANALYSIS CONTEXT:
- Framework: {context.framework}
- Analysis Depth: {context.analysis_depth}
- Total Evidence Files: {len(context.evidence_files)}
- Evidence Files: {', '.join(context.evidence_files)}

EVIDENCE QUALITY INDICATORS:
"""
        
        if mapping:
            enhanced += f"""- Evidence Quality Score: {mapping.evidence_quality_score:.2f}/1.0
- Coverage Score: {mapping.coverage_score:.2f}/1.0
- Relevant Evidence Sources: {len(mapping.relevant_evidence)} files
"""
        
        enhanced += f"""
CONTROL-SPECIFIC EVIDENCE:
{evidence_text}

ANALYSIS INSTRUCTIONS:
- Focus on evidence that directly addresses the control requirement
- Consider the evidence quality and coverage scores in your assessment
- If evidence quality is low (<0.5), be more conservative in compliance determination
- If coverage is incomplete (<0.4), identify specific evidence gaps
- Provide specific technical details found in the evidence
"""
        
        return enhanced
    
    def _extract_control_keywords(self, control: ControlData) -> List[str]:
        """Extract key terms from control for evidence matching"""
        import re
        
        # Combine control name and requirement
        text = f"{control.control_name} {control.requirement}".lower()
        
        # Extract meaningful keywords (3+ characters, not common words)
        common_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'have', 'from', 'they', 'been', 'have', 'their'}
        words = re.findall(r'\b\w{3,}\b', text)
        keywords = [word for word in words if word not in common_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_relevance_score(self, control_keywords: List[str], 
                                 evidence_sentences: List[str], domain: str) -> float:
        """Calculate relevance score between control and evidence"""
        if not evidence_sentences or not control_keywords:
            return 0.0
        
        # Combine all evidence sentences
        evidence_text = ' '.join(evidence_sentences).lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in control_keywords if keyword in evidence_text)
        keyword_score = matches / len(control_keywords)
        
        # Domain relevance boost
        domain_relevance = {
            'access_control': 0.2,
            'encryption': 0.15,
            'monitoring': 0.15,
            'compliance': 0.1,
            'vulnerability': 0.1,
            'incident_response': 0.1,
            'network_security': 0.1,
            'data_protection': 0.1
        }
        
        domain_boost = domain_relevance.get(domain, 0.05)
        
        return min(1.0, keyword_score + domain_boost)
