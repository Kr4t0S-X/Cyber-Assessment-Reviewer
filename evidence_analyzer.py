"""
Advanced Evidence Review System for Cyber Assessment Reviewer
Implements sophisticated evidence quality scoring and analysis
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

from models import ControlData, AssessmentResult

logger = logging.getLogger(__name__)

class EvidenceType(Enum):
    """Types of evidence commonly found in cybersecurity assessments"""
    POLICY_DOCUMENT = "policy_document"
    CONFIGURATION_SCREENSHOT = "configuration_screenshot"
    AUDIT_LOG = "audit_log"
    PROCEDURE_DOCUMENTATION = "procedure_documentation"
    TECHNICAL_SPECIFICATION = "technical_specification"
    COMPLIANCE_REPORT = "compliance_report"
    TRAINING_RECORD = "training_record"
    INCIDENT_REPORT = "incident_report"
    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    RISK_ASSESSMENT = "risk_assessment"
    UNKNOWN = "unknown"

class EvidenceQuality(Enum):
    """Evidence quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    INSUFFICIENT = "insufficient"

@dataclass
class EvidenceMetrics:
    """Comprehensive evidence quality metrics"""
    control_id: str
    overall_quality_score: float
    completeness_score: float
    relevance_score: float
    freshness_score: float
    technical_depth_score: float
    credibility_score: float
    evidence_types: List[str]
    evidence_gaps: List[str]
    recommendations: List[str]
    contradiction_flags: List[str]
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'control_id': self.control_id,
            'overall_quality_score': self.overall_quality_score,
            'completeness_score': self.completeness_score,
            'relevance_score': self.relevance_score,
            'freshness_score': self.freshness_score,
            'technical_depth_score': self.technical_depth_score,
            'credibility_score': self.credibility_score,
            'evidence_types': self.evidence_types,
            'evidence_gaps': self.evidence_gaps,
            'recommendations': self.recommendations,
            'contradiction_flags': self.contradiction_flags
        }

@dataclass
class EvidenceFragment:
    """Individual piece of evidence with metadata"""
    content: str
    source: str
    evidence_type: EvidenceType
    confidence: float
    relevance_score: float
    technical_indicators: List[str]
    timestamps: List[str]
    
class AdvancedEvidenceAnalyzer:
    """Advanced evidence analysis system with quality scoring"""
    
    def __init__(self, config=None):
        self.config = config
        self.evidence_patterns = self._initialize_evidence_patterns()
        self.quality_indicators = self._initialize_quality_indicators()
        self.gap_detectors = self._initialize_gap_detectors()
        self.contradiction_patterns = self._initialize_contradiction_patterns()
        
    def _initialize_evidence_patterns(self) -> Dict[EvidenceType, List[str]]:
        """Initialize patterns for identifying evidence types"""
        return {
            EvidenceType.POLICY_DOCUMENT: [
                r'policy', r'standard', r'guideline', r'procedure', r'governance',
                r'framework', r'charter', r'directive', r'regulation', r'compliance'
            ],
            EvidenceType.CONFIGURATION_SCREENSHOT: [
                r'screenshot', r'configuration', r'settings', r'interface', r'dashboard',
                r'console', r'admin panel', r'management interface', r'gui'
            ],
            EvidenceType.AUDIT_LOG: [
                r'audit log', r'system log', r'security log', r'access log', r'event log',
                r'log file', r'syslog', r'audit trail', r'logging'
            ],
            EvidenceType.PROCEDURE_DOCUMENTATION: [
                r'procedure', r'process', r'workflow', r'step.*by.*step', r'instruction',
                r'manual', r'guide', r'runbook', r'checklist'
            ],
            EvidenceType.TECHNICAL_SPECIFICATION: [
                r'technical specification', r'system architecture', r'network diagram',
                r'infrastructure', r'technical design', r'implementation details'
            ],
            EvidenceType.COMPLIANCE_REPORT: [
                r'compliance report', r'assessment report', r'audit report', r'certification',
                r'attestation', r'compliance status', r'assessment results'
            ],
            EvidenceType.TRAINING_RECORD: [
                r'training', r'education', r'awareness', r'certification', r'course',
                r'learning', r'workshop', r'seminar', r'training record'
            ],
            EvidenceType.INCIDENT_REPORT: [
                r'incident', r'security event', r'breach', r'compromise', r'incident response',
                r'security incident', r'event report', r'post.*incident'
            ],
            EvidenceType.VULNERABILITY_SCAN: [
                r'vulnerability scan', r'vuln scan', r'security scan', r'penetration test',
                r'pen test', r'vulnerability assessment', r'security assessment'
            ],
            EvidenceType.RISK_ASSESSMENT: [
                r'risk assessment', r'risk analysis', r'threat assessment', r'risk evaluation',
                r'risk matrix', r'risk register', r'threat model'
            ]
        }
    
    def _initialize_quality_indicators(self) -> Dict[str, List[str]]:
        """Initialize quality indicators for evidence assessment"""
        return {
            'high_quality': [
                r'documented', r'verified', r'approved', r'reviewed', r'validated',
                r'tested', r'monitored', r'audited', r'certified', r'compliant',
                r'implemented', r'configured', r'established', r'maintained'
            ],
            'technical_depth': [
                r'configuration', r'implementation', r'architecture', r'protocol',
                r'algorithm', r'encryption', r'authentication', r'authorization',
                r'firewall', r'intrusion', r'detection', r'prevention', r'monitoring'
            ],
            'credibility': [
                r'third.*party', r'external.*audit', r'independent.*assessment',
                r'certified.*auditor', r'compliance.*officer', r'security.*team',
                r'management.*approved', r'board.*approved', r'executive.*review'
            ],
            'freshness': [
                r'recent', r'current', r'updated', r'latest', r'new', r'revised',
                r'modified', r'amended', r'refreshed', r'ongoing'
            ],
            'completeness': [
                r'comprehensive', r'complete', r'full', r'detailed', r'thorough',
                r'extensive', r'end.*to.*end', r'holistic', r'all.*inclusive'
            ]
        }
    
    def _initialize_gap_detectors(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting evidence gaps"""
        return {
            'missing_implementation': [
                r'not.*implemented', r'not.*configured', r'not.*established',
                r'not.*deployed', r'not.*activated', r'not.*enabled',
                r'pending', r'planned', r'future', r'to.*be.*implemented'
            ],
            'missing_documentation': [
                r'not.*documented', r'no.*documentation', r'undocumented',
                r'missing.*documentation', r'not.*written', r'verbal.*only'
            ],
            'missing_testing': [
                r'not.*tested', r'no.*testing', r'untested', r'not.*validated',
                r'not.*verified', r'no.*validation', r'not.*confirmed'
            ],
            'missing_monitoring': [
                r'not.*monitored', r'no.*monitoring', r'unmonitored',
                r'not.*tracked', r'no.*tracking', r'not.*logged'
            ],
            'outdated_evidence': [
                r'outdated', r'old', r'legacy', r'deprecated', r'obsolete',
                r'previous.*version', r'superseded', r'replaced'
            ]
        }
    
    def _initialize_contradiction_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting contradictions"""
        return {
            'implementation_contradiction': [
                (r'implemented', r'not.*implemented'),
                (r'configured', r'not.*configured'),
                (r'enabled', r'disabled'),
                (r'active', r'inactive')
            ],
            'policy_contradiction': [
                (r'required', r'optional'),
                (r'mandatory', r'recommended'),
                (r'enforced', r'not.*enforced')
            ],
            'status_contradiction': [
                (r'compliant', r'non.*compliant'),
                (r'operational', r'non.*operational'),
                (r'functional', r'non.*functional')
            ]
        }
    
    def analyze_evidence_quality(self, control: ControlData, 
                                evidence_texts: Dict[str, Dict[str, str]]) -> EvidenceMetrics:
        """Perform comprehensive evidence quality analysis"""
        
        # Extract evidence fragments
        evidence_fragments = self._extract_evidence_fragments(evidence_texts)
        
        # Calculate quality dimensions
        completeness_score = self._calculate_completeness_score(control, evidence_fragments)
        relevance_score = self._calculate_relevance_score(control, evidence_fragments)
        freshness_score = self._calculate_freshness_score(evidence_fragments)
        technical_depth_score = self._calculate_technical_depth_score(evidence_fragments)
        credibility_score = self._calculate_credibility_score(evidence_fragments)
        
        # Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality_score(
            completeness_score, relevance_score, freshness_score, 
            technical_depth_score, credibility_score
        )
        
        # Identify evidence types
        evidence_types = self._identify_evidence_types(evidence_fragments)
        
        # Detect evidence gaps
        evidence_gaps = self._detect_evidence_gaps(control, evidence_fragments)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            control, completeness_score, relevance_score, freshness_score,
            technical_depth_score, credibility_score, evidence_gaps
        )
        
        # Detect contradictions
        contradiction_flags = self._detect_contradictions(evidence_fragments)
        
        return EvidenceMetrics(
            control_id=control.control_id,
            overall_quality_score=overall_quality_score,
            completeness_score=completeness_score,
            relevance_score=relevance_score,
            freshness_score=freshness_score,
            technical_depth_score=technical_depth_score,
            credibility_score=credibility_score,
            evidence_types=evidence_types,
            evidence_gaps=evidence_gaps,
            recommendations=recommendations,
            contradiction_flags=contradiction_flags
        )
    
    def _extract_evidence_fragments(self, evidence_texts: Dict[str, Dict[str, str]]) -> List[EvidenceFragment]:
        """Extract and categorize evidence fragments"""
        fragments = []
        
        for filename, sections in evidence_texts.items():
            for section_name, content in sections.items():
                if not content or "Error" in section_name:
                    continue
                
                # Identify evidence type
                evidence_type = self._classify_evidence_type(content)
                
                # Extract technical indicators
                technical_indicators = self._extract_technical_indicators(content)
                
                # Extract timestamps
                timestamps = self._extract_timestamps(content)
                
                # Calculate fragment relevance
                relevance_score = self._calculate_fragment_relevance(content)
                
                # Estimate confidence
                confidence = self._estimate_fragment_confidence(content, evidence_type)
                
                fragment = EvidenceFragment(
                    content=content,
                    source=f"{filename}:{section_name}",
                    evidence_type=evidence_type,
                    confidence=confidence,
                    relevance_score=relevance_score,
                    technical_indicators=technical_indicators,
                    timestamps=timestamps
                )
                
                fragments.append(fragment)
        
        return fragments
    
    def _classify_evidence_type(self, content: str) -> EvidenceType:
        """Classify evidence type based on content"""
        content_lower = content.lower()
        
        # Score each evidence type
        type_scores = {}
        
        for evidence_type, patterns in self.evidence_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            
            if score > 0:
                type_scores[evidence_type] = score
        
        # Return the type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        return EvidenceType.UNKNOWN
    
    def _extract_technical_indicators(self, content: str) -> List[str]:
        """Extract technical indicators from content"""
        indicators = []
        content_lower = content.lower()
        
        technical_patterns = [
            r'configuration', r'implementation', r'deployment', r'architecture',
            r'protocol', r'algorithm', r'encryption', r'authentication',
            r'authorization', r'firewall', r'intrusion', r'detection',
            r'prevention', r'monitoring', r'logging', r'audit',
            r'compliance', r'vulnerability', r'patch', r'update'
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, content_lower)
            indicators.extend(matches)
        
        return list(set(indicators))  # Remove duplicates
    
    def _extract_timestamps(self, content: str) -> List[str]:
        """Extract timestamps and dates from content"""
        timestamps = []
        
        # Date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\b\d{1,2}\s+\w+\s+\d{4}\b',  # DD Month YYYY
            r'\b\w+\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            timestamps.extend(matches)
        
        return timestamps
    
    def _calculate_fragment_relevance(self, content: str) -> float:
        """Calculate relevance score for an evidence fragment"""
        content_lower = content.lower()
        
        # High relevance indicators
        high_relevance_terms = [
            'security', 'control', 'compliance', 'audit', 'risk', 'policy',
            'procedure', 'implementation', 'configuration', 'monitoring'
        ]
        
        relevance_score = 0.0
        for term in high_relevance_terms:
            if term in content_lower:
                relevance_score += 0.1
        
        # Length bonus (longer content generally more relevant)
        length_score = min(0.3, len(content) / 1000)  # Max 0.3 for 1000+ chars
        
        return min(1.0, relevance_score + length_score)
    
    def _estimate_fragment_confidence(self, content: str, evidence_type: EvidenceType) -> float:
        """Estimate confidence in evidence fragment"""
        confidence = 0.5  # Base confidence
        
        # Type-based confidence
        type_confidence = {
            EvidenceType.POLICY_DOCUMENT: 0.8,
            EvidenceType.CONFIGURATION_SCREENSHOT: 0.9,
            EvidenceType.AUDIT_LOG: 0.85,
            EvidenceType.COMPLIANCE_REPORT: 0.9,
            EvidenceType.TECHNICAL_SPECIFICATION: 0.8,
            EvidenceType.UNKNOWN: 0.3
        }
        
        confidence = type_confidence.get(evidence_type, 0.5)
        
        # Quality indicators boost confidence
        content_lower = content.lower()
        for indicator in self.quality_indicators['high_quality']:
            if re.search(indicator, content_lower):
                confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_completeness_score(self, control: ControlData, 
                                    evidence_fragments: List[EvidenceFragment]) -> float:
        """Calculate evidence completeness score"""
        if not evidence_fragments:
            return 0.0
        
        # Define required evidence types for different control categories
        control_categories = {
            'access_control': [
                EvidenceType.POLICY_DOCUMENT,
                EvidenceType.CONFIGURATION_SCREENSHOT,
                EvidenceType.AUDIT_LOG
            ],
            'data_protection': [
                EvidenceType.POLICY_DOCUMENT,
                EvidenceType.TECHNICAL_SPECIFICATION,
                EvidenceType.CONFIGURATION_SCREENSHOT
            ],
            'monitoring': [
                EvidenceType.CONFIGURATION_SCREENSHOT,
                EvidenceType.AUDIT_LOG,
                EvidenceType.PROCEDURE_DOCUMENTATION
            ],
            'incident_response': [
                EvidenceType.PROCEDURE_DOCUMENTATION,
                EvidenceType.TRAINING_RECORD,
                EvidenceType.INCIDENT_REPORT
            ]
        }
        
        # Determine control category
        control_text = f"{control.control_name} {control.requirement}".lower()
        category = 'general'
        
        if any(term in control_text for term in ['access', 'authentication', 'authorization']):
            category = 'access_control'
        elif any(term in control_text for term in ['data', 'encryption', 'protection']):
            category = 'data_protection'
        elif any(term in control_text for term in ['monitoring', 'logging', 'detection']):
            category = 'monitoring'
        elif any(term in control_text for term in ['incident', 'response', 'recovery']):
            category = 'incident_response'
        
        # Calculate completeness based on required evidence types
        required_types = control_categories.get(category, [])
        if not required_types:
            # General completeness based on variety
            unique_types = set(fragment.evidence_type for fragment in evidence_fragments)
            return min(1.0, len(unique_types) / 3)  # 3 types = full completeness
        
        # Check coverage of required types
        available_types = set(fragment.evidence_type for fragment in evidence_fragments)
        coverage = len(available_types.intersection(required_types)) / len(required_types)
        
        # Bonus for additional evidence types
        additional_types = available_types - set(required_types)
        bonus = min(0.2, len(additional_types) * 0.1)
        
        return min(1.0, coverage + bonus)
    
    def _calculate_relevance_score(self, control: ControlData, 
                                 evidence_fragments: List[EvidenceFragment]) -> float:
        """Calculate evidence relevance score"""
        if not evidence_fragments:
            return 0.0
        
        # Extract key terms from control
        control_keywords = self._extract_control_keywords(control)
        
        relevance_scores = []
        for fragment in evidence_fragments:
            # Base relevance from fragment
            relevance = fragment.relevance_score
            
            # Keyword matching bonus
            content_lower = fragment.content.lower()
            keyword_matches = sum(1 for keyword in control_keywords if keyword in content_lower)
            keyword_bonus = min(0.3, keyword_matches / len(control_keywords))
            
            # Technical depth bonus
            technical_bonus = min(0.2, len(fragment.technical_indicators) / 10)
            
            total_relevance = relevance + keyword_bonus + technical_bonus
            relevance_scores.append(min(1.0, total_relevance))
        
        return sum(relevance_scores) / len(relevance_scores)
    
    def _calculate_freshness_score(self, evidence_fragments: List[EvidenceFragment]) -> float:
        """Calculate evidence freshness score"""
        if not evidence_fragments:
            return 0.0
        
        freshness_scores = []
        current_date = datetime.now()
        
        for fragment in evidence_fragments:
            fragment_score = 0.5  # Base score
            
            # Check for freshness indicators
            content_lower = fragment.content.lower()
            for indicator in self.quality_indicators['freshness']:
                if re.search(indicator, content_lower):
                    fragment_score += 0.1
            
            # Check for outdated indicators
            for gap_type, patterns in self.gap_detectors.items():
                if gap_type == 'outdated_evidence':
                    for pattern in patterns:
                        if re.search(pattern, content_lower):
                            fragment_score -= 0.2
            
            # Try to parse dates if available
            for timestamp in fragment.timestamps:
                try:
                    # Simple date parsing (could be enhanced)
                    if len(timestamp) >= 4:  # At least a year
                        year = int(timestamp[-4:])
                        age = current_date.year - year
                        if age <= 1:
                            fragment_score += 0.3
                        elif age <= 2:
                            fragment_score += 0.1
                        elif age >= 5:
                            fragment_score -= 0.2
                except:
                    pass
            
            freshness_scores.append(max(0.0, min(1.0, fragment_score)))
        
        return sum(freshness_scores) / len(freshness_scores)
    
    def _calculate_technical_depth_score(self, evidence_fragments: List[EvidenceFragment]) -> float:
        """Calculate technical depth score"""
        if not evidence_fragments:
            return 0.0
        
        technical_scores = []
        for fragment in evidence_fragments:
            # Base score from technical indicators
            technical_score = min(0.5, len(fragment.technical_indicators) / 10)
            
            # Check for technical depth indicators
            content_lower = fragment.content.lower()
            for indicator in self.quality_indicators['technical_depth']:
                if re.search(indicator, content_lower):
                    technical_score += 0.1
            
            # Evidence type bonus
            if fragment.evidence_type in [EvidenceType.TECHNICAL_SPECIFICATION, 
                                        EvidenceType.CONFIGURATION_SCREENSHOT]:
                technical_score += 0.2
            
            technical_scores.append(min(1.0, technical_score))
        
        return sum(technical_scores) / len(technical_scores)
    
    def _calculate_credibility_score(self, evidence_fragments: List[EvidenceFragment]) -> float:
        """Calculate evidence credibility score"""
        if not evidence_fragments:
            return 0.0
        
        credibility_scores = []
        for fragment in evidence_fragments:
            credibility = 0.5  # Base credibility
            
            # Check for credibility indicators
            content_lower = fragment.content.lower()
            for indicator in self.quality_indicators['credibility']:
                if re.search(indicator, content_lower):
                    credibility += 0.1
            
            # Evidence type credibility
            type_credibility = {
                EvidenceType.COMPLIANCE_REPORT: 0.9,
                EvidenceType.AUDIT_LOG: 0.8,
                EvidenceType.POLICY_DOCUMENT: 0.7,
                EvidenceType.CONFIGURATION_SCREENSHOT: 0.8,
                EvidenceType.UNKNOWN: 0.3
            }
            
            base_credibility = type_credibility.get(fragment.evidence_type, 0.5)
            credibility = max(credibility, base_credibility)
            
            credibility_scores.append(min(1.0, credibility))
        
        return sum(credibility_scores) / len(credibility_scores)
    
    def _calculate_overall_quality_score(self, completeness: float, relevance: float,
                                       freshness: float, technical_depth: float,
                                       credibility: float) -> float:
        """Calculate overall quality score with weighted dimensions"""
        weights = {
            'completeness': 0.25,
            'relevance': 0.25,
            'credibility': 0.20,
            'technical_depth': 0.15,
            'freshness': 0.15
        }
        
        return (
            completeness * weights['completeness'] +
            relevance * weights['relevance'] +
            credibility * weights['credibility'] +
            technical_depth * weights['technical_depth'] +
            freshness * weights['freshness']
        )
    
    def _identify_evidence_types(self, evidence_fragments: List[EvidenceFragment]) -> List[str]:
        """Identify all evidence types present"""
        types = set()
        for fragment in evidence_fragments:
            types.add(fragment.evidence_type.value)
        return list(types)
    
    def _detect_evidence_gaps(self, control: ControlData, 
                            evidence_fragments: List[EvidenceFragment]) -> List[str]:
        """Detect evidence gaps and missing elements"""
        gaps = []
        
        # Check for missing evidence patterns
        all_content = ' '.join(fragment.content for fragment in evidence_fragments).lower()
        
        for gap_type, patterns in self.gap_detectors.items():
            for pattern in patterns:
                if re.search(pattern, all_content):
                    gap_description = self._format_gap_description(gap_type, pattern)
                    if gap_description not in gaps:
                        gaps.append(gap_description)
        
        # Check for missing evidence types
        available_types = set(fragment.evidence_type for fragment in evidence_fragments)
        
        # Determine what evidence types should be present
        control_text = f"{control.control_name} {control.requirement}".lower()
        
        if 'policy' in control_text and EvidenceType.POLICY_DOCUMENT not in available_types:
            gaps.append("Missing policy documentation")
        
        if 'configuration' in control_text and EvidenceType.CONFIGURATION_SCREENSHOT not in available_types:
            gaps.append("Missing configuration evidence")
        
        if 'audit' in control_text and EvidenceType.AUDIT_LOG not in available_types:
            gaps.append("Missing audit log evidence")
        
        if 'procedure' in control_text and EvidenceType.PROCEDURE_DOCUMENTATION not in available_types:
            gaps.append("Missing procedure documentation")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _detect_contradictions(self, evidence_fragments: List[EvidenceFragment]) -> List[str]:
        """Detect contradictions in evidence"""
        contradictions = []
        
        # Combine all content
        all_content = ' '.join(fragment.content for fragment in evidence_fragments).lower()
        
        # Check for contradictory patterns
        for contradiction_type, pattern_pairs in self.contradiction_patterns.items():
            for positive_pattern, negative_pattern in pattern_pairs:
                if re.search(positive_pattern, all_content) and re.search(negative_pattern, all_content):
                    contradiction_desc = f"Contradiction detected: {positive_pattern} vs {negative_pattern}"
                    contradictions.append(contradiction_desc)
        
        return contradictions[:3]  # Limit to top 3 contradictions
    
    def _generate_recommendations(self, control: ControlData, completeness: float, 
                                relevance: float, freshness: float, technical_depth: float,
                                credibility: float, evidence_gaps: List[str]) -> List[str]:
        """Generate recommendations for improving evidence quality"""
        recommendations = []
        
        # Completeness recommendations
        if completeness < 0.6:
            recommendations.append("Provide additional evidence types to demonstrate comprehensive control implementation")
        
        # Relevance recommendations
        if relevance < 0.6:
            recommendations.append("Ensure evidence directly addresses the specific control requirements")
        
        # Freshness recommendations
        if freshness < 0.6:
            recommendations.append("Update evidence with more recent documentation and configurations")
        
        # Technical depth recommendations
        if technical_depth < 0.6:
            recommendations.append("Include more technical details such as configurations, logs, and implementation specifics")
        
        # Credibility recommendations
        if credibility < 0.6:
            recommendations.append("Provide evidence from authoritative sources such as audit reports or official documentation")
        
        # Gap-specific recommendations
        for gap in evidence_gaps:
            if "policy" in gap.lower():
                recommendations.append("Provide formal policy documentation")
            elif "configuration" in gap.lower():
                recommendations.append("Include configuration screenshots or settings exports")
            elif "audit" in gap.lower():
                recommendations.append("Provide audit logs or monitoring evidence")
            elif "procedure" in gap.lower():
                recommendations.append("Document operational procedures and processes")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _extract_control_keywords(self, control: ControlData) -> List[str]:
        """Extract key terms from control for matching"""
        text = f"{control.control_name} {control.requirement}".lower()
        
        # Remove common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract meaningful words (3+ characters)
        words = re.findall(r'\b\w{3,}\b', text)
        keywords = [word for word in words if word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def _format_gap_description(self, gap_type: str, pattern: str) -> str:
        """Format gap description for user readability"""
        gap_descriptions = {
            'missing_implementation': "Implementation evidence is missing or inadequate",
            'missing_documentation': "Formal documentation is not provided",
            'missing_testing': "Testing or validation evidence is absent",
            'missing_monitoring': "Monitoring or tracking evidence is insufficient",
            'outdated_evidence': "Evidence appears to be outdated or obsolete"
        }
        
        return gap_descriptions.get(gap_type, f"Evidence gap detected: {gap_type}")
    
    def export_evidence_report(self, metrics: EvidenceMetrics, 
                             output_path: str = "evidence_report.json"):
        """Export detailed evidence quality report"""
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            logger.info(f"Evidence quality report exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export evidence report: {e}")
    
    def analyze_evidence_trends(self, evidence_history: List[EvidenceMetrics]) -> Dict[str, any]:
        """Analyze trends in evidence quality over time"""
        if len(evidence_history) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trends for each dimension
        trends = {}
        
        for dimension in ['overall_quality_score', 'completeness_score', 'relevance_score', 
                         'freshness_score', 'technical_depth_score', 'credibility_score']:
            values = [getattr(metrics, dimension) for metrics in evidence_history]
            
            if len(values) >= 2:
                trend = (values[-1] - values[0]) / len(values)
                trends[dimension] = {
                    'trend': 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable',
                    'slope': trend,
                    'current_value': values[-1],
                    'historical_average': sum(values) / len(values)
                }
        
        return trends