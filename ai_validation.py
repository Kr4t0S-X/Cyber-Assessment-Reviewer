"""
AI Response Validation Module for Cyber Assessment Reviewer
Implements quality assurance and validation mechanisms for AI-generated assessments
Enhanced with multi-dimensional accuracy metrics and advanced validation
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from models import AssessmentResult, ControlData
from accuracy_engine import AccuracyDimension, EnhancedAccuracyEngine
from evidence_analyzer import AdvancedEvidenceAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of AI response validation"""
    is_valid: bool
    confidence_adjustment: float  # Factor to adjust confidence score
    validation_issues: List[str]
    quality_score: float  # 0.0 to 1.0
    recommendations: List[str]
    # Enhanced validation metrics
    accuracy_dimensions: Dict[str, float] = None
    evidence_quality_score: float = 0.0
    technical_specificity_score: float = 0.0
    consistency_score: float = 0.0
    
    def __post_init__(self):
        if self.accuracy_dimensions is None:
            self.accuracy_dimensions = {}

class AIResponseValidator:
    """Validates AI responses for quality, completeness, and accuracy"""
    
    # Minimum requirements for valid responses
    MIN_FINDINGS_COUNT = 1
    MIN_RISKS_COUNT = 1
    MIN_REMEDIATION_COUNT = 1
    MIN_RATIONALE_LENGTH = 50
    
    def __init__(self, config=None):
        self.config = config
        self.accuracy_engine = EnhancedAccuracyEngine(config)
        self.evidence_analyzer = AdvancedEvidenceAnalyzer(config)
    
    # Quality indicators
    QUALITY_INDICATORS = {
        'specific_technical_terms': [
            'configuration', 'implementation', 'policy', 'procedure', 'control',
            'authentication', 'authorization', 'encryption', 'monitoring', 'logging',
            'vulnerability', 'patch', 'access', 'network', 'firewall', 'certificate'
        ],
        'evidence_references': [
            'evidence shows', 'document indicates', 'configuration demonstrates',
            'policy states', 'procedure outlines', 'log shows', 'screenshot shows',
            'report indicates', 'assessment reveals', 'testing confirms'
        ],
        'risk_language': [
            'risk', 'threat', 'vulnerability', 'exposure', 'impact', 'likelihood',
            'consequence', 'mitigation', 'remediation', 'control gap'
        ]
    }
    
    def __init__(self, config=None):
        self.config = config
        self.accuracy_engine = EnhancedAccuracyEngine(config)
        self.evidence_analyzer = AdvancedEvidenceAnalyzer(config)
        self.validation_rules = [
            self._validate_completeness,
            self._validate_evidence_support,
            self._validate_technical_specificity,
            self._validate_risk_assessment,
            self._validate_remediation_quality,
            self._validate_consistency,
            self._validate_compliance_rationale,
            self._validate_enhanced_accuracy_dimensions
        ]
    
    def validate_response(self, result: AssessmentResult, evidence_text: str, 
                         control: ControlData, framework: str = "NIST") -> ValidationResult:
        """Comprehensive validation of AI response with enhanced metrics"""
        validation_issues = []
        quality_scores = []
        recommendations = []
        
        # Run all validation rules
        for rule in self.validation_rules:
            try:
                issues, score, recs = rule(result, evidence_text, control)
                validation_issues.extend(issues)
                quality_scores.append(score)
                recommendations.extend(recs)
            except Exception as e:
                logger.error(f"Validation rule failed: {e}")
                quality_scores.append(0.5)  # Neutral score for failed validation
        
        # Calculate overall quality score
        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Enhanced validation with accuracy dimensions
        accuracy_dimensions = self._calculate_accuracy_dimensions(result, evidence_text, control, framework)
        
        # Calculate evidence quality score
        evidence_quality_score = self._calculate_evidence_quality_score(evidence_text, control)
        
        # Calculate technical specificity score
        technical_specificity_score = self._calculate_technical_specificity_score(result)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(result)
        
        # Determine if response is valid (enhanced criteria)
        is_valid = (
            len(validation_issues) == 0 and 
            overall_quality >= 0.6 and
            evidence_quality_score >= 0.4 and
            technical_specificity_score >= 0.3
        )
        
        # Calculate confidence adjustment (enhanced)
        confidence_adjustment = self._calculate_enhanced_confidence_adjustment(
            overall_quality, validation_issues, accuracy_dimensions, evidence_quality_score
        )
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            validation_issues=validation_issues,
            quality_score=overall_quality,
            recommendations=list(set(recommendations)),  # Remove duplicates
            accuracy_dimensions=accuracy_dimensions,
            evidence_quality_score=evidence_quality_score,
            technical_specificity_score=technical_specificity_score,
            consistency_score=consistency_score
        )
    
    def _validate_completeness(self, result: AssessmentResult, evidence_text: str, 
                              control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate that response contains all required fields"""
        issues = []
        recommendations = []
        
        # Check required fields
        if len(result.key_findings) < self.MIN_FINDINGS_COUNT:
            issues.append("Insufficient key findings provided")
            recommendations.append("Provide more specific findings based on evidence")
        
        if len(result.risks) < self.MIN_RISKS_COUNT:
            issues.append("Insufficient risk identification")
            recommendations.append("Identify specific security risks related to this control")
        
        if len(result.remediation) < self.MIN_REMEDIATION_COUNT:
            issues.append("Insufficient remediation steps")
            recommendations.append("Provide actionable remediation steps")
        
        if len(result.compliance_rationale) < self.MIN_RATIONALE_LENGTH:
            issues.append("Compliance rationale too brief or missing")
            recommendations.append("Provide detailed explanation for compliance determination")
        
        # Check for empty or generic responses
        generic_phrases = ['not specified', 'not provided', 'unclear', 'unknown', 'general']
        for finding in result.key_findings:
            if any(phrase in finding.lower() for phrase in generic_phrases):
                issues.append("Generic or vague findings detected")
                recommendations.append("Provide specific, evidence-based findings")
                break
        
        quality_score = max(0.0, 1.0 - (len(issues) * 0.2))
        return issues, quality_score, recommendations
    
    def _validate_evidence_support(self, result: AssessmentResult, evidence_text: str, 
                                  control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate that conclusions are supported by evidence"""
        issues = []
        recommendations = []
        
        # Check if findings reference evidence
        evidence_ref_count = 0
        for finding in result.key_findings + result.risks + result.remediation:
            if any(indicator in finding.lower() for indicator in self.QUALITY_INDICATORS['evidence_references']):
                evidence_ref_count += 1
        
        if evidence_ref_count == 0:
            issues.append("Findings do not reference specific evidence")
            recommendations.append("Reference specific evidence when making findings")
        
        # Check compliance determination alignment
        if result.compliance_status == "Compliant" and result.risk_level in ["Critical", "High"]:
            issues.append("Inconsistent compliance status and risk level")
            recommendations.append("Ensure compliance determination aligns with identified risks")
        
        if result.evidence_validity == "Invalid" and result.compliance_status == "Compliant":
            issues.append("Cannot be compliant with invalid evidence")
            recommendations.append("Review compliance determination when evidence is invalid")
        
        quality_score = max(0.0, 1.0 - (len(issues) * 0.25))
        return issues, quality_score, recommendations
    
    def _validate_technical_specificity(self, result: AssessmentResult, evidence_text: str, 
                                       control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate technical specificity and depth of analysis"""
        issues = []
        recommendations = []
        
        # Count technical terms in response
        all_response_text = ' '.join(result.key_findings + result.risks + result.remediation + 
                                   result.technical_details + [result.compliance_rationale])
        
        technical_term_count = sum(1 for term in self.QUALITY_INDICATORS['specific_technical_terms'] 
                                 if term in all_response_text.lower())
        
        if technical_term_count < 3:
            issues.append("Insufficient technical specificity")
            recommendations.append("Include more specific technical details and terminology")
        
        # Check for technical details field usage
        if len(result.technical_details) == 0:
            issues.append("No technical details provided")
            recommendations.append("Extract and document specific technical configurations or implementations")
        
        quality_score = min(1.0, technical_term_count / 5.0)  # Scale based on technical terms
        return issues, quality_score, recommendations
    
    def _validate_risk_assessment(self, result: AssessmentResult, evidence_text: str, 
                                 control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate quality of risk assessment"""
        issues = []
        recommendations = []
        
        # Check risk language usage
        risk_text = ' '.join(result.risks + [result.compliance_rationale])
        risk_term_count = sum(1 for term in self.QUALITY_INDICATORS['risk_language'] 
                            if term in risk_text.lower())
        
        if risk_term_count < 2:
            issues.append("Insufficient risk analysis language")
            recommendations.append("Use more specific risk assessment terminology")
        
        # Validate risk level consistency
        high_risk_indicators = ['critical', 'severe', 'major', 'significant']
        low_risk_indicators = ['minor', 'low', 'minimal', 'negligible']
        
        risk_content = risk_text.lower()
        if result.risk_level in ["Critical", "High"]:
            if not any(indicator in risk_content for indicator in high_risk_indicators):
                issues.append("Risk level not supported by risk description")
                recommendations.append("Ensure risk descriptions support the assigned risk level")
        
        quality_score = min(1.0, risk_term_count / 3.0)
        return issues, quality_score, recommendations
    
    def _validate_remediation_quality(self, result: AssessmentResult, evidence_text: str, 
                                     control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate quality and actionability of remediation steps"""
        issues = []
        recommendations = []
        
        # Check for actionable language
        actionable_verbs = ['implement', 'configure', 'enable', 'disable', 'update', 
                           'install', 'review', 'establish', 'develop', 'document']
        
        actionable_count = 0
        for remediation in result.remediation:
            if any(verb in remediation.lower() for verb in actionable_verbs):
                actionable_count += 1
        
        if actionable_count < len(result.remediation) * 0.5:
            issues.append("Remediation steps lack actionable language")
            recommendations.append("Use specific action verbs in remediation steps")
        
        # Check for prioritization
        priority_indicators = ['first', 'immediately', 'urgent', 'priority', 'critical']
        has_prioritization = any(indicator in ' '.join(result.remediation).lower() 
                               for indicator in priority_indicators)
        
        if not has_prioritization and len(result.remediation) > 1:
            recommendations.append("Consider prioritizing remediation steps")
        
        quality_score = actionable_count / max(1, len(result.remediation))
        return issues, quality_score, recommendations
    
    def _validate_consistency(self, result: AssessmentResult, evidence_text: str, 
                             control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate internal consistency of the assessment"""
        issues = []
        recommendations = []
        
        # Check confidence score alignment
        if result.confidence_score > 0.8 and result.evidence_validity in ["Invalid", "No Evidence"]:
            issues.append("High confidence with invalid/no evidence")
            recommendations.append("Lower confidence when evidence is insufficient")
        
        if result.confidence_score < 0.3 and result.evidence_validity == "Valid":
            issues.append("Low confidence with valid evidence")
            recommendations.append("Increase confidence when evidence is strong")
        
        # Check evidence gaps vs compliance status
        if len(result.evidence_gaps) > 2 and result.compliance_status == "Compliant":
            issues.append("Multiple evidence gaps but marked compliant")
            recommendations.append("Consider partial compliance when evidence gaps exist")
        
        quality_score = 1.0 - (len(issues) * 0.3)
        return issues, quality_score, recommendations
    
    def _validate_compliance_rationale(self, result: AssessmentResult, evidence_text: str, 
                                      control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate the quality of compliance rationale"""
        issues = []
        recommendations = []
        
        rationale = result.compliance_rationale.lower()
        
        # Check for evidence references in rationale
        if not any(ref in rationale for ref in ['evidence', 'document', 'shows', 'demonstrates']):
            issues.append("Rationale does not reference evidence")
            recommendations.append("Include specific evidence references in compliance rationale")
        
        # Check for control requirement references
        control_keywords = control.requirement.lower().split()[:5]  # First 5 words
        if not any(keyword in rationale for keyword in control_keywords if len(keyword) > 3):
            issues.append("Rationale does not address control requirements")
            recommendations.append("Ensure rationale addresses specific control requirements")
        
        quality_score = len(rationale) / 200.0  # Score based on detail level
        quality_score = min(1.0, quality_score)
        
        return issues, quality_score, recommendations
    
    def _calculate_confidence_adjustment(self, quality_score: float, issues: List[str]) -> float:
        """Calculate confidence score adjustment based on validation results"""
        base_adjustment = quality_score  # Start with quality score
        
        # Reduce confidence for critical issues
        critical_issue_count = sum(1 for issue in issues if any(word in issue.lower() 
                                 for word in ['inconsistent', 'invalid', 'insufficient']))
        
        adjustment = base_adjustment - (critical_issue_count * 0.1)
        return max(0.1, min(1.0, adjustment))  # Keep between 0.1 and 1.0
    
    # Enhanced validation methods
    def _validate_enhanced_accuracy_dimensions(self, result: AssessmentResult, evidence_text: str, 
                                             control: ControlData) -> Tuple[List[str], float, List[str]]:
        """Validate using enhanced accuracy dimensions"""
        issues = []
        recommendations = []
        
        # Calculate accuracy dimensions for validation
        accuracy_dimensions = self._calculate_accuracy_dimensions(result, evidence_text, control, "NIST")
        
        # Check each dimension against thresholds
        dimension_thresholds = {
            'compliance_accuracy': 0.6,
            'risk_assessment_accuracy': 0.5,
            'finding_quality': 0.5,
            'evidence_utilization': 0.4,
            'technical_specificity': 0.3
        }
        
        for dimension, score in accuracy_dimensions.items():
            threshold = dimension_thresholds.get(dimension, 0.5)
            if score < threshold:
                issues.append(f"Low {dimension.replace('_', ' ')}: {score:.2f} < {threshold}")
                recommendations.append(f"Improve {dimension.replace('_', ' ')} in AI response")
        
        # Overall accuracy dimension score
        overall_score = sum(accuracy_dimensions.values()) / len(accuracy_dimensions) if accuracy_dimensions else 0.5
        
        return issues, overall_score, recommendations
    
    def _calculate_accuracy_dimensions(self, result: AssessmentResult, evidence_text: str, 
                                     control: ControlData, framework: str) -> Dict[str, float]:
        """Calculate accuracy dimensions for validation"""
        dimensions = {}
        
        # Compliance accuracy (based on alignment with evidence)
        dimensions['compliance_accuracy'] = self._assess_compliance_accuracy(result, evidence_text, control)
        
        # Risk assessment accuracy (based on risk level consistency)
        dimensions['risk_assessment_accuracy'] = self._assess_risk_accuracy(result, control)
        
        # Finding quality (based on specificity and relevance)
        dimensions['finding_quality'] = self._assess_finding_quality(result, evidence_text)
        
        # Evidence utilization (based on how well evidence is used)
        dimensions['evidence_utilization'] = self._assess_evidence_utilization(result, evidence_text)
        
        # Technical specificity (based on technical depth)
        dimensions['technical_specificity'] = self._assess_technical_specificity(result)
        
        return dimensions
    
    def _calculate_evidence_quality_score(self, evidence_text: str, control: ControlData) -> float:
        """Calculate evidence quality score"""
        if not evidence_text:
            return 0.0
        
        # Use evidence analyzer to get quality score
        try:
            # Create dummy evidence structure for analysis
            evidence_texts = {"validation_evidence": {"main": evidence_text}}
            evidence_metrics = self.evidence_analyzer.analyze_evidence_quality(control, evidence_texts)
            return evidence_metrics.overall_quality_score
        except Exception as e:
            logger.warning(f"Evidence quality analysis failed: {e}")
            return 0.5  # Default score
    
    def _calculate_technical_specificity_score(self, result: AssessmentResult) -> float:
        """Calculate technical specificity score"""
        if not result.technical_details:
            return 0.2
        
        # Score based on technical details presence and quality
        technical_score = 0.0
        
        # Base score for having technical details
        technical_score += 0.3
        
        # Score based on number of technical details
        technical_score += min(0.4, len(result.technical_details) * 0.1)
        
        # Score based on technical term density
        technical_terms = [
            'configuration', 'implementation', 'protocol', 'algorithm', 'encryption',
            'authentication', 'authorization', 'monitoring', 'logging', 'compliance'
        ]
        
        all_technical_text = ' '.join(result.technical_details).lower()
        term_count = sum(1 for term in technical_terms if term in all_technical_text)
        technical_score += min(0.3, term_count * 0.05)
        
        return min(1.0, technical_score)
    
    def _calculate_consistency_score(self, result: AssessmentResult) -> float:
        """Calculate consistency score"""
        consistency_score = 1.0
        
        # Check compliance-risk alignment
        if result.compliance_status == "Compliant" and result.risk_level in ["Critical", "High"]:
            consistency_score -= 0.3
        
        if result.compliance_status == "Non-Compliant" and result.risk_level in ["Low"]:
            consistency_score -= 0.3
        
        # Check evidence-confidence alignment
        if result.evidence_validity == "Invalid" and result.confidence_score > 0.7:
            consistency_score -= 0.2
        
        # Check evidence gaps vs compliance
        if len(result.evidence_gaps) > 2 and result.compliance_status == "Compliant":
            consistency_score -= 0.2
        
        return max(0.0, consistency_score)
    
    def _calculate_enhanced_confidence_adjustment(self, quality_score: float, issues: List[str],
                                                accuracy_dimensions: Dict[str, float], 
                                                evidence_quality_score: float) -> float:
        """Calculate enhanced confidence adjustment"""
        base_adjustment = quality_score
        
        # Adjust for accuracy dimensions
        avg_accuracy = sum(accuracy_dimensions.values()) / len(accuracy_dimensions) if accuracy_dimensions else 0.5
        accuracy_adjustment = avg_accuracy * 0.3
        
        # Adjust for evidence quality
        evidence_adjustment = evidence_quality_score * 0.2
        
        # Reduce for critical issues
        critical_issue_count = sum(1 for issue in issues if any(word in issue.lower() 
                                 for word in ['inconsistent', 'invalid', 'insufficient', 'low']))
        
        adjustment = base_adjustment + accuracy_adjustment + evidence_adjustment - (critical_issue_count * 0.1)
        return max(0.1, min(1.0, adjustment))
    
    def _assess_compliance_accuracy(self, result: AssessmentResult, evidence_text: str, 
                                  control: ControlData) -> float:
        """Assess compliance accuracy based on evidence alignment"""
        if not evidence_text:
            return 0.3
        
        # Simple assessment based on evidence support
        evidence_lower = evidence_text.lower()
        
        # Positive compliance indicators
        positive_indicators = ['compliant', 'implemented', 'configured', 'established', 'documented']
        negative_indicators = ['non-compliant', 'not implemented', 'missing', 'absent', 'lacking']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in evidence_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in evidence_lower)
        
        # Assess alignment
        if result.compliance_status == "Compliant" and positive_count > negative_count:
            return 0.8
        elif result.compliance_status == "Non-Compliant" and negative_count > positive_count:
            return 0.8
        elif result.compliance_status == "Partially Compliant":
            return 0.6
        else:
            return 0.4
    
    def _assess_risk_accuracy(self, result: AssessmentResult, control: ControlData) -> float:
        """Assess risk assessment accuracy"""
        # Risk level consistency with compliance status
        if result.compliance_status == "Compliant" and result.risk_level in ["Low", "Medium"]:
            return 0.8
        elif result.compliance_status == "Non-Compliant" and result.risk_level in ["High", "Critical"]:
            return 0.8
        elif result.compliance_status == "Partially Compliant" and result.risk_level in ["Medium", "High"]:
            return 0.7
        else:
            return 0.5
    
    def _assess_finding_quality(self, result: AssessmentResult, evidence_text: str) -> float:
        """Assess finding quality based on specificity and relevance"""
        if not result.key_findings:
            return 0.2
        
        quality_score = 0.0
        
        # Base score for having findings
        quality_score += 0.3
        
        # Score based on finding specificity
        for finding in result.key_findings:
            if any(term in finding.lower() for term in ['specific', 'detailed', 'configured', 'implemented']):
                quality_score += 0.1
        
        # Score based on evidence references
        if evidence_text:
            for finding in result.key_findings:
                # Check if finding references evidence concepts
                if any(word in finding.lower() for word in evidence_text.lower().split()[:20]):
                    quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _assess_evidence_utilization(self, result: AssessmentResult, evidence_text: str) -> float:
        """Assess evidence utilization quality"""
        if not evidence_text:
            return 0.0
        
        utilization_score = 0.0
        
        # Score based on evidence references
        if result.evidence_references:
            utilization_score += 0.4
        
        # Score based on technical details extraction
        if result.technical_details:
            utilization_score += 0.3
        
        # Score based on evidence gaps identification
        if result.evidence_gaps:
            utilization_score += 0.3
        
        return min(1.0, utilization_score)
    
    def _assess_technical_specificity(self, result: AssessmentResult) -> float:
        """Assess technical specificity of the response"""
        return self._calculate_technical_specificity_score(result)
