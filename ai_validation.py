"""
AI Response Validation Module for Cyber Assessment Reviewer
Implements quality assurance and validation mechanisms for AI-generated assessments
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from models import AssessmentResult, ControlData

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of AI response validation"""
    is_valid: bool
    confidence_adjustment: float  # Factor to adjust confidence score
    validation_issues: List[str]
    quality_score: float  # 0.0 to 1.0
    recommendations: List[str]

class AIResponseValidator:
    """Validates AI responses for quality, completeness, and accuracy"""
    
    # Minimum requirements for valid responses
    MIN_FINDINGS_COUNT = 1
    MIN_RISKS_COUNT = 1
    MIN_REMEDIATION_COUNT = 1
    MIN_RATIONALE_LENGTH = 50
    
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
    
    def __init__(self):
        self.validation_rules = [
            self._validate_completeness,
            self._validate_evidence_support,
            self._validate_technical_specificity,
            self._validate_risk_assessment,
            self._validate_remediation_quality,
            self._validate_consistency,
            self._validate_compliance_rationale
        ]
    
    def validate_response(self, result: AssessmentResult, evidence_text: str, 
                         control: ControlData) -> ValidationResult:
        """Comprehensive validation of AI response"""
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
        
        # Determine if response is valid
        is_valid = len(validation_issues) == 0 and overall_quality >= 0.6
        
        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(overall_quality, validation_issues)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            validation_issues=validation_issues,
            quality_score=overall_quality,
            recommendations=list(set(recommendations))  # Remove duplicates
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
