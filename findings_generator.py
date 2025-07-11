"""
Enhanced Findings Generation System for Cyber Assessment Reviewer
Implements advanced finding classification, technical specificity, and actionable insights
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import json

from models import ControlData, AssessmentResult

logger = logging.getLogger(__name__)

class FindingType(Enum):
    """Types of findings that can be generated"""
    COMPLIANCE_FINDING = "compliance_finding"
    TECHNICAL_FINDING = "technical_finding"
    PROCEDURAL_FINDING = "procedural_finding"
    DOCUMENTATION_FINDING = "documentation_finding"
    CONFIGURATION_FINDING = "configuration_finding"
    MONITORING_FINDING = "monitoring_finding"
    RISK_FINDING = "risk_finding"
    GAP_FINDING = "gap_finding"

class FindingSeverity(Enum):
    """Severity levels for findings"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class FindingCategory(Enum):
    """Categories for organizing findings"""
    SECURITY_CONTROL = "security_control"
    POLICY_GOVERNANCE = "policy_governance"
    TECHNICAL_IMPLEMENTATION = "technical_implementation"
    OPERATIONAL_PROCESS = "operational_process"
    COMPLIANCE_ADHERENCE = "compliance_adherence"
    RISK_MANAGEMENT = "risk_management"

@dataclass
class EnhancedFinding:
    """Enhanced finding with detailed metadata"""
    finding_id: str
    finding_type: FindingType
    severity: FindingSeverity
    category: FindingCategory
    title: str
    description: str
    evidence_references: List[str]
    technical_details: List[str]
    impact_analysis: str
    remediation_steps: List[str]
    remediation_priority: str
    effort_estimate: str
    confidence_score: float
    framework_mappings: Dict[str, str]
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'finding_id': self.finding_id,
            'finding_type': self.finding_type.value,
            'severity': self.severity.value,
            'category': self.category.value,
            'title': self.title,
            'description': self.description,
            'evidence_references': self.evidence_references,
            'technical_details': self.technical_details,
            'impact_analysis': self.impact_analysis,
            'remediation_steps': self.remediation_steps,
            'remediation_priority': self.remediation_priority,
            'effort_estimate': self.effort_estimate,
            'confidence_score': self.confidence_score,
            'framework_mappings': self.framework_mappings
        }

@dataclass
class FindingCorrelation:
    """Correlation between findings"""
    finding_ids: List[str]
    correlation_type: str
    correlation_strength: float
    description: str

class EnhancedFindingsGenerator:
    """Advanced findings generation system"""
    
    def __init__(self, config=None):
        self.config = config
        self.finding_templates = self._initialize_finding_templates()
        self.technical_indicators = self._initialize_technical_indicators()
        self.severity_rules = self._initialize_severity_rules()
        self.remediation_templates = self._initialize_remediation_templates()
        self.impact_patterns = self._initialize_impact_patterns()
        
    def _initialize_finding_templates(self) -> Dict[FindingType, Dict[str, str]]:
        """Initialize finding templates for different types"""
        return {
            FindingType.COMPLIANCE_FINDING: {
                'positive': "Control {control_id} demonstrates {compliance_aspect} through {evidence_summary}",
                'negative': "Control {control_id} shows {compliance_gap} with insufficient {missing_element}",
                'partial': "Control {control_id} partially meets requirements but lacks {improvement_area}"
            },
            FindingType.TECHNICAL_FINDING: {
                'configuration': "Technical analysis reveals {technical_aspect} in {system_component}",
                'implementation': "Implementation review shows {implementation_status} of {security_mechanism}",
                'architecture': "System architecture demonstrates {architectural_finding} affecting {security_domain}"
            },
            FindingType.PROCEDURAL_FINDING: {
                'process': "Procedural review identifies {process_status} in {operational_area}",
                'workflow': "Workflow analysis shows {workflow_finding} affecting {business_process}",
                'governance': "Governance review reveals {governance_aspect} in {management_area}"
            },
            FindingType.DOCUMENTATION_FINDING: {
                'missing': "Documentation gap identified: {missing_documentation} for {control_area}",
                'outdated': "Documentation review shows {outdated_elements} requiring update",
                'inadequate': "Documentation quality issue: {quality_concern} in {document_type}"
            },
            FindingType.CONFIGURATION_FINDING: {
                'secure': "Configuration analysis shows {security_configuration} properly implemented",
                'insecure': "Configuration review reveals {security_weakness} in {system_component}",
                'missing': "Configuration gap: {missing_configuration} not found in {target_system}"
            },
            FindingType.MONITORING_FINDING: {
                'effective': "Monitoring analysis shows {monitoring_capability} effectively implemented",
                'gaps': "Monitoring gaps identified: {monitoring_deficiency} in {monitored_area}",
                'inadequate': "Monitoring review reveals {monitoring_weakness} affecting {security_visibility}"
            },
            FindingType.RISK_FINDING: {
                'high': "Risk analysis identifies {risk_factor} with {impact_level} potential impact",
                'mitigated': "Risk assessment shows {risk_area} adequately mitigated through {mitigation_approach}",
                'unmitigated': "Unmitigated risk identified: {risk_description} in {risk_domain}"
            },
            FindingType.GAP_FINDING: {
                'implementation': "Implementation gap: {gap_description} between {expected_state} and {actual_state}",
                'coverage': "Coverage gap identified: {coverage_deficiency} in {security_domain}",
                'maturity': "Maturity gap: {maturity_finding} indicating {improvement_opportunity}"
            }
        }
    
    def _initialize_technical_indicators(self) -> Dict[str, List[str]]:
        """Initialize technical indicators for different domains"""
        return {
            'access_control': [
                'multi-factor authentication', 'role-based access control', 'privileged access management',
                'access control matrix', 'identity management', 'authentication protocols',
                'authorization mechanisms', 'access reviews', 'account lifecycle'
            ],
            'encryption': [
                'encryption algorithms', 'key management', 'certificate management',
                'cryptographic protocols', 'data at rest encryption', 'data in transit encryption',
                'encryption strength', 'key rotation', 'cryptographic standards'
            ],
            'monitoring': [
                'log aggregation', 'security information and event management', 'intrusion detection',
                'behavior analytics', 'threat intelligence', 'correlation rules',
                'alerting mechanisms', 'incident detection', 'anomaly detection'
            ],
            'network_security': [
                'firewall rules', 'network segmentation', 'intrusion prevention',
                'network monitoring', 'traffic analysis', 'network access control',
                'virtual private networks', 'network architecture', 'perimeter security'
            ],
            'vulnerability_management': [
                'vulnerability scanning', 'patch management', 'security testing',
                'penetration testing', 'vulnerability assessment', 'remediation tracking',
                'risk prioritization', 'security baseline', 'configuration management'
            ],
            'incident_response': [
                'incident detection', 'incident classification', 'response procedures',
                'escalation processes', 'forensic capabilities', 'recovery procedures',
                'communication plans', 'lessons learned', 'incident metrics'
            ],
            'data_protection': [
                'data classification', 'data loss prevention', 'data retention',
                'data disposal', 'privacy controls', 'data masking',
                'data backup', 'data recovery', 'data integrity'
            ]
        }
    
    def _initialize_severity_rules(self) -> Dict[str, Dict[str, any]]:
        """Initialize rules for determining finding severity"""
        return {
            'critical': {
                'keywords': ['critical vulnerability', 'complete failure', 'no implementation', 'severe risk'],
                'impact_levels': ['critical', 'severe', 'catastrophic'],
                'risk_factors': ['data breach', 'system compromise', 'regulatory violation'],
                'base_score': 0.9
            },
            'high': {
                'keywords': ['significant gap', 'major deficiency', 'high risk', 'substantial weakness'],
                'impact_levels': ['high', 'significant', 'major'],
                'risk_factors': ['unauthorized access', 'data exposure', 'compliance violation'],
                'base_score': 0.7
            },
            'medium': {
                'keywords': ['moderate risk', 'partial implementation', 'improvement needed'],
                'impact_levels': ['medium', 'moderate', 'noticeable'],
                'risk_factors': ['limited exposure', 'process inefficiency', 'minor compliance gap'],
                'base_score': 0.5
            },
            'low': {
                'keywords': ['minor issue', 'best practice', 'enhancement opportunity'],
                'impact_levels': ['low', 'minor', 'minimal'],
                'risk_factors': ['administrative burden', 'minor inefficiency', 'cosmetic issue'],
                'base_score': 0.3
            },
            'informational': {
                'keywords': ['observation', 'recommendation', 'best practice'],
                'impact_levels': ['informational', 'negligible', 'none'],
                'risk_factors': ['process improvement', 'efficiency gain', 'documentation'],
                'base_score': 0.1
            }
        }
    
    def _initialize_remediation_templates(self) -> Dict[str, List[str]]:
        """Initialize remediation templates for different categories"""
        return {
            'implementation': [
                "Implement {security_control} in {target_system}",
                "Deploy {security_technology} to address {security_gap}",
                "Configure {system_component} according to {security_standard}",
                "Establish {security_process} for {operational_area}"
            ],
            'documentation': [
                "Document {process_or_procedure} including {specific_requirements}",
                "Create {document_type} covering {scope_area}",
                "Update {existing_document} to reflect {current_state}",
                "Develop {policy_or_standard} for {governance_area}"
            ],
            'configuration': [
                "Reconfigure {system_component} to enable {security_feature}",
                "Adjust {configuration_parameter} to improve {security_aspect}",
                "Disable {insecure_feature} in {target_system}",
                "Enable {security_setting} across {system_scope}"
            ],
            'process': [
                "Establish {operational_process} for {business_function}",
                "Implement {governance_process} in {organizational_unit}",
                "Create {workflow_process} to manage {operational_activity}",
                "Develop {management_process} for {oversight_area}"
            ],
            'monitoring': [
                "Implement {monitoring_capability} for {monitored_area}",
                "Configure {monitoring_tool} to detect {security_event}",
                "Establish {alerting_mechanism} for {critical_condition}",
                "Deploy {detection_technology} in {network_segment}"
            ],
            'training': [
                "Provide {training_type} to {target_audience}",
                "Conduct {awareness_program} focusing on {security_topic}",
                "Develop {educational_material} covering {subject_area}",
                "Implement {skill_development} program for {technical_staff}"
            ]
        }
    
    def _initialize_impact_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for impact analysis"""
        return {
            'business_impact': [
                "operational disruption", "financial loss", "reputational damage",
                "competitive disadvantage", "customer impact", "service availability"
            ],
            'security_impact': [
                "data breach", "unauthorized access", "system compromise",
                "data loss", "privacy violation", "security incident"
            ],
            'compliance_impact': [
                "regulatory violation", "audit finding", "compliance failure",
                "legal exposure", "penalty risk", "certification loss"
            ],
            'operational_impact': [
                "process inefficiency", "manual effort", "resource consumption",
                "performance degradation", "maintenance overhead", "user experience"
            ]
        }
    
    def generate_enhanced_findings(self, control: ControlData, assessment_result: AssessmentResult,
                                 evidence_quality_score: float = 0.5,
                                 framework: str = "NIST") -> List[EnhancedFinding]:
        """Generate enhanced findings with technical specificity"""
        
        findings = []
        
        # Generate primary compliance finding
        compliance_finding = self._generate_compliance_finding(
            control, assessment_result, evidence_quality_score, framework
        )
        findings.append(compliance_finding)
        
        # Generate technical findings
        technical_findings = self._generate_technical_findings(
            control, assessment_result, evidence_quality_score, framework
        )
        findings.extend(technical_findings)
        
        # Generate procedural findings
        procedural_findings = self._generate_procedural_findings(
            control, assessment_result, evidence_quality_score, framework
        )
        findings.extend(procedural_findings)
        
        # Generate gap findings
        gap_findings = self._generate_gap_findings(
            control, assessment_result, evidence_quality_score, framework
        )
        findings.extend(gap_findings)
        
        # Generate risk findings
        risk_findings = self._generate_risk_findings(
            control, assessment_result, evidence_quality_score, framework
        )
        findings.extend(risk_findings)
        
        # Enhance findings with correlations
        findings = self._enhance_findings_with_correlations(findings)
        
        return findings
    
    def _generate_compliance_finding(self, control: ControlData, assessment_result: AssessmentResult,
                                   evidence_quality_score: float, framework: str) -> EnhancedFinding:
        """Generate primary compliance finding"""
        
        finding_id = f"{control.control_id}_COMPLIANCE_001"
        
        # Determine compliance status and severity
        severity = self._determine_compliance_severity(assessment_result.compliance_status, 
                                                     assessment_result.risk_level)
        
        # Generate finding title and description
        title = self._generate_compliance_title(control, assessment_result)
        description = self._generate_compliance_description(control, assessment_result, evidence_quality_score)
        
        # Extract technical details
        technical_details = self._extract_technical_details(assessment_result, control)
        
        # Generate impact analysis
        impact_analysis = self._generate_impact_analysis(control, assessment_result)
        
        # Generate remediation steps
        remediation_steps = self._generate_remediation_steps(control, assessment_result)
        
        # Determine effort estimate
        effort_estimate = self._estimate_remediation_effort(assessment_result.compliance_status, 
                                                          len(remediation_steps))
        
        # Calculate confidence score
        confidence_score = self._calculate_finding_confidence(assessment_result, evidence_quality_score)
        
        # Generate framework mappings
        framework_mappings = self._generate_framework_mappings(control, framework)
        
        return EnhancedFinding(
            finding_id=finding_id,
            finding_type=FindingType.COMPLIANCE_FINDING,
            severity=severity,
            category=FindingCategory.COMPLIANCE_ADHERENCE,
            title=title,
            description=description,
            evidence_references=assessment_result.evidence_references[:3],
            technical_details=technical_details,
            impact_analysis=impact_analysis,
            remediation_steps=remediation_steps,
            remediation_priority=self._determine_remediation_priority(severity),
            effort_estimate=effort_estimate,
            confidence_score=confidence_score,
            framework_mappings=framework_mappings
        )
    
    def _generate_technical_findings(self, control: ControlData, assessment_result: AssessmentResult,
                                   evidence_quality_score: float, framework: str) -> List[EnhancedFinding]:
        """Generate technical findings based on assessment"""
        
        findings = []
        
        # Identify technical domain
        technical_domain = self._identify_technical_domain(control)
        
        # Generate findings for each technical detail
        for i, technical_detail in enumerate(assessment_result.technical_details[:3]):
            finding_id = f"{control.control_id}_TECHNICAL_{i+1:03d}"
            
            # Analyze technical detail
            severity = self._analyze_technical_severity(technical_detail)
            category = self._determine_technical_category(technical_detail, technical_domain)
            
            # Generate technical finding
            title = self._generate_technical_title(technical_detail, technical_domain)
            description = self._generate_technical_description(technical_detail, control)
            
            # Extract specific technical indicators
            technical_indicators = self._extract_technical_indicators(technical_detail, technical_domain)
            
            # Generate technical remediation
            technical_remediation = self._generate_technical_remediation(technical_detail, technical_domain)
            
            finding = EnhancedFinding(
                finding_id=finding_id,
                finding_type=FindingType.TECHNICAL_FINDING,
                severity=severity,
                category=category,
                title=title,
                description=description,
                evidence_references=assessment_result.evidence_references[:2],
                technical_details=technical_indicators,
                impact_analysis=self._generate_technical_impact_analysis(technical_detail),
                remediation_steps=technical_remediation,
                remediation_priority=self._determine_remediation_priority(severity),
                effort_estimate=self._estimate_technical_effort(technical_detail),
                confidence_score=assessment_result.confidence_score * 0.9,  # Slightly lower for technical
                framework_mappings=self._generate_framework_mappings(control, framework)
            )
            
            findings.append(finding)
        
        return findings
    
    def _generate_procedural_findings(self, control: ControlData, assessment_result: AssessmentResult,
                                    evidence_quality_score: float, framework: str) -> List[EnhancedFinding]:
        """Generate procedural findings"""
        
        findings = []
        
        # Look for procedural aspects in the assessment
        procedural_keywords = ['procedure', 'process', 'workflow', 'governance', 'management']
        
        for finding_text in assessment_result.key_findings:
            if any(keyword in finding_text.lower() for keyword in procedural_keywords):
                finding_id = f"{control.control_id}_PROCEDURAL_{len(findings)+1:03d}"
                
                severity = self._determine_procedural_severity(finding_text)
                
                finding = EnhancedFinding(
                    finding_id=finding_id,
                    finding_type=FindingType.PROCEDURAL_FINDING,
                    severity=severity,
                    category=FindingCategory.OPERATIONAL_PROCESS,
                    title=self._generate_procedural_title(finding_text),
                    description=self._generate_procedural_description(finding_text, control),
                    evidence_references=assessment_result.evidence_references[:2],
                    technical_details=[],
                    impact_analysis=self._generate_procedural_impact_analysis(finding_text),
                    remediation_steps=self._generate_procedural_remediation(finding_text),
                    remediation_priority=self._determine_remediation_priority(severity),
                    effort_estimate=self._estimate_procedural_effort(finding_text),
                    confidence_score=assessment_result.confidence_score * 0.85,
                    framework_mappings=self._generate_framework_mappings(control, framework)
                )
                
                findings.append(finding)
        
        return findings[:2]  # Limit to 2 procedural findings
    
    def _generate_gap_findings(self, control: ControlData, assessment_result: AssessmentResult,
                             evidence_quality_score: float, framework: str) -> List[EnhancedFinding]:
        """Generate gap findings based on evidence gaps"""
        
        findings = []
        
        for i, gap in enumerate(assessment_result.evidence_gaps[:2]):
            finding_id = f"{control.control_id}_GAP_{i+1:03d}"
            
            severity = self._determine_gap_severity(gap, assessment_result.compliance_status)
            
            finding = EnhancedFinding(
                finding_id=finding_id,
                finding_type=FindingType.GAP_FINDING,
                severity=severity,
                category=FindingCategory.SECURITY_CONTROL,
                title=self._generate_gap_title(gap),
                description=self._generate_gap_description(gap, control),
                evidence_references=[],
                technical_details=[],
                impact_analysis=self._generate_gap_impact_analysis(gap),
                remediation_steps=self._generate_gap_remediation(gap),
                remediation_priority=self._determine_remediation_priority(severity),
                effort_estimate=self._estimate_gap_effort(gap),
                confidence_score=assessment_result.confidence_score * 0.8,
                framework_mappings=self._generate_framework_mappings(control, framework)
            )
            
            findings.append(finding)
        
        return findings
    
    def _generate_risk_findings(self, control: ControlData, assessment_result: AssessmentResult,
                              evidence_quality_score: float, framework: str) -> List[EnhancedFinding]:
        """Generate risk-based findings"""
        
        findings = []
        
        # Generate findings based on identified risks
        for i, risk in enumerate(assessment_result.risks[:2]):
            finding_id = f"{control.control_id}_RISK_{i+1:03d}"
            
            severity = self._determine_risk_finding_severity(risk, assessment_result.risk_level)
            
            finding = EnhancedFinding(
                finding_id=finding_id,
                finding_type=FindingType.RISK_FINDING,
                severity=severity,
                category=FindingCategory.RISK_MANAGEMENT,
                title=self._generate_risk_title(risk),
                description=self._generate_risk_description(risk, control),
                evidence_references=assessment_result.evidence_references[:2],
                technical_details=[],
                impact_analysis=self._generate_risk_impact_analysis(risk),
                remediation_steps=self._generate_risk_remediation(risk),
                remediation_priority=self._determine_remediation_priority(severity),
                effort_estimate=self._estimate_risk_effort(risk),
                confidence_score=assessment_result.confidence_score * 0.9,
                framework_mappings=self._generate_framework_mappings(control, framework)
            )
            
            findings.append(finding)
        
        return findings
    
    def _enhance_findings_with_correlations(self, findings: List[EnhancedFinding]) -> List[EnhancedFinding]:
        """Enhance findings with correlation analysis"""
        
        # Find correlations between findings
        correlations = []
        
        for i, finding1 in enumerate(findings):
            for j, finding2 in enumerate(findings[i+1:], i+1):
                correlation = self._analyze_finding_correlation(finding1, finding2)
                if correlation.correlation_strength > 0.5:
                    correlations.append(correlation)
        
        # Add correlation information to findings
        for finding in findings:
            related_findings = []
            for correlation in correlations:
                if finding.finding_id in correlation.finding_ids:
                    other_finding_id = [fid for fid in correlation.finding_ids if fid != finding.finding_id][0]
                    related_findings.append(f"Related to {other_finding_id}: {correlation.description}")
            
            if related_findings:
                finding.technical_details.extend(related_findings[:2])
        
        return findings
    
    def _determine_compliance_severity(self, compliance_status: str, risk_level: str) -> FindingSeverity:
        """Determine severity based on compliance status and risk level"""
        
        if compliance_status == "Non-Compliant":
            if risk_level == "Critical":
                return FindingSeverity.CRITICAL
            elif risk_level == "High":
                return FindingSeverity.HIGH
            else:
                return FindingSeverity.MEDIUM
        elif compliance_status == "Partially Compliant":
            if risk_level in ["Critical", "High"]:
                return FindingSeverity.HIGH
            else:
                return FindingSeverity.MEDIUM
        else:  # Compliant
            if risk_level == "Critical":
                return FindingSeverity.MEDIUM
            else:
                return FindingSeverity.LOW
    
    def _generate_compliance_title(self, control: ControlData, assessment_result: AssessmentResult) -> str:
        """Generate compliance finding title"""
        
        status_templates = {
            "Compliant": f"Control {control.control_id} demonstrates effective implementation",
            "Partially Compliant": f"Control {control.control_id} shows partial compliance with gaps",
            "Non-Compliant": f"Control {control.control_id} fails to meet compliance requirements"
        }
        
        return status_templates.get(assessment_result.compliance_status, 
                                  f"Control {control.control_id} compliance assessment")
    
    def _generate_compliance_description(self, control: ControlData, assessment_result: AssessmentResult,
                                       evidence_quality_score: float) -> str:
        """Generate detailed compliance description"""
        
        description = f"Assessment of control {control.control_id} ({control.control_name}) "
        description += f"resulted in {assessment_result.compliance_status} status with {assessment_result.risk_level} risk level. "
        
        if assessment_result.key_findings:
            description += f"Key findings include: {'; '.join(assessment_result.key_findings[:2])}. "
        
        if evidence_quality_score < 0.6:
            description += "Evidence quality is below acceptable standards, affecting assessment confidence. "
        
        description += f"Overall confidence in this assessment is {assessment_result.confidence_score:.1%}."
        
        return description
    
    def _extract_technical_details(self, assessment_result: AssessmentResult, control: ControlData) -> List[str]:
        """Extract technical details from assessment result"""
        
        technical_details = []
        
        # Use existing technical details
        technical_details.extend(assessment_result.technical_details[:3])
        
        # Extract technical aspects from findings
        for finding in assessment_result.key_findings:
            technical_terms = self._extract_technical_terms(finding)
            if technical_terms:
                technical_details.append(f"Technical aspects: {', '.join(technical_terms)}")
        
        return technical_details[:5]  # Limit to 5 technical details
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text"""
        
        technical_terms = []
        text_lower = text.lower()
        
        # Check all technical indicators
        for domain, terms in self.technical_indicators.items():
            for term in terms:
                if term in text_lower:
                    technical_terms.append(term)
        
        return list(set(technical_terms))  # Remove duplicates
    
    def _generate_impact_analysis(self, control: ControlData, assessment_result: AssessmentResult) -> str:
        """Generate impact analysis"""
        
        impact_analysis = f"Impact analysis for control {control.control_id}: "
        
        # Business impact
        if assessment_result.compliance_status == "Non-Compliant":
            impact_analysis += "Non-compliance may result in regulatory violations, audit findings, and potential penalties. "
        
        # Security impact
        if assessment_result.risk_level in ["Critical", "High"]:
            impact_analysis += "High security risk may lead to data breaches, unauthorized access, or system compromise. "
        
        # Operational impact
        if assessment_result.evidence_gaps:
            impact_analysis += "Evidence gaps indicate potential operational inefficiencies and control weaknesses. "
        
        return impact_analysis
    
    def _generate_remediation_steps(self, control: ControlData, assessment_result: AssessmentResult) -> List[str]:
        """Generate remediation steps"""
        
        remediation_steps = []
        
        # Use existing remediation from assessment
        remediation_steps.extend(assessment_result.remediation[:3])
        
        # Add specific remediation based on compliance status
        if assessment_result.compliance_status == "Non-Compliant":
            remediation_steps.append(f"Implement comprehensive {control.control_name.lower()} controls")
        elif assessment_result.compliance_status == "Partially Compliant":
            remediation_steps.append(f"Address identified gaps in {control.control_name.lower()} implementation")
        
        # Add evidence-based remediation
        if assessment_result.evidence_gaps:
            remediation_steps.append("Provide additional evidence to support control implementation")
        
        return remediation_steps[:5]  # Limit to 5 steps
    
    def _estimate_remediation_effort(self, compliance_status: str, remediation_count: int) -> str:
        """Estimate remediation effort"""
        
        if compliance_status == "Non-Compliant":
            return "High" if remediation_count > 3 else "Medium"
        elif compliance_status == "Partially Compliant":
            return "Medium" if remediation_count > 2 else "Low"
        else:
            return "Low"
    
    def _calculate_finding_confidence(self, assessment_result: AssessmentResult, 
                                    evidence_quality_score: float) -> float:
        """Calculate confidence score for finding"""
        
        # Base confidence from assessment
        base_confidence = assessment_result.confidence_score
        
        # Adjust for evidence quality
        evidence_adjustment = evidence_quality_score * 0.2
        
        # Adjust for completeness
        completeness_factor = 1.0
        if not assessment_result.key_findings:
            completeness_factor *= 0.8
        if not assessment_result.technical_details:
            completeness_factor *= 0.9
        
        return min(1.0, (base_confidence + evidence_adjustment) * completeness_factor)
    
    def _generate_framework_mappings(self, control: ControlData, framework: str) -> Dict[str, str]:
        """Generate framework mappings"""
        
        mappings = {
            'primary_framework': framework,
            'control_id': control.control_id,
            'control_name': control.control_name
        }
        
        # Add cross-framework mappings if applicable
        if framework == "NIST":
            mappings['iso27001_mapping'] = self._map_nist_to_iso27001(control.control_id)
        elif framework == "ISO27001":
            mappings['nist_mapping'] = self._map_iso27001_to_nist(control.control_id)
        
        return mappings
    
    def _determine_remediation_priority(self, severity: FindingSeverity) -> str:
        """Determine remediation priority based on severity"""
        
        priority_mapping = {
            FindingSeverity.CRITICAL: "Immediate",
            FindingSeverity.HIGH: "High",
            FindingSeverity.MEDIUM: "Medium",
            FindingSeverity.LOW: "Low",
            FindingSeverity.INFORMATIONAL: "Low"
        }
        
        return priority_mapping.get(severity, "Medium")
    
    def _identify_technical_domain(self, control: ControlData) -> str:
        """Identify technical domain for control"""
        
        control_text = f"{control.control_name} {control.requirement}".lower()
        
        # Check each technical domain
        for domain, indicators in self.technical_indicators.items():
            if any(indicator in control_text for indicator in indicators):
                return domain
        
        return "general"
    
    def _analyze_technical_severity(self, technical_detail: str) -> FindingSeverity:
        """Analyze severity of technical detail"""
        
        detail_lower = technical_detail.lower()
        
        # Check severity keywords
        for severity_level, rules in self.severity_rules.items():
            if any(keyword in detail_lower for keyword in rules['keywords']):
                return FindingSeverity(severity_level)
        
        return FindingSeverity.MEDIUM
    
    def _determine_technical_category(self, technical_detail: str, domain: str) -> FindingCategory:
        """Determine category for technical finding"""
        
        detail_lower = technical_detail.lower()
        
        if 'configuration' in detail_lower:
            return FindingCategory.TECHNICAL_IMPLEMENTATION
        elif 'policy' in detail_lower:
            return FindingCategory.POLICY_GOVERNANCE
        elif 'process' in detail_lower:
            return FindingCategory.OPERATIONAL_PROCESS
        else:
            return FindingCategory.SECURITY_CONTROL
    
    def _generate_technical_title(self, technical_detail: str, domain: str) -> str:
        """Generate title for technical finding"""
        
        # Extract key technical aspect
        key_aspect = technical_detail.split('.')[0] if '.' in technical_detail else technical_detail
        
        return f"Technical Analysis: {key_aspect[:50]}..."
    
    def _generate_technical_description(self, technical_detail: str, control: ControlData) -> str:
        """Generate description for technical finding"""
        
        return f"Technical analysis of control {control.control_id} reveals: {technical_detail}"
    
    def _extract_technical_indicators(self, technical_detail: str, domain: str) -> List[str]:
        """Extract technical indicators from detail"""
        
        indicators = []
        detail_lower = technical_detail.lower()
        
        # Get domain-specific indicators
        domain_indicators = self.technical_indicators.get(domain, [])
        
        for indicator in domain_indicators:
            if indicator in detail_lower:
                indicators.append(indicator)
        
        return indicators
    
    def _generate_technical_remediation(self, technical_detail: str, domain: str) -> List[str]:
        """Generate technical remediation steps"""
        
        remediation = []
        detail_lower = technical_detail.lower()
        
        # Domain-specific remediation
        if domain == "access_control":
            remediation.append("Review and update access control configurations")
            remediation.append("Implement multi-factor authentication where missing")
        elif domain == "encryption":
            remediation.append("Upgrade encryption algorithms to current standards")
            remediation.append("Implement proper key management practices")
        elif domain == "monitoring":
            remediation.append("Enhance monitoring and alerting capabilities")
            remediation.append("Implement comprehensive log analysis")
        
        return remediation[:3]
    
    def _generate_technical_impact_analysis(self, technical_detail: str) -> str:
        """Generate impact analysis for technical finding"""
        
        return f"Technical impact: {technical_detail[:100]}... may affect system security and operational efficiency."
    
    def _estimate_technical_effort(self, technical_detail: str) -> str:
        """Estimate effort for technical remediation"""
        
        detail_lower = technical_detail.lower()
        
        if any(term in detail_lower for term in ['configuration', 'setting', 'parameter']):
            return "Low"
        elif any(term in detail_lower for term in ['implementation', 'deployment', 'architecture']):
            return "High"
        else:
            return "Medium"
    
    def _analyze_finding_correlation(self, finding1: EnhancedFinding, 
                                   finding2: EnhancedFinding) -> FindingCorrelation:
        """Analyze correlation between two findings"""
        
        correlation_strength = 0.0
        correlation_type = "none"
        description = ""
        
        # Check for category correlation
        if finding1.category == finding2.category:
            correlation_strength += 0.3
            correlation_type = "category"
        
        # Check for severity correlation
        if finding1.severity == finding2.severity:
            correlation_strength += 0.2
        
        # Check for technical correlation
        common_technical_terms = set(finding1.technical_details) & set(finding2.technical_details)
        if common_technical_terms:
            correlation_strength += 0.4
            correlation_type = "technical"
        
        # Generate description
        if correlation_strength > 0.5:
            description = f"Findings are correlated through {correlation_type} relationship"
        
        return FindingCorrelation(
            finding_ids=[finding1.finding_id, finding2.finding_id],
            correlation_type=correlation_type,
            correlation_strength=correlation_strength,
            description=description
        )
    
    def _map_nist_to_iso27001(self, nist_control_id: str) -> str:
        """Map NIST control to ISO27001 (simplified mapping)"""
        
        # This is a simplified mapping - in practice, would use comprehensive mapping tables
        mapping = {
            'PR.AC-1': 'A.9.1.1',
            'PR.AC-2': 'A.9.1.2',
            'PR.DS-1': 'A.10.1.1',
            'PR.DS-2': 'A.10.1.2'
        }
        
        return mapping.get(nist_control_id, "Not mapped")
    
    def _map_iso27001_to_nist(self, iso_control_id: str) -> str:
        """Map ISO27001 control to NIST (simplified mapping)"""
        
        # This is a simplified mapping - in practice, would use comprehensive mapping tables
        mapping = {
            'A.9.1.1': 'PR.AC-1',
            'A.9.1.2': 'PR.AC-2',
            'A.10.1.1': 'PR.DS-1',
            'A.10.1.2': 'PR.DS-2'
        }
        
        return mapping.get(iso_control_id, "Not mapped")
    
    # Additional helper methods for other finding types...
    def _determine_procedural_severity(self, finding_text: str) -> FindingSeverity:
        """Determine severity for procedural finding"""
        return FindingSeverity.MEDIUM  # Default for procedural findings
    
    def _generate_procedural_title(self, finding_text: str) -> str:
        """Generate procedural finding title"""
        return f"Procedural Finding: {finding_text[:50]}..."
    
    def _generate_procedural_description(self, finding_text: str, control: ControlData) -> str:
        """Generate procedural finding description"""
        return f"Procedural analysis of control {control.control_id}: {finding_text}"
    
    def _generate_procedural_impact_analysis(self, finding_text: str) -> str:
        """Generate procedural impact analysis"""
        return f"Procedural impact: {finding_text[:100]}... may affect operational efficiency."
    
    def _generate_procedural_remediation(self, finding_text: str) -> List[str]:
        """Generate procedural remediation steps"""
        return ["Review and update procedural documentation", "Implement process improvements"]
    
    def _estimate_procedural_effort(self, finding_text: str) -> str:
        """Estimate procedural remediation effort"""
        return "Medium"
    
    def _determine_gap_severity(self, gap: str, compliance_status: str) -> FindingSeverity:
        """Determine severity for gap finding"""
        if compliance_status == "Non-Compliant":
            return FindingSeverity.HIGH
        else:
            return FindingSeverity.MEDIUM
    
    def _generate_gap_title(self, gap: str) -> str:
        """Generate gap finding title"""
        return f"Evidence Gap: {gap[:50]}..."
    
    def _generate_gap_description(self, gap: str, control: ControlData) -> str:
        """Generate gap finding description"""
        return f"Evidence gap identified for control {control.control_id}: {gap}"
    
    def _generate_gap_impact_analysis(self, gap: str) -> str:
        """Generate gap impact analysis"""
        return f"Gap impact: {gap} may affect assessment confidence and control effectiveness."
    
    def _generate_gap_remediation(self, gap: str) -> List[str]:
        """Generate gap remediation steps"""
        return [f"Address identified gap: {gap}", "Provide additional supporting evidence"]
    
    def _estimate_gap_effort(self, gap: str) -> str:
        """Estimate gap remediation effort"""
        return "Low"
    
    def _determine_risk_finding_severity(self, risk: str, risk_level: str) -> FindingSeverity:
        """Determine severity for risk finding"""
        if risk_level == "Critical":
            return FindingSeverity.CRITICAL
        elif risk_level == "High":
            return FindingSeverity.HIGH
        elif risk_level == "Medium":
            return FindingSeverity.MEDIUM
        else:
            return FindingSeverity.LOW
    
    def _generate_risk_title(self, risk: str) -> str:
        """Generate risk finding title"""
        return f"Risk Finding: {risk[:50]}..."
    
    def _generate_risk_description(self, risk: str, control: ControlData) -> str:
        """Generate risk finding description"""
        return f"Risk analysis for control {control.control_id}: {risk}"
    
    def _generate_risk_impact_analysis(self, risk: str) -> str:
        """Generate risk impact analysis"""
        return f"Risk impact: {risk} may result in security incidents or compliance violations."
    
    def _generate_risk_remediation(self, risk: str) -> List[str]:
        """Generate risk remediation steps"""
        return [f"Mitigate identified risk: {risk[:50]}...", "Implement additional security controls"]
    
    def _estimate_risk_effort(self, risk: str) -> str:
        """Estimate risk remediation effort"""
        return "Medium"
    
    def export_findings_report(self, findings: List[EnhancedFinding], 
                             output_path: str = "findings_report.json"):
        """Export detailed findings report"""
        try:
            findings_data = [finding.to_dict() for finding in findings]
            with open(output_path, 'w') as f:
                json.dump(findings_data, f, indent=2)
            logger.info(f"Findings report exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export findings report: {e}")