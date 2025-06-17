"""
Cybersecurity Knowledge Base for Enhanced AI Analysis
Contains domain-specific knowledge and validation rules for cybersecurity frameworks
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum

class ControlType(Enum):
    """Types of cybersecurity controls"""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"
    ADMINISTRATIVE = "administrative"
    TECHNICAL = "technical"
    PHYSICAL = "physical"

@dataclass
class ControlKnowledge:
    """Knowledge about a specific control type"""
    control_type: ControlType
    required_evidence_types: List[str]
    critical_keywords: List[str]
    common_implementations: List[str]
    validation_criteria: List[str]

class CybersecurityKnowledgeBase:
    """Knowledge base for cybersecurity control analysis"""
    
    # Framework-specific control mappings
    FRAMEWORK_CONTROLS = {
        "NIST": {
            "ID": ["asset_management", "governance", "risk_assessment"],
            "PR": ["access_control", "data_security", "protective_technology"],
            "DE": ["anomaly_detection", "continuous_monitoring"],
            "RS": ["response_planning", "communications", "analysis"],
            "RC": ["recovery_planning", "improvements"]
        },
        "ISO27001": {
            "A.5": ["information_security_policies"],
            "A.6": ["organization_of_information_security"],
            "A.7": ["human_resource_security"],
            "A.8": ["asset_management"],
            "A.9": ["access_control"],
            "A.10": ["cryptography"],
            "A.11": ["physical_and_environmental_security"],
            "A.12": ["operations_security"],
            "A.13": ["communications_security"],
            "A.14": ["system_acquisition_development_maintenance"],
            "A.15": ["supplier_relationships"],
            "A.16": ["information_security_incident_management"],
            "A.17": ["business_continuity"],
            "A.18": ["compliance"]
        }
    }
    
    # Evidence type requirements by control category
    EVIDENCE_REQUIREMENTS = {
        "access_control": [
            "access_control_matrix", "user_provisioning_procedures", "authentication_logs",
            "privileged_access_documentation", "role_definitions", "access_review_reports"
        ],
        "encryption": [
            "encryption_policy", "key_management_procedures", "certificate_inventory",
            "encryption_configuration", "cryptographic_standards"
        ],
        "monitoring": [
            "monitoring_procedures", "log_configuration", "alert_definitions",
            "incident_response_logs", "security_monitoring_reports"
        ],
        "vulnerability_management": [
            "vulnerability_scan_reports", "patch_management_procedures", 
            "vulnerability_assessment_policy", "remediation_tracking"
        ],
        "incident_response": [
            "incident_response_plan", "incident_logs", "communication_procedures",
            "forensic_procedures", "lessons_learned_reports"
        ]
    }
    
    # Critical security indicators by domain
    SECURITY_INDICATORS = {
        "access_control": {
            "strong": ["multi-factor authentication", "role-based access", "least privilege", 
                      "regular access reviews", "automated provisioning"],
            "weak": ["shared accounts", "default passwords", "no access reviews", 
                    "excessive privileges", "manual provisioning"]
        },
        "encryption": {
            "strong": ["AES-256", "TLS 1.3", "certificate management", "key rotation", 
                      "hardware security modules"],
            "weak": ["weak ciphers", "self-signed certificates", "hardcoded keys", 
                    "no key management", "deprecated protocols"]
        },
        "monitoring": {
            "strong": ["real-time monitoring", "centralized logging", "automated alerting", 
                      "log integrity", "correlation rules"],
            "weak": ["no monitoring", "local logs only", "manual review", 
                    "no alerting", "log gaps"]
        }
    }
    
    # Risk assessment criteria
    RISK_CRITERIA = {
        "critical": {
            "conditions": ["no implementation", "fundamental security failure", 
                          "regulatory violation", "data breach risk"],
            "indicators": ["no evidence", "failed testing", "known vulnerabilities", 
                          "compliance violation"]
        },
        "high": {
            "conditions": ["partial implementation", "significant gaps", 
                          "weak controls", "limited monitoring"],
            "indicators": ["incomplete evidence", "configuration issues", 
                          "manual processes", "delayed patching"]
        },
        "medium": {
            "conditions": ["mostly implemented", "minor gaps", 
                          "adequate controls", "some automation"],
            "indicators": ["good evidence", "minor issues", 
                          "documented procedures", "regular reviews"]
        },
        "low": {
            "conditions": ["fully implemented", "comprehensive controls", 
                          "automated processes", "continuous monitoring"],
            "indicators": ["strong evidence", "no issues found", 
                          "automated controls", "proactive management"]
        }
    }
    
    @classmethod
    def get_control_knowledge(cls, control_id: str, framework: str) -> Optional[ControlKnowledge]:
        """Get knowledge about a specific control"""
        # This would be expanded with detailed control mappings
        # For now, return basic knowledge based on control ID patterns
        
        if framework == "NIST":
            if control_id.startswith("PR.AC"):
                return ControlKnowledge(
                    control_type=ControlType.PREVENTIVE,
                    required_evidence_types=cls.EVIDENCE_REQUIREMENTS["access_control"],
                    critical_keywords=["access", "authentication", "authorization", "identity"],
                    common_implementations=["RBAC", "MFA", "SSO", "PAM"],
                    validation_criteria=["access matrix", "user reviews", "authentication logs"]
                )
            elif control_id.startswith("PR.DS"):
                return ControlKnowledge(
                    control_type=ControlType.PREVENTIVE,
                    required_evidence_types=cls.EVIDENCE_REQUIREMENTS["encryption"],
                    critical_keywords=["encryption", "cryptography", "data protection"],
                    common_implementations=["AES", "TLS", "PKI", "DLP"],
                    validation_criteria=["encryption policy", "key management", "data classification"]
                )
        
        return None
    
    @classmethod
    def assess_evidence_quality(cls, evidence_text: str, control_category: str) -> Dict[str, float]:
        """Assess the quality of evidence for a control category"""
        scores = {
            "completeness": 0.0,
            "specificity": 0.0,
            "technical_depth": 0.0,
            "currency": 0.0
        }
        
        evidence_lower = evidence_text.lower()
        
        # Check for required evidence types
        required_evidence = cls.EVIDENCE_REQUIREMENTS.get(control_category, [])
        if required_evidence:
            found_evidence = sum(1 for req in required_evidence if any(word in evidence_lower for word in req.split('_')))
            scores["completeness"] = min(1.0, found_evidence / len(required_evidence))
        
        # Check for technical specificity
        technical_terms = ["configuration", "implementation", "version", "setting", "parameter"]
        technical_count = sum(1 for term in technical_terms if term in evidence_lower)
        scores["technical_depth"] = min(1.0, technical_count / 3.0)
        
        # Check for specific indicators
        if control_category in cls.SECURITY_INDICATORS:
            strong_indicators = cls.SECURITY_INDICATORS[control_category]["strong"]
            weak_indicators = cls.SECURITY_INDICATORS[control_category]["weak"]
            
            strong_count = sum(1 for indicator in strong_indicators if indicator in evidence_lower)
            weak_count = sum(1 for indicator in weak_indicators if indicator in evidence_lower)
            
            # Specificity based on strong vs weak indicators
            if strong_count + weak_count > 0:
                scores["specificity"] = strong_count / (strong_count + weak_count)
            else:
                scores["specificity"] = 0.5  # Neutral if no indicators found
        
        # Check for currency indicators (dates, recent versions, etc.)
        currency_indicators = ["2023", "2024", "current", "latest", "updated", "recent"]
        currency_count = sum(1 for indicator in currency_indicators if indicator in evidence_lower)
        scores["currency"] = min(1.0, currency_count / 2.0)
        
        return scores
    
    @classmethod
    def determine_risk_level(cls, evidence_quality: Dict[str, float], 
                           compliance_indicators: List[str]) -> str:
        """Determine risk level based on evidence quality and compliance indicators"""
        
        # Calculate overall evidence score
        avg_quality = sum(evidence_quality.values()) / len(evidence_quality)
        
        # Check for critical risk indicators
        critical_indicators = ["no implementation", "failed", "vulnerable", "breach", "violation"]
        high_indicators = ["partial", "incomplete", "manual", "delayed", "gaps"]
        
        compliance_text = ' '.join(compliance_indicators).lower()
        
        if any(indicator in compliance_text for indicator in critical_indicators) or avg_quality < 0.3:
            return "Critical"
        elif any(indicator in compliance_text for indicator in high_indicators) or avg_quality < 0.5:
            return "High"
        elif avg_quality < 0.7:
            return "Medium"
        else:
            return "Low"
    
    @classmethod
    def get_remediation_templates(cls, control_category: str, risk_level: str) -> List[str]:
        """Get remediation templates based on control category and risk level"""
        
        base_remediations = {
            "access_control": [
                "Implement multi-factor authentication for all user accounts",
                "Conduct quarterly access reviews and remove unnecessary privileges",
                "Establish role-based access control with least privilege principles",
                "Document and approve all privileged access procedures"
            ],
            "encryption": [
                "Implement strong encryption (AES-256) for data at rest and in transit",
                "Establish comprehensive key management procedures",
                "Deploy certificate management system with automated renewal",
                "Conduct regular cryptographic algorithm reviews"
            ],
            "monitoring": [
                "Implement centralized security information and event management (SIEM)",
                "Configure real-time alerting for security events",
                "Establish log retention and integrity protection procedures",
                "Conduct regular monitoring effectiveness reviews"
            ]
        }
        
        remediations = base_remediations.get(control_category, [
            "Review and update security policies and procedures",
            "Implement appropriate technical controls",
            "Establish monitoring and review processes",
            "Provide security awareness training"
        ])
        
        # Prioritize based on risk level
        if risk_level in ["Critical", "High"]:
            return [f"URGENT: {rem}" for rem in remediations[:2]] + remediations[2:]
        else:
            return remediations
    
    @classmethod
    def validate_compliance_determination(cls, compliance_status: str, 
                                        evidence_quality: Dict[str, float],
                                        risk_level: str) -> List[str]:
        """Validate compliance determination against evidence and risk"""
        issues = []
        
        avg_quality = sum(evidence_quality.values()) / len(evidence_quality)
        
        # Check for inconsistencies
        if compliance_status == "Compliant" and risk_level in ["Critical", "High"]:
            issues.append("Compliance status inconsistent with high/critical risk level")
        
        if compliance_status == "Compliant" and avg_quality < 0.6:
            issues.append("Compliance determination not supported by evidence quality")
        
        if compliance_status == "Non-Compliant" and avg_quality > 0.8:
            issues.append("Non-compliance determination may be too conservative given strong evidence")
        
        if evidence_quality["completeness"] < 0.4:
            issues.append("Insufficient evidence to make reliable compliance determination")
        
        return issues
