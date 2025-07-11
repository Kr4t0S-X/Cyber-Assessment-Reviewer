"""
Advanced Risk Assessment Engine for Cyber Assessment Reviewer
Implements dynamic risk scoring, interdependency analysis, and quantitative risk metrics
"""

import logging
import json
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

from models import AssessmentResult, ControlData, RiskMetrics

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Risk categories for classification"""
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    TECHNICAL = "technical"
    REPUTATIONAL = "reputational"

class RiskImpactDomain(Enum):
    """Domains of risk impact"""
    CONFIDENTIALITY = "confidentiality"
    INTEGRITY = "integrity"
    AVAILABILITY = "availability"
    PRIVACY = "privacy"
    SAFETY = "safety"
    FINANCIAL = "financial"
    REPUTATION = "reputation"
    OPERATIONS = "operations"

class RiskTreatment(Enum):
    """Risk treatment strategies"""
    MITIGATE = "mitigate"
    TRANSFER = "transfer"
    AVOID = "avoid"
    ACCEPT = "accept"

@dataclass
class RiskFactor:
    """Individual risk factor"""
    factor_id: str
    description: str
    likelihood: float  # 0.0 to 1.0
    impact: float     # 0.0 to 1.0
    category: RiskCategory
    impact_domains: List[RiskImpactDomain]
    confidence: float
    evidence_quality: float
    
@dataclass
class RiskScenario:
    """Risk scenario with multiple factors"""
    scenario_id: str
    description: str
    threat_vector: str
    vulnerability: str
    risk_factors: List[RiskFactor]
    inherent_risk: float
    residual_risk: float
    risk_treatment: RiskTreatment
    controls: List[str]
    
@dataclass
class RiskInterdependency:
    """Risk interdependency relationship"""
    primary_risk: str
    dependent_risk: str
    dependency_type: str
    correlation_strength: float
    amplification_factor: float
    description: str

@dataclass
class QuantitativeRiskMetrics:
    """Quantitative risk assessment metrics"""
    control_id: str
    value_at_risk: float
    expected_loss: float
    risk_exposure: float
    risk_concentration: float
    risk_velocity: float
    time_to_impact: int  # days
    recovery_time: int   # days
    business_impact_score: float
    technical_impact_score: float
    compliance_impact_score: float
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'control_id': self.control_id,
            'value_at_risk': self.value_at_risk,
            'expected_loss': self.expected_loss,
            'risk_exposure': self.risk_exposure,
            'risk_concentration': self.risk_concentration,
            'risk_velocity': self.risk_velocity,
            'time_to_impact': self.time_to_impact,
            'recovery_time': self.recovery_time,
            'business_impact_score': self.business_impact_score,
            'technical_impact_score': self.technical_impact_score,
            'compliance_impact_score': self.compliance_impact_score
        }

@dataclass
class DynamicRiskProfile:
    """Dynamic risk profile with contextual factors"""
    control_id: str
    baseline_risk: float
    contextual_multipliers: Dict[str, float]
    temporal_factors: Dict[str, float]
    environmental_factors: Dict[str, float]
    organizational_factors: Dict[str, float]
    threat_landscape_factors: Dict[str, float]
    adjusted_risk: float
    risk_trajectory: str  # increasing, stable, decreasing
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'control_id': self.control_id,
            'baseline_risk': self.baseline_risk,
            'contextual_multipliers': self.contextual_multipliers,
            'temporal_factors': self.temporal_factors,
            'environmental_factors': self.environmental_factors,
            'organizational_factors': self.organizational_factors,
            'threat_landscape_factors': self.threat_landscape_factors,
            'adjusted_risk': self.adjusted_risk,
            'risk_trajectory': self.risk_trajectory
        }

class AdvancedRiskEngine:
    """Advanced risk assessment engine with dynamic scoring"""
    
    def __init__(self, config=None):
        self.config = config
        self.risk_models = self._initialize_risk_models()
        self.threat_intelligence = self._initialize_threat_intelligence()
        self.impact_matrices = self._initialize_impact_matrices()
        self.risk_tolerance_thresholds = self._initialize_risk_tolerance()
        self.business_context = self._initialize_business_context()
        
    def _initialize_risk_models(self) -> Dict[str, Dict[str, any]]:
        """Initialize risk assessment models"""
        return {
            'nist_framework': {
                'likelihood_factors': {
                    'threat_source': {'advanced_persistent_threat': 0.8, 'insider_threat': 0.6, 'script_kiddie': 0.3},
                    'vulnerability_severity': {'critical': 0.9, 'high': 0.7, 'medium': 0.5, 'low': 0.3},
                    'control_effectiveness': {'effective': 0.2, 'partially_effective': 0.6, 'ineffective': 0.9}
                },
                'impact_factors': {
                    'data_sensitivity': {'highly_sensitive': 0.9, 'sensitive': 0.7, 'internal': 0.5, 'public': 0.2},
                    'system_criticality': {'critical': 0.9, 'high': 0.7, 'medium': 0.5, 'low': 0.3},
                    'regulatory_scope': {'regulated': 0.8, 'partially_regulated': 0.5, 'unregulated': 0.2}
                }
            },
            'iso27001_framework': {
                'likelihood_factors': {
                    'threat_frequency': {'very_high': 0.9, 'high': 0.7, 'medium': 0.5, 'low': 0.3, 'very_low': 0.1},
                    'vulnerability_exploitability': {'easy': 0.8, 'moderate': 0.6, 'difficult': 0.4, 'very_difficult': 0.2}
                },
                'impact_factors': {
                    'business_impact': {'catastrophic': 0.9, 'major': 0.7, 'moderate': 0.5, 'minor': 0.3, 'insignificant': 0.1},
                    'information_classification': {'top_secret': 0.9, 'secret': 0.7, 'confidential': 0.5, 'internal': 0.3}
                }
            }
        }
    
    def _initialize_threat_intelligence(self) -> Dict[str, Dict[str, any]]:
        """Initialize threat intelligence data"""
        return {
            'current_threats': {
                'ransomware': {'probability': 0.7, 'impact': 0.8, 'trend': 'increasing'},
                'phishing': {'probability': 0.8, 'impact': 0.6, 'trend': 'stable'},
                'insider_threat': {'probability': 0.4, 'impact': 0.7, 'trend': 'increasing'},
                'supply_chain': {'probability': 0.5, 'impact': 0.8, 'trend': 'increasing'},
                'zero_day': {'probability': 0.3, 'impact': 0.9, 'trend': 'stable'}
            },
            'industry_threats': {
                'financial': {'account_takeover': 0.8, 'fraud': 0.7, 'regulatory_violation': 0.6},
                'healthcare': {'data_breach': 0.8, 'privacy_violation': 0.7, 'system_downtime': 0.6},
                'technology': {'ip_theft': 0.7, 'system_compromise': 0.8, 'service_disruption': 0.6},
                'government': {'nation_state': 0.6, 'data_breach': 0.7, 'service_disruption': 0.5}
            },
            'vulnerability_trends': {
                'unpatched_systems': {'criticality': 0.8, 'prevalence': 0.7, 'exploitation_rate': 0.6},
                'misconfigurations': {'criticality': 0.7, 'prevalence': 0.8, 'exploitation_rate': 0.5},
                'weak_credentials': {'criticality': 0.6, 'prevalence': 0.8, 'exploitation_rate': 0.7},
                'social_engineering': {'criticality': 0.7, 'prevalence': 0.6, 'exploitation_rate': 0.8}
            }
        }
    
    def _initialize_impact_matrices(self) -> Dict[str, Dict[str, float]]:
        """Initialize impact assessment matrices"""
        return {
            'financial_impact': {
                'direct_costs': {'incident_response': 0.3, 'system_recovery': 0.4, 'legal_fees': 0.2, 'fines': 0.5},
                'indirect_costs': {'reputation_damage': 0.6, 'business_interruption': 0.7, 'customer_churn': 0.5},
                'opportunity_costs': {'delayed_projects': 0.4, 'lost_partnerships': 0.5, 'market_share': 0.6}
            },
            'operational_impact': {
                'system_availability': {'complete_outage': 0.9, 'partial_outage': 0.6, 'degraded_performance': 0.4},
                'data_integrity': {'data_corruption': 0.8, 'data_loss': 0.9, 'data_manipulation': 0.7},
                'process_efficiency': {'workflow_disruption': 0.6, 'manual_processes': 0.5, 'delays': 0.4}
            },
            'compliance_impact': {
                'regulatory_violations': {'major_violation': 0.9, 'minor_violation': 0.6, 'notification_required': 0.4},
                'certification_impact': {'certificate_revocation': 0.8, 'audit_findings': 0.6, 'remediation_required': 0.5},
                'legal_exposure': {'lawsuits': 0.8, 'regulatory_action': 0.7, 'investigations': 0.6}
            }
        }
    
    def _initialize_risk_tolerance(self) -> Dict[str, Dict[str, float]]:
        """Initialize risk tolerance thresholds"""
        return {
            'enterprise': {
                'critical_systems': {'low': 0.2, 'medium': 0.4, 'high': 0.6, 'critical': 0.8},
                'sensitive_data': {'low': 0.1, 'medium': 0.3, 'high': 0.5, 'critical': 0.7},
                'regulatory_compliance': {'low': 0.1, 'medium': 0.2, 'high': 0.3, 'critical': 0.4}
            },
            'sme': {
                'critical_systems': {'low': 0.3, 'medium': 0.5, 'high': 0.7, 'critical': 0.9},
                'sensitive_data': {'low': 0.2, 'medium': 0.4, 'high': 0.6, 'critical': 0.8},
                'regulatory_compliance': {'low': 0.2, 'medium': 0.3, 'high': 0.4, 'critical': 0.5}
            }
        }
    
    def _initialize_business_context(self) -> Dict[str, any]:
        """Initialize business context factors"""
        return {
            'industry_risk_multipliers': {
                'financial_services': 1.3,
                'healthcare': 1.2,
                'government': 1.4,
                'technology': 1.1,
                'manufacturing': 1.0,
                'retail': 1.1,
                'education': 0.9
            },
            'organization_size_factors': {
                'enterprise': {'complexity': 1.2, 'resources': 0.8, 'visibility': 1.3},
                'large': {'complexity': 1.1, 'resources': 0.9, 'visibility': 1.2},
                'medium': {'complexity': 1.0, 'resources': 1.0, 'visibility': 1.0},
                'small': {'complexity': 0.9, 'resources': 1.2, 'visibility': 0.8}
            },
            'geographic_risk_factors': {
                'high_risk_regions': 1.3,
                'medium_risk_regions': 1.1,
                'low_risk_regions': 0.9
            }
        }
    
    def calculate_dynamic_risk(self, control: ControlData, assessment_result: AssessmentResult,
                             evidence_quality_score: float = 0.5,
                             business_context: Dict[str, any] = None) -> DynamicRiskProfile:
        """Calculate dynamic risk profile with contextual factors"""
        
        # Calculate baseline risk
        baseline_risk = self._calculate_baseline_risk(control, assessment_result, evidence_quality_score)
        
        # Calculate contextual multipliers
        contextual_multipliers = self._calculate_contextual_multipliers(control, assessment_result)
        
        # Calculate temporal factors
        temporal_factors = self._calculate_temporal_factors(control, assessment_result)
        
        # Calculate environmental factors
        environmental_factors = self._calculate_environmental_factors(control, business_context)
        
        # Calculate organizational factors
        organizational_factors = self._calculate_organizational_factors(control, business_context)
        
        # Calculate threat landscape factors
        threat_landscape_factors = self._calculate_threat_landscape_factors(control)
        
        # Calculate adjusted risk
        adjusted_risk = self._calculate_adjusted_risk(
            baseline_risk, contextual_multipliers, temporal_factors,
            environmental_factors, organizational_factors, threat_landscape_factors
        )
        
        # Determine risk trajectory
        risk_trajectory = self._determine_risk_trajectory(
            baseline_risk, adjusted_risk, temporal_factors, threat_landscape_factors
        )
        
        return DynamicRiskProfile(
            control_id=control.control_id,
            baseline_risk=baseline_risk,
            contextual_multipliers=contextual_multipliers,
            temporal_factors=temporal_factors,
            environmental_factors=environmental_factors,
            organizational_factors=organizational_factors,
            threat_landscape_factors=threat_landscape_factors,
            adjusted_risk=adjusted_risk,
            risk_trajectory=risk_trajectory
        )
    
    def calculate_quantitative_risk(self, control: ControlData, assessment_result: AssessmentResult,
                                  business_context: Dict[str, any] = None) -> QuantitativeRiskMetrics:
        """Calculate quantitative risk metrics"""
        
        # Calculate Value at Risk (VaR)
        value_at_risk = self._calculate_value_at_risk(control, assessment_result, business_context)
        
        # Calculate Expected Loss
        expected_loss = self._calculate_expected_loss(control, assessment_result, business_context)
        
        # Calculate Risk Exposure
        risk_exposure = self._calculate_risk_exposure(control, assessment_result)
        
        # Calculate Risk Concentration
        risk_concentration = self._calculate_risk_concentration(control, assessment_result)
        
        # Calculate Risk Velocity
        risk_velocity = self._calculate_risk_velocity(control, assessment_result)
        
        # Calculate Time to Impact
        time_to_impact = self._calculate_time_to_impact(control, assessment_result)
        
        # Calculate Recovery Time
        recovery_time = self._calculate_recovery_time(control, assessment_result)
        
        # Calculate impact scores
        business_impact_score = self._calculate_business_impact_score(control, assessment_result, business_context)
        technical_impact_score = self._calculate_technical_impact_score(control, assessment_result)
        compliance_impact_score = self._calculate_compliance_impact_score(control, assessment_result)
        
        return QuantitativeRiskMetrics(
            control_id=control.control_id,
            value_at_risk=value_at_risk,
            expected_loss=expected_loss,
            risk_exposure=risk_exposure,
            risk_concentration=risk_concentration,
            risk_velocity=risk_velocity,
            time_to_impact=time_to_impact,
            recovery_time=recovery_time,
            business_impact_score=business_impact_score,
            technical_impact_score=technical_impact_score,
            compliance_impact_score=compliance_impact_score
        )
    
    def analyze_risk_interdependencies(self, controls: List[ControlData], 
                                     assessment_results: List[AssessmentResult]) -> List[RiskInterdependency]:
        """Analyze risk interdependencies between controls"""
        
        interdependencies = []
        
        for i, control1 in enumerate(controls):
            for j, control2 in enumerate(controls[i+1:], i+1):
                if i < len(assessment_results) and j < len(assessment_results):
                    interdependency = self._analyze_control_interdependency(
                        control1, control2, assessment_results[i], assessment_results[j]
                    )
                    
                    if interdependency.correlation_strength > 0.3:
                        interdependencies.append(interdependency)
        
        return interdependencies
    
    def _calculate_baseline_risk(self, control: ControlData, assessment_result: AssessmentResult,
                               evidence_quality_score: float) -> float:
        """Calculate baseline risk score"""
        
        # Risk level mapping
        risk_level_scores = {
            'Critical': 0.9,
            'High': 0.7,
            'Medium': 0.5,
            'Low': 0.3,
            'Unknown': 0.5
        }
        
        # Compliance status impact
        compliance_impact = {
            'Compliant': 0.8,
            'Partially Compliant': 1.0,
            'Non-Compliant': 1.2,
            'Unknown': 1.1
        }
        
        # Evidence quality impact
        evidence_impact = 1.0 + (0.5 - evidence_quality_score)  # Lower quality = higher risk
        
        # Confidence impact
        confidence_impact = 1.0 + (0.8 - assessment_result.confidence_score)  # Lower confidence = higher risk
        
        # Calculate baseline risk
        base_risk = risk_level_scores.get(assessment_result.risk_level, 0.5)
        compliance_multiplier = compliance_impact.get(assessment_result.compliance_status, 1.0)
        
        baseline_risk = base_risk * compliance_multiplier * evidence_impact * confidence_impact
        
        return min(1.0, baseline_risk)
    
    def _calculate_contextual_multipliers(self, control: ControlData, 
                                        assessment_result: AssessmentResult) -> Dict[str, float]:
        """Calculate contextual risk multipliers"""
        
        multipliers = {}
        
        # Control criticality multiplier
        control_text = f"{control.control_name} {control.requirement}".lower()
        if any(term in control_text for term in ['critical', 'essential', 'mandatory']):
            multipliers['criticality'] = 1.2
        elif any(term in control_text for term in ['important', 'significant']):
            multipliers['criticality'] = 1.1
        else:
            multipliers['criticality'] = 1.0
        
        # Risk count multiplier
        risk_count = len(assessment_result.risks)
        if risk_count > 3:
            multipliers['risk_complexity'] = 1.2
        elif risk_count > 1:
            multipliers['risk_complexity'] = 1.1
        else:
            multipliers['risk_complexity'] = 1.0
        
        # Gap multiplier
        gap_count = len(assessment_result.evidence_gaps)
        if gap_count > 2:
            multipliers['evidence_gaps'] = 1.15
        elif gap_count > 0:
            multipliers['evidence_gaps'] = 1.05
        else:
            multipliers['evidence_gaps'] = 1.0
        
        return multipliers
    
    def _calculate_temporal_factors(self, control: ControlData, 
                                  assessment_result: AssessmentResult) -> Dict[str, float]:
        """Calculate temporal risk factors"""
        
        factors = {}
        
        # Time sensitivity
        if 'immediate' in assessment_result.compliance_rationale.lower():
            factors['urgency'] = 1.3
        elif 'urgent' in assessment_result.compliance_rationale.lower():
            factors['urgency'] = 1.2
        else:
            factors['urgency'] = 1.0
        
        # Trend factor (assuming risk is increasing over time)
        factors['trend'] = 1.05  # 5% increase due to evolving threats
        
        # Seasonality factor (some risks are seasonal)
        current_month = datetime.now().month
        if current_month in [11, 12, 1]:  # Holiday season
            factors['seasonality'] = 1.1
        else:
            factors['seasonality'] = 1.0
        
        return factors
    
    def _calculate_environmental_factors(self, control: ControlData, 
                                       business_context: Dict[str, any]) -> Dict[str, float]:
        """Calculate environmental risk factors"""
        
        factors = {}
        
        if business_context:
            # Industry factor
            industry = business_context.get('industry', 'general')
            industry_multiplier = self.business_context['industry_risk_multipliers'].get(industry, 1.0)
            factors['industry'] = industry_multiplier
            
            # Geographic factor
            geographic_risk = business_context.get('geographic_risk', 'medium')
            geographic_multiplier = self.business_context['geographic_risk_factors'].get(
                f"{geographic_risk}_risk_regions", 1.0
            )
            factors['geographic'] = geographic_multiplier
            
            # Regulatory environment
            regulatory_intensity = business_context.get('regulatory_intensity', 'medium')
            if regulatory_intensity == 'high':
                factors['regulatory'] = 1.2
            elif regulatory_intensity == 'low':
                factors['regulatory'] = 0.9
            else:
                factors['regulatory'] = 1.0
        else:
            factors = {'industry': 1.0, 'geographic': 1.0, 'regulatory': 1.0}
        
        return factors
    
    def _calculate_organizational_factors(self, control: ControlData, 
                                        business_context: Dict[str, any]) -> Dict[str, float]:
        """Calculate organizational risk factors"""
        
        factors = {}
        
        if business_context:
            # Organization size
            org_size = business_context.get('organization_size', 'medium')
            size_factors = self.business_context['organization_size_factors'].get(org_size, {})
            factors['complexity'] = size_factors.get('complexity', 1.0)
            factors['resources'] = size_factors.get('resources', 1.0)
            factors['visibility'] = size_factors.get('visibility', 1.0)
            
            # Security maturity
            security_maturity = business_context.get('security_maturity', 'medium')
            if security_maturity == 'low':
                factors['maturity'] = 1.3
            elif security_maturity == 'high':
                factors['maturity'] = 0.8
            else:
                factors['maturity'] = 1.0
        else:
            factors = {'complexity': 1.0, 'resources': 1.0, 'visibility': 1.0, 'maturity': 1.0}
        
        return factors
    
    def _calculate_threat_landscape_factors(self, control: ControlData) -> Dict[str, float]:
        """Calculate threat landscape factors"""
        
        factors = {}
        
        # Identify relevant threats based on control
        control_text = f"{control.control_name} {control.requirement}".lower()
        
        relevant_threats = []
        if 'access' in control_text or 'authentication' in control_text:
            relevant_threats.extend(['phishing', 'insider_threat'])
        if 'data' in control_text or 'information' in control_text:
            relevant_threats.extend(['ransomware', 'data_breach'])
        if 'system' in control_text or 'network' in control_text:
            relevant_threats.extend(['zero_day', 'system_compromise'])
        
        # Calculate threat intensity
        threat_intensity = 1.0
        for threat in relevant_threats:
            if threat in self.threat_intelligence['current_threats']:
                threat_data = self.threat_intelligence['current_threats'][threat]
                threat_score = threat_data['probability'] * threat_data['impact']
                threat_intensity = max(threat_intensity, 1.0 + threat_score * 0.3)
        
        factors['threat_intensity'] = min(1.5, threat_intensity)
        
        # Vulnerability trends
        vulnerability_factor = 1.0
        if 'configuration' in control_text:
            vulnerability_factor *= 1.1  # Misconfigurations are common
        if 'credential' in control_text:
            vulnerability_factor *= 1.15  # Credential issues are prevalent
        
        factors['vulnerability_trends'] = min(1.3, vulnerability_factor)
        
        return factors
    
    def _calculate_adjusted_risk(self, baseline_risk: float, contextual_multipliers: Dict[str, float],
                               temporal_factors: Dict[str, float], environmental_factors: Dict[str, float],
                               organizational_factors: Dict[str, float], 
                               threat_landscape_factors: Dict[str, float]) -> float:
        """Calculate final adjusted risk score"""
        
        # Start with baseline risk
        adjusted_risk = baseline_risk
        
        # Apply all multipliers
        all_factors = [
            contextual_multipliers.values(),
            temporal_factors.values(),
            environmental_factors.values(),
            organizational_factors.values(),
            threat_landscape_factors.values()
        ]
        
        for factor_group in all_factors:
            for factor_value in factor_group:
                adjusted_risk *= factor_value
        
        # Apply logarithmic scaling to prevent extreme values
        adjusted_risk = 1.0 - math.exp(-adjusted_risk * 2.0)
        
        return min(1.0, adjusted_risk)
    
    def _determine_risk_trajectory(self, baseline_risk: float, adjusted_risk: float,
                                 temporal_factors: Dict[str, float], 
                                 threat_landscape_factors: Dict[str, float]) -> str:
        """Determine risk trajectory"""
        
        # Calculate trend indicators
        trend_score = 0.0
        
        # Temporal trend
        if temporal_factors.get('trend', 1.0) > 1.0:
            trend_score += 0.3
        
        # Threat landscape trend
        if threat_landscape_factors.get('threat_intensity', 1.0) > 1.1:
            trend_score += 0.4
        
        # Risk adjustment trend
        if adjusted_risk > baseline_risk * 1.1:
            trend_score += 0.3
        elif adjusted_risk < baseline_risk * 0.9:
            trend_score -= 0.3
        
        # Determine trajectory
        if trend_score > 0.5:
            return 'increasing'
        elif trend_score < -0.3:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_value_at_risk(self, control: ControlData, assessment_result: AssessmentResult,
                               business_context: Dict[str, any]) -> float:
        """Calculate Value at Risk (VaR)"""
        
        # Base asset value (placeholder - would be from business context)
        base_asset_value = business_context.get('asset_value', 1000000) if business_context else 1000000
        
        # Risk probability
        risk_probability = self._get_risk_probability(assessment_result.risk_level)
        
        # Impact factor
        impact_factor = self._get_impact_factor(control, assessment_result)
        
        # Calculate VaR (95% confidence level)
        var_95 = base_asset_value * risk_probability * impact_factor
        
        return var_95
    
    def _calculate_expected_loss(self, control: ControlData, assessment_result: AssessmentResult,
                               business_context: Dict[str, any]) -> float:
        """Calculate Expected Loss"""
        
        # Base calculations
        probability = self._get_risk_probability(assessment_result.risk_level)
        impact = self._get_impact_factor(control, assessment_result)
        
        # Asset value
        asset_value = business_context.get('asset_value', 1000000) if business_context else 1000000
        
        # Expected loss = Probability * Impact * Asset Value
        expected_loss = probability * impact * asset_value
        
        return expected_loss
    
    def _calculate_risk_exposure(self, control: ControlData, assessment_result: AssessmentResult) -> float:
        """Calculate Risk Exposure"""
        
        # Base exposure from compliance status
        compliance_exposure = {
            'Compliant': 0.2,
            'Partially Compliant': 0.6,
            'Non-Compliant': 0.9,
            'Unknown': 0.5
        }
        
        base_exposure = compliance_exposure.get(assessment_result.compliance_status, 0.5)
        
        # Adjust for evidence quality
        evidence_adjustment = 1.0 + (0.5 - assessment_result.confidence_score)
        
        # Risk exposure
        risk_exposure = base_exposure * evidence_adjustment
        
        return min(1.0, risk_exposure)
    
    def _calculate_risk_concentration(self, control: ControlData, assessment_result: AssessmentResult) -> float:
        """Calculate Risk Concentration"""
        
        # Risk concentration based on number of risks and their severity
        risk_count = len(assessment_result.risks)
        
        if risk_count == 0:
            return 0.1
        elif risk_count <= 2:
            return 0.4
        elif risk_count <= 4:
            return 0.7
        else:
            return 0.9
    
    def _calculate_risk_velocity(self, control: ControlData, assessment_result: AssessmentResult) -> float:
        """Calculate Risk Velocity (speed of risk realization)"""
        
        # Base velocity from risk level
        risk_velocity_base = {
            'Critical': 0.9,
            'High': 0.7,
            'Medium': 0.5,
            'Low': 0.3,
            'Unknown': 0.5
        }
        
        base_velocity = risk_velocity_base.get(assessment_result.risk_level, 0.5)
        
        # Adjust for control type
        control_text = control.control_name.lower()
        if 'access' in control_text:
            base_velocity *= 1.2  # Access controls can be bypassed quickly
        elif 'monitoring' in control_text:
            base_velocity *= 0.8  # Monitoring failures take time to exploit
        
        return min(1.0, base_velocity)
    
    def _calculate_time_to_impact(self, control: ControlData, assessment_result: AssessmentResult) -> int:
        """Calculate Time to Impact in days"""
        
        # Base time to impact
        base_times = {
            'Critical': 1,
            'High': 7,
            'Medium': 30,
            'Low': 90,
            'Unknown': 30
        }
        
        base_time = base_times.get(assessment_result.risk_level, 30)
        
        # Adjust for compliance status
        if assessment_result.compliance_status == 'Non-Compliant':
            base_time = int(base_time * 0.5)  # Faster impact for non-compliant
        elif assessment_result.compliance_status == 'Compliant':
            base_time = int(base_time * 2.0)  # Slower impact for compliant
        
        return max(1, base_time)
    
    def _calculate_recovery_time(self, control: ControlData, assessment_result: AssessmentResult) -> int:
        """Calculate Recovery Time in days"""
        
        # Base recovery time
        base_recovery = {
            'Critical': 3,
            'High': 7,
            'Medium': 14,
            'Low': 30,
            'Unknown': 14
        }
        
        recovery_time = base_recovery.get(assessment_result.risk_level, 14)
        
        # Adjust for number of remediation steps
        remediation_count = len(assessment_result.remediation)
        if remediation_count > 3:
            recovery_time *= 2
        elif remediation_count > 1:
            recovery_time = int(recovery_time * 1.5)
        
        return recovery_time
    
    def _calculate_business_impact_score(self, control: ControlData, assessment_result: AssessmentResult,
                                       business_context: Dict[str, any]) -> float:
        """Calculate Business Impact Score"""
        
        # Base business impact
        base_impact = 0.5
        
        # Adjust for control criticality
        control_text = f"{control.control_name} {control.requirement}".lower()
        if any(term in control_text for term in ['critical', 'essential', 'business']):
            base_impact += 0.3
        
        # Adjust for compliance status
        if assessment_result.compliance_status == 'Non-Compliant':
            base_impact += 0.2
        
        # Adjust for industry
        if business_context:
            industry = business_context.get('industry', 'general')
            industry_multiplier = self.business_context['industry_risk_multipliers'].get(industry, 1.0)
            base_impact *= industry_multiplier
        
        return min(1.0, base_impact)
    
    def _calculate_technical_impact_score(self, control: ControlData, assessment_result: AssessmentResult) -> float:
        """Calculate Technical Impact Score"""
        
        # Base technical impact
        base_impact = 0.5
        
        # Adjust for technical details
        if assessment_result.technical_details:
            base_impact += 0.2
        
        # Adjust for control type
        control_text = control.control_name.lower()
        if 'technical' in control_text or 'system' in control_text:
            base_impact += 0.3
        
        return min(1.0, base_impact)
    
    def _calculate_compliance_impact_score(self, control: ControlData, assessment_result: AssessmentResult) -> float:
        """Calculate Compliance Impact Score"""
        
        # Base compliance impact
        compliance_impact = {
            'Compliant': 0.2,
            'Partially Compliant': 0.6,
            'Non-Compliant': 0.9,
            'Unknown': 0.5
        }
        
        base_impact = compliance_impact.get(assessment_result.compliance_status, 0.5)
        
        # Adjust for control requirement complexity
        requirement_length = len(control.requirement)
        if requirement_length > 200:
            base_impact += 0.1  # Complex requirements = higher compliance impact
        
        return min(1.0, base_impact)
    
    def _analyze_control_interdependency(self, control1: ControlData, control2: ControlData,
                                       result1: AssessmentResult, result2: AssessmentResult) -> RiskInterdependency:
        """Analyze interdependency between two controls"""
        
        # Calculate correlation strength
        correlation_strength = 0.0
        
        # Check for common keywords
        keywords1 = set(control1.control_name.lower().split())
        keywords2 = set(control2.control_name.lower().split())
        common_keywords = keywords1.intersection(keywords2)
        
        if common_keywords:
            correlation_strength += 0.3
        
        # Check for risk correlation
        if result1.risk_level == result2.risk_level:
            correlation_strength += 0.2
        
        # Check for compliance correlation
        if result1.compliance_status == result2.compliance_status:
            correlation_strength += 0.2
        
        # Check for common risks
        common_risks = set(result1.risks).intersection(set(result2.risks))
        if common_risks:
            correlation_strength += 0.3
        
        # Determine dependency type
        dependency_type = "none"
        if correlation_strength > 0.7:
            dependency_type = "strong"
        elif correlation_strength > 0.5:
            dependency_type = "moderate"
        elif correlation_strength > 0.3:
            dependency_type = "weak"
        
        # Calculate amplification factor
        amplification_factor = 1.0 + (correlation_strength * 0.5)
        
        return RiskInterdependency(
            primary_risk=control1.control_id,
            dependent_risk=control2.control_id,
            dependency_type=dependency_type,
            correlation_strength=correlation_strength,
            amplification_factor=amplification_factor,
            description=f"Controls {control1.control_id} and {control2.control_id} show {dependency_type} interdependency"
        )
    
    def _get_risk_probability(self, risk_level: str) -> float:
        """Get risk probability based on risk level"""
        
        probabilities = {
            'Critical': 0.8,
            'High': 0.6,
            'Medium': 0.4,
            'Low': 0.2,
            'Unknown': 0.3
        }
        
        return probabilities.get(risk_level, 0.3)
    
    def _get_impact_factor(self, control: ControlData, assessment_result: AssessmentResult) -> float:
        """Get impact factor based on control and assessment"""
        
        # Base impact factor
        base_impact = 0.5
        
        # Adjust for compliance status
        if assessment_result.compliance_status == 'Non-Compliant':
            base_impact += 0.3
        elif assessment_result.compliance_status == 'Partially Compliant':
            base_impact += 0.2
        
        # Adjust for control criticality
        control_text = control.control_name.lower()
        if any(term in control_text for term in ['critical', 'essential', 'key']):
            base_impact += 0.2
        
        return min(1.0, base_impact)
    
    def generate_risk_report(self, risk_profiles: List[DynamicRiskProfile],
                           quantitative_metrics: List[QuantitativeRiskMetrics],
                           interdependencies: List[RiskInterdependency],
                           output_path: str = "risk_report.json"):
        """Generate comprehensive risk report"""
        
        try:
            report_data = {
                'risk_profiles': [profile.to_dict() for profile in risk_profiles],
                'quantitative_metrics': [metrics.to_dict() for metrics in quantitative_metrics],
                'interdependencies': [
                    {
                        'primary_risk': dep.primary_risk,
                        'dependent_risk': dep.dependent_risk,
                        'dependency_type': dep.dependency_type,
                        'correlation_strength': dep.correlation_strength,
                        'amplification_factor': dep.amplification_factor,
                        'description': dep.description
                    } for dep in interdependencies
                ],
                'summary': {
                    'total_controls': len(risk_profiles),
                    'high_risk_controls': len([p for p in risk_profiles if p.adjusted_risk > 0.7]),
                    'increasing_risk_controls': len([p for p in risk_profiles if p.risk_trajectory == 'increasing']),
                    'total_interdependencies': len(interdependencies),
                    'strong_interdependencies': len([d for d in interdependencies if d.dependency_type == 'strong'])
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Risk report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export risk report: {e}")
    
    def get_risk_treatment_recommendations(self, risk_profile: DynamicRiskProfile,
                                        quantitative_metrics: QuantitativeRiskMetrics) -> List[str]:
        """Get risk treatment recommendations"""
        
        recommendations = []
        
        # High risk recommendations
        if risk_profile.adjusted_risk > 0.8:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Implement emergency risk mitigation measures")
            recommendations.append("Consider risk transfer options (insurance, outsourcing)")
        
        # Increasing risk recommendations
        if risk_profile.risk_trajectory == 'increasing':
            recommendations.append("Monitor risk trajectory closely and implement preventive measures")
            recommendations.append("Review and update risk assessment regularly")
        
        # High VaR recommendations
        if quantitative_metrics.value_at_risk > 500000:  # Example threshold
            recommendations.append("Consider financial risk transfer mechanisms")
            recommendations.append("Implement additional risk controls to reduce exposure")
        
        # Fast risk velocity recommendations
        if quantitative_metrics.risk_velocity > 0.7:
            recommendations.append("Implement rapid response capabilities")
            recommendations.append("Establish continuous monitoring for early detection")
        
        return recommendations[:5]  # Limit to top 5 recommendations