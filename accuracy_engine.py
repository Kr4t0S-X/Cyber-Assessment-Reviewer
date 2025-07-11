"""
Enhanced Accuracy Calculation Engine for Cyber Assessment Reviewer
Implements multi-dimensional accuracy scoring with framework-specific weights
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from models import AssessmentResult, ControlData, RiskMetrics

logger = logging.getLogger(__name__)

class AccuracyDimension(Enum):
    """Accuracy scoring dimensions"""
    COMPLIANCE_ACCURACY = "compliance_accuracy"
    RISK_ASSESSMENT_ACCURACY = "risk_assessment_accuracy"
    FINDING_QUALITY = "finding_quality"
    EVIDENCE_UTILIZATION = "evidence_utilization"
    TECHNICAL_SPECIFICITY = "technical_specificity"
    REMEDIATION_ACTIONABILITY = "remediation_actionability"
    CONSISTENCY = "consistency"
    CONFIDENCE_CALIBRATION = "confidence_calibration"

@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics"""
    overall_accuracy: float
    dimension_scores: Dict[str, float]
    framework_specific_score: float
    confidence_weighted_score: float
    improvement_trend: float
    detailed_breakdown: Dict[str, any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'overall_accuracy': self.overall_accuracy,
            'dimension_scores': self.dimension_scores,
            'framework_specific_score': self.framework_specific_score,
            'confidence_weighted_score': self.confidence_weighted_score,
            'improvement_trend': self.improvement_trend,
            'detailed_breakdown': self.detailed_breakdown,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class FrameworkAccuracyBaseline:
    """Framework-specific accuracy baselines"""
    framework: str
    baseline_scores: Dict[str, float]
    expected_ranges: Dict[str, Tuple[float, float]]
    critical_dimensions: List[str]
    weight_adjustments: Dict[str, float]

class EnhancedAccuracyEngine:
    """Enhanced accuracy calculation engine with multi-dimensional scoring"""
    
    def __init__(self, config=None):
        self.config = config
        self.accuracy_history = []
        self.framework_baselines = self._initialize_framework_baselines()
        self.dimension_weights = self._initialize_dimension_weights()
        
    def _initialize_framework_baselines(self) -> Dict[str, FrameworkAccuracyBaseline]:
        """Initialize framework-specific accuracy baselines"""
        baselines = {
            "NIST": FrameworkAccuracyBaseline(
                framework="NIST",
                baseline_scores={
                    "compliance_accuracy": 0.85,
                    "risk_assessment_accuracy": 0.80,
                    "finding_quality": 0.75,
                    "evidence_utilization": 0.80,
                    "technical_specificity": 0.70,
                    "remediation_actionability": 0.75,
                    "consistency": 0.85,
                    "confidence_calibration": 0.80
                },
                expected_ranges={
                    "compliance_accuracy": (0.80, 0.95),
                    "risk_assessment_accuracy": (0.75, 0.90),
                    "finding_quality": (0.70, 0.85),
                    "evidence_utilization": (0.75, 0.90),
                    "technical_specificity": (0.65, 0.80),
                    "remediation_actionability": (0.70, 0.85),
                    "consistency": (0.80, 0.95),
                    "confidence_calibration": (0.75, 0.90)
                },
                critical_dimensions=["compliance_accuracy", "risk_assessment_accuracy", "consistency"],
                weight_adjustments={
                    "compliance_accuracy": 1.2,
                    "risk_assessment_accuracy": 1.1,
                    "consistency": 1.1
                }
            ),
            "ISO27001": FrameworkAccuracyBaseline(
                framework="ISO27001",
                baseline_scores={
                    "compliance_accuracy": 0.88,
                    "risk_assessment_accuracy": 0.85,
                    "finding_quality": 0.80,
                    "evidence_utilization": 0.85,
                    "technical_specificity": 0.75,
                    "remediation_actionability": 0.80,
                    "consistency": 0.90,
                    "confidence_calibration": 0.85
                },
                expected_ranges={
                    "compliance_accuracy": (0.85, 0.95),
                    "risk_assessment_accuracy": (0.80, 0.95),
                    "finding_quality": (0.75, 0.90),
                    "evidence_utilization": (0.80, 0.95),
                    "technical_specificity": (0.70, 0.85),
                    "remediation_actionability": (0.75, 0.90),
                    "consistency": (0.85, 0.95),
                    "confidence_calibration": (0.80, 0.95)
                },
                critical_dimensions=["compliance_accuracy", "consistency", "evidence_utilization"],
                weight_adjustments={
                    "compliance_accuracy": 1.3,
                    "consistency": 1.2,
                    "evidence_utilization": 1.1
                }
            )
        }
        
        # Add more frameworks
        for framework in ["SOC2", "CIS", "PCI-DSS"]:
            baselines[framework] = self._create_default_baseline(framework)
        
        return baselines
    
    def _create_default_baseline(self, framework: str) -> FrameworkAccuracyBaseline:
        """Create default baseline for frameworks not explicitly defined"""
        return FrameworkAccuracyBaseline(
            framework=framework,
            baseline_scores={
                "compliance_accuracy": 0.82,
                "risk_assessment_accuracy": 0.78,
                "finding_quality": 0.75,
                "evidence_utilization": 0.80,
                "technical_specificity": 0.70,
                "remediation_actionability": 0.75,
                "consistency": 0.85,
                "confidence_calibration": 0.80
            },
            expected_ranges={
                "compliance_accuracy": (0.75, 0.90),
                "risk_assessment_accuracy": (0.70, 0.85),
                "finding_quality": (0.65, 0.80),
                "evidence_utilization": (0.70, 0.85),
                "technical_specificity": (0.60, 0.75),
                "remediation_actionability": (0.65, 0.80),
                "consistency": (0.75, 0.90),
                "confidence_calibration": (0.70, 0.85)
            },
            critical_dimensions=["compliance_accuracy", "risk_assessment_accuracy"],
            weight_adjustments={
                "compliance_accuracy": 1.1,
                "risk_assessment_accuracy": 1.0
            }
        )
    
    def _initialize_dimension_weights(self) -> Dict[str, float]:
        """Initialize base dimension weights"""
        return {
            "compliance_accuracy": 0.20,
            "risk_assessment_accuracy": 0.18,
            "finding_quality": 0.15,
            "evidence_utilization": 0.12,
            "technical_specificity": 0.10,
            "remediation_actionability": 0.10,
            "consistency": 0.10,
            "confidence_calibration": 0.05
        }
    
    def calculate_enhanced_accuracy(self, ai_results: List[AssessmentResult], 
                                  expected_results: List[Dict[str, any]], 
                                  framework: str,
                                  evidence_quality_scores: Dict[str, float] = None) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics"""
        
        if not ai_results or not expected_results:
            return self._create_empty_metrics()
        
        # Calculate each dimension
        dimension_scores = {}
        
        # 1. Compliance Accuracy
        dimension_scores["compliance_accuracy"] = self._calculate_compliance_accuracy(
            ai_results, expected_results
        )
        
        # 2. Risk Assessment Accuracy
        dimension_scores["risk_assessment_accuracy"] = self._calculate_risk_assessment_accuracy(
            ai_results, expected_results
        )
        
        # 3. Finding Quality
        dimension_scores["finding_quality"] = self._calculate_finding_quality(
            ai_results, expected_results
        )
        
        # 4. Evidence Utilization
        dimension_scores["evidence_utilization"] = self._calculate_evidence_utilization(
            ai_results, evidence_quality_scores or {}
        )
        
        # 5. Technical Specificity
        dimension_scores["technical_specificity"] = self._calculate_technical_specificity(
            ai_results
        )
        
        # 6. Remediation Actionability
        dimension_scores["remediation_actionability"] = self._calculate_remediation_actionability(
            ai_results
        )
        
        # 7. Consistency
        dimension_scores["consistency"] = self._calculate_consistency(
            ai_results, expected_results
        )
        
        # 8. Confidence Calibration
        dimension_scores["confidence_calibration"] = self._calculate_confidence_calibration(
            ai_results, expected_results
        )
        
        # Calculate overall accuracy with framework-specific weights
        overall_accuracy = self._calculate_overall_accuracy(
            dimension_scores, framework
        )
        
        # Calculate framework-specific score
        framework_specific_score = self._calculate_framework_specific_score(
            dimension_scores, framework
        )
        
        # Calculate confidence-weighted score
        confidence_weighted_score = self._calculate_confidence_weighted_score(
            dimension_scores, ai_results
        )
        
        # Calculate improvement trend
        improvement_trend = self._calculate_improvement_trend(overall_accuracy)
        
        # Create detailed breakdown
        detailed_breakdown = self._create_detailed_breakdown(
            ai_results, expected_results, dimension_scores, framework
        )
        
        metrics = AccuracyMetrics(
            overall_accuracy=overall_accuracy,
            dimension_scores=dimension_scores,
            framework_specific_score=framework_specific_score,
            confidence_weighted_score=confidence_weighted_score,
            improvement_trend=improvement_trend,
            detailed_breakdown=detailed_breakdown,
            timestamp=datetime.now()
        )
        
        # Store in history
        self.accuracy_history.append(metrics)
        
        return metrics
    
    def _calculate_compliance_accuracy(self, ai_results: List[AssessmentResult], 
                                     expected_results: List[Dict[str, any]]) -> float:
        """Calculate compliance status accuracy"""
        if not ai_results or not expected_results:
            return 0.0
        
        matches = 0
        total = min(len(ai_results), len(expected_results))
        
        for i in range(total):
            ai_compliance = ai_results[i].compliance_status
            expected_compliance = expected_results[i].get('expected_compliance', 'Unknown')
            
            if ai_compliance == expected_compliance:
                matches += 1
            elif self._is_partial_match(ai_compliance, expected_compliance):
                matches += 0.5
        
        return matches / total if total > 0 else 0.0
    
    def _calculate_risk_assessment_accuracy(self, ai_results: List[AssessmentResult], 
                                          expected_results: List[Dict[str, any]]) -> float:
        """Calculate risk level accuracy with severity tolerance"""
        if not ai_results or not expected_results:
            return 0.0
        
        risk_values = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4, "Unknown": 0}
        total_score = 0
        total_comparisons = 0
        
        for i in range(min(len(ai_results), len(expected_results))):
            ai_risk = ai_results[i].risk_level
            expected_risk = expected_results[i].get('expected_risk_level', 'Unknown')
            
            ai_value = risk_values.get(ai_risk, 0)
            expected_value = risk_values.get(expected_risk, 0)
            
            if ai_value == expected_value:
                total_score += 1.0
            elif abs(ai_value - expected_value) == 1:
                total_score += 0.7  # Close but not exact
            elif abs(ai_value - expected_value) == 2:
                total_score += 0.3  # Moderate difference
            # More than 2 levels off gets 0 points
            
            total_comparisons += 1
        
        return total_score / total_comparisons if total_comparisons > 0 else 0.0
    
    def _calculate_finding_quality(self, ai_results: List[AssessmentResult], 
                                 expected_results: List[Dict[str, any]]) -> float:
        """Calculate quality of findings against expected findings"""
        if not ai_results:
            return 0.0
        
        quality_scores = []
        
        for i, result in enumerate(ai_results):
            expected = expected_results[i] if i < len(expected_results) else {}
            expected_findings = expected.get('expected_findings', [])
            
            if not expected_findings:
                # Score based on finding completeness and specificity
                quality_score = self._score_finding_completeness(result.key_findings)
            else:
                # Score based on similarity to expected findings
                quality_score = self._score_finding_similarity(result.key_findings, expected_findings)
            
            quality_scores.append(quality_score)
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    def _calculate_evidence_utilization(self, ai_results: List[AssessmentResult], 
                                      evidence_quality_scores: Dict[str, float]) -> float:
        """Calculate how well evidence was utilized"""
        if not ai_results:
            return 0.0
        
        utilization_scores = []
        
        for result in ai_results:
            # Score based on evidence references and quality
            evidence_score = 0.0
            
            # Check if evidence is referenced in findings
            evidence_references = len(result.evidence_references)
            if evidence_references > 0:
                evidence_score += 0.3
            
            # Check if technical details are extracted
            if result.technical_details:
                evidence_score += 0.3
            
            # Check if evidence gaps are identified
            if result.evidence_gaps:
                evidence_score += 0.2
            
            # Check quality of evidence utilization
            control_evidence_score = evidence_quality_scores.get(result.control_id, 0.5)
            evidence_score += control_evidence_score * 0.2
            
            utilization_scores.append(min(1.0, evidence_score))
        
        return sum(utilization_scores) / len(utilization_scores) if utilization_scores else 0.0
    
    def _calculate_technical_specificity(self, ai_results: List[AssessmentResult]) -> float:
        """Calculate technical specificity of analysis"""
        if not ai_results:
            return 0.0
        
        technical_terms = [
            'configuration', 'implementation', 'policy', 'procedure', 'control',
            'authentication', 'authorization', 'encryption', 'monitoring', 'logging',
            'vulnerability', 'patch', 'access', 'network', 'firewall', 'certificate',
            'audit', 'compliance', 'security', 'risk', 'threat', 'incident'
        ]
        
        specificity_scores = []
        
        for result in ai_results:
            # Combine all text fields
            all_text = ' '.join(
                result.key_findings + result.risks + result.remediation + 
                result.technical_details + [result.compliance_rationale]
            ).lower()
            
            # Count technical terms
            term_count = sum(1 for term in technical_terms if term in all_text)
            
            # Score based on term density and technical details presence
            text_length = len(all_text.split())
            if text_length > 0:
                term_density = term_count / text_length
                base_score = min(1.0, term_density * 10)  # Scale up density
                
                # Bonus for having technical details
                if result.technical_details:
                    base_score += 0.2
                
                specificity_scores.append(min(1.0, base_score))
            else:
                specificity_scores.append(0.0)
        
        return sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0.0
    
    def _calculate_remediation_actionability(self, ai_results: List[AssessmentResult]) -> float:
        """Calculate actionability of remediation steps"""
        if not ai_results:
            return 0.0
        
        actionable_verbs = [
            'implement', 'configure', 'enable', 'disable', 'update', 'install',
            'review', 'establish', 'develop', 'document', 'create', 'deploy',
            'monitor', 'assess', 'validate', 'verify', 'test', 'upgrade'
        ]
        
        actionability_scores = []
        
        for result in ai_results:
            if not result.remediation:
                actionability_scores.append(0.0)
                continue
            
            actionable_count = 0
            for remediation in result.remediation:
                if any(verb in remediation.lower() for verb in actionable_verbs):
                    actionable_count += 1
            
            # Score based on percentage of actionable steps
            actionability_score = actionable_count / len(result.remediation)
            
            # Bonus for specific timelines or priorities
            priority_indicators = ['immediate', 'urgent', 'first', 'priority', 'critical']
            if any(indicator in ' '.join(result.remediation).lower() for indicator in priority_indicators):
                actionability_score += 0.2
            
            actionability_scores.append(min(1.0, actionability_score))
        
        return sum(actionability_scores) / len(actionability_scores) if actionability_scores else 0.0
    
    def _calculate_consistency(self, ai_results: List[AssessmentResult], 
                             expected_results: List[Dict[str, any]]) -> float:
        """Calculate internal consistency of assessments"""
        if not ai_results:
            return 0.0
        
        consistency_scores = []
        
        for result in ai_results:
            consistency_score = 1.0
            
            # Check compliance-risk alignment
            if result.compliance_status == "Compliant" and result.risk_level in ["Critical", "High"]:
                consistency_score -= 0.3
            
            if result.compliance_status == "Non-Compliant" and result.risk_level in ["Low"]:
                consistency_score -= 0.3
            
            # Check evidence-confidence alignment
            if result.evidence_validity == "Invalid" and result.confidence_score > 0.7:
                consistency_score -= 0.2
            
            if result.evidence_validity == "Valid" and result.confidence_score < 0.3:
                consistency_score -= 0.2
            
            # Check evidence gaps vs compliance
            if len(result.evidence_gaps) > 2 and result.compliance_status == "Compliant":
                consistency_score -= 0.2
            
            consistency_scores.append(max(0.0, consistency_score))
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def _calculate_confidence_calibration(self, ai_results: List[AssessmentResult], 
                                        expected_results: List[Dict[str, any]]) -> float:
        """Calculate how well confidence scores align with actual accuracy"""
        if not ai_results or not expected_results:
            return 0.0
        
        calibration_scores = []
        
        for i, result in enumerate(ai_results):
            if i >= len(expected_results):
                break
            
            expected = expected_results[i]
            
            # Calculate actual accuracy for this result
            actual_accuracy = 0.0
            comparisons = 0
            
            if 'expected_compliance' in expected:
                actual_accuracy += 1.0 if result.compliance_status == expected['expected_compliance'] else 0.0
                comparisons += 1
            
            if 'expected_risk_level' in expected:
                risk_values = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
                ai_risk = risk_values.get(result.risk_level, 0)
                expected_risk = risk_values.get(expected['expected_risk_level'], 0)
                actual_accuracy += 1.0 if ai_risk == expected_risk else max(0.0, 1.0 - abs(ai_risk - expected_risk) * 0.3)
                comparisons += 1
            
            if comparisons > 0:
                actual_accuracy /= comparisons
                
                # Compare with confidence score
                confidence_diff = abs(result.confidence_score - actual_accuracy)
                calibration_score = max(0.0, 1.0 - confidence_diff * 2)  # Penalize large differences
                calibration_scores.append(calibration_score)
        
        return sum(calibration_scores) / len(calibration_scores) if calibration_scores else 0.0
    
    def _calculate_overall_accuracy(self, dimension_scores: Dict[str, float], 
                                  framework: str) -> float:
        """Calculate overall accuracy with framework-specific weights"""
        baseline = self.framework_baselines.get(framework, self.framework_baselines["NIST"])
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            base_weight = self.dimension_weights.get(dimension, 0.0)
            framework_adjustment = baseline.weight_adjustments.get(dimension, 1.0)
            final_weight = base_weight * framework_adjustment
            
            weighted_score += score * final_weight
            total_weight += final_weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_framework_specific_score(self, dimension_scores: Dict[str, float], 
                                          framework: str) -> float:
        """Calculate framework-specific accuracy score"""
        baseline = self.framework_baselines.get(framework, self.framework_baselines["NIST"])
        
        # Weight critical dimensions more heavily
        critical_weight = 0.0
        critical_score = 0.0
        
        for dimension in baseline.critical_dimensions:
            if dimension in dimension_scores:
                critical_score += dimension_scores[dimension]
                critical_weight += 1.0
        
        critical_avg = critical_score / critical_weight if critical_weight > 0 else 0.0
        
        # Combine with overall score
        overall_avg = sum(dimension_scores.values()) / len(dimension_scores)
        
        return 0.7 * critical_avg + 0.3 * overall_avg
    
    def _calculate_confidence_weighted_score(self, dimension_scores: Dict[str, float], 
                                           ai_results: List[AssessmentResult]) -> float:
        """Calculate confidence-weighted accuracy score"""
        if not ai_results:
            return 0.0
        
        # Weight the overall accuracy by confidence scores
        total_confidence = sum(result.confidence_score for result in ai_results)
        avg_confidence = total_confidence / len(ai_results)
        
        overall_accuracy = sum(dimension_scores.values()) / len(dimension_scores)
        
        # Adjust based on confidence calibration
        confidence_adjustment = min(1.0, avg_confidence * 1.2)  # Slight boost for high confidence
        
        return overall_accuracy * confidence_adjustment
    
    def _calculate_improvement_trend(self, current_accuracy: float) -> float:
        """Calculate improvement trend over recent assessments"""
        if len(self.accuracy_history) < 2:
            return 0.0
        
        # Look at last 5 assessments
        recent_accuracies = [m.overall_accuracy for m in self.accuracy_history[-5:]]
        
        if len(recent_accuracies) < 2:
            return 0.0
        
        # Simple linear trend
        x = list(range(len(recent_accuracies)))
        y = recent_accuracies
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        return slope  # Positive = improving, negative = declining
    
    def _create_detailed_breakdown(self, ai_results: List[AssessmentResult], 
                                 expected_results: List[Dict[str, any]], 
                                 dimension_scores: Dict[str, float], 
                                 framework: str) -> Dict[str, any]:
        """Create detailed accuracy breakdown"""
        baseline = self.framework_baselines.get(framework, self.framework_baselines["NIST"])
        
        breakdown = {
            'framework': framework,
            'total_assessments': len(ai_results),
            'baseline_comparison': {},
            'dimension_analysis': {},
            'improvement_areas': [],
            'strengths': []
        }
        
        # Compare against baseline
        for dimension, score in dimension_scores.items():
            baseline_score = baseline.baseline_scores.get(dimension, 0.7)
            expected_range = baseline.expected_ranges.get(dimension, (0.6, 0.8))
            
            breakdown['baseline_comparison'][dimension] = {
                'score': score,
                'baseline': baseline_score,
                'expected_range': expected_range,
                'performance': 'above' if score > baseline_score else 'below' if score < baseline_score else 'at',
                'in_range': expected_range[0] <= score <= expected_range[1]
            }
            
            # Identify improvement areas and strengths
            if score < baseline_score - 0.1:
                breakdown['improvement_areas'].append({
                    'dimension': dimension,
                    'score': score,
                    'gap': baseline_score - score
                })
            elif score > baseline_score + 0.1:
                breakdown['strengths'].append({
                    'dimension': dimension,
                    'score': score,
                    'advantage': score - baseline_score
                })
        
        # Dimension-specific analysis
        for dimension, score in dimension_scores.items():
            breakdown['dimension_analysis'][dimension] = {
                'score': score,
                'weight': self.dimension_weights.get(dimension, 0.0),
                'framework_weight': baseline.weight_adjustments.get(dimension, 1.0),
                'is_critical': dimension in baseline.critical_dimensions
            }
        
        return breakdown
    
    def _score_finding_completeness(self, findings: List[str]) -> float:
        """Score findings based on completeness and specificity"""
        if not findings:
            return 0.0
        
        score = 0.0
        
        # Base score for having findings
        score += 0.3
        
        # Score based on number of findings
        score += min(0.3, len(findings) * 0.1)
        
        # Score based on specificity
        specific_terms = ['specific', 'configuration', 'implementation', 'documented', 'evidence']
        for finding in findings:
            if any(term in finding.lower() for term in specific_terms):
                score += 0.1
        
        return min(1.0, score)
    
    def _score_finding_similarity(self, ai_findings: List[str], 
                                expected_findings: List[str]) -> float:
        """Score findings based on similarity to expected findings"""
        if not expected_findings:
            return self._score_finding_completeness(ai_findings)
        
        if not ai_findings:
            return 0.0
        
        matches = 0
        for expected in expected_findings:
            expected_words = set(expected.lower().split())
            for ai_finding in ai_findings:
                ai_words = set(ai_finding.lower().split())
                overlap = len(expected_words.intersection(ai_words))
                similarity = overlap / len(expected_words.union(ai_words))
                if similarity > 0.3:
                    matches += 1
                    break
        
        return matches / len(expected_findings)
    
    def _is_partial_match(self, ai_compliance: str, expected_compliance: str) -> bool:
        """Check if compliance statuses are partial matches"""
        partial_matches = [
            ("Compliant", "Partially Compliant"),
            ("Non-Compliant", "Partially Compliant"),
            ("Partially Compliant", "Compliant"),
            ("Partially Compliant", "Non-Compliant")
        ]
        
        return (ai_compliance, expected_compliance) in partial_matches or \
               (expected_compliance, ai_compliance) in partial_matches
    
    def _create_empty_metrics(self) -> AccuracyMetrics:
        """Create empty accuracy metrics for error cases"""
        return AccuracyMetrics(
            overall_accuracy=0.0,
            dimension_scores={},
            framework_specific_score=0.0,
            confidence_weighted_score=0.0,
            improvement_trend=0.0,
            detailed_breakdown={},
            timestamp=datetime.now()
        )
    
    def export_accuracy_report(self, metrics: AccuracyMetrics, 
                             output_path: str = "accuracy_report.json"):
        """Export detailed accuracy report"""
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            logger.info(f"Accuracy report exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export accuracy report: {e}")
    
    def get_improvement_recommendations(self, metrics: AccuracyMetrics) -> List[str]:
        """Get recommendations for improving accuracy"""
        recommendations = []
        
        for area in metrics.detailed_breakdown.get('improvement_areas', []):
            dimension = area['dimension']
            gap = area['gap']
            
            if dimension == 'compliance_accuracy' and gap > 0.1:
                recommendations.append(
                    "Improve compliance determination by enhancing evidence analysis and "
                    "framework requirement mapping"
                )
            elif dimension == 'risk_assessment_accuracy' and gap > 0.1:
                recommendations.append(
                    "Enhance risk assessment by implementing better threat modeling and "
                    "impact analysis"
                )
            elif dimension == 'finding_quality' and gap > 0.1:
                recommendations.append(
                    "Improve finding quality by increasing technical specificity and "
                    "evidence references"
                )
            elif dimension == 'technical_specificity' and gap > 0.1:
                recommendations.append(
                    "Increase technical specificity by extracting more detailed technical "
                    "information from evidence"
                )
        
        return recommendations