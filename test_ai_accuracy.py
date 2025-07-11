#!/usr/bin/env python3
"""
AI Accuracy Testing Framework for Cyber Assessment Reviewer
Tests and validates AI model improvements with real cybersecurity scenarios
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from models import AssessmentResult, ControlData
from ai_backend import CyberAssessmentReviewer
from analysis_workflow import OptimizedAnalysisWorkflow
from cybersecurity_knowledge import CybersecurityKnowledgeBase
from config import Config
from accuracy_engine import EnhancedAccuracyEngine, AccuracyMetrics
from evidence_analyzer import AdvancedEvidenceAnalyzer
from findings_generator import EnhancedFindingsGenerator
from risk_engine import AdvancedRiskEngine

logger = logging.getLogger(__name__)

def safe_dataclass_to_dict(obj):
    """Safely convert dataclass to dictionary, handling None and non-dataclass objects"""
    if obj is None:
        return None

    # Check if it's a dataclass instance
    if hasattr(obj, '__dataclass_fields__'):
        try:
            return asdict(obj)
        except Exception as e:
            logger.warning(f"Failed to convert dataclass to dict: {e}")
            # Fallback: manually convert to dict
            return {field: getattr(obj, field, None) for field in obj.__dataclass_fields__}

    # If it's already a dict, return as is
    if isinstance(obj, dict):
        return obj

    # For other types, try to convert to string representation
    try:
        return str(obj)
    except Exception:
        return f"<Unserializable object of type {type(obj).__name__}>"

@dataclass
class TestCase:
    """Test case for AI accuracy validation"""
    test_id: str
    control: ControlData
    evidence_text: str
    expected_compliance: str
    expected_risk_level: str
    expected_findings: List[str]
    framework: str
    description: str

@dataclass
class TestResult:
    """Result of AI accuracy test"""
    test_id: str
    passed: bool
    accuracy_score: float
    compliance_match: bool
    risk_level_match: bool
    findings_quality: float
    issues: List[str]
    ai_result: AssessmentResult
    # Enhanced metrics
    accuracy_metrics: AccuracyMetrics = None
    evidence_quality_score: float = 0.0
    technical_specificity_score: float = 0.0
    risk_assessment_score: float = 0.0
    enhanced_findings: List[dict] = None
    
    def __post_init__(self):
        if self.enhanced_findings is None:
            self.enhanced_findings = []

class AIAccuracyTester:
    """Framework for testing AI accuracy improvements"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.reviewer = CyberAssessmentReviewer(self.config)
        self.workflow = OptimizedAnalysisWorkflow(self.reviewer)
        self.knowledge_base = CybersecurityKnowledgeBase()
        # Enhanced components
        self.accuracy_engine = EnhancedAccuracyEngine(self.config)
        self.evidence_analyzer = AdvancedEvidenceAnalyzer(self.config)
        self.findings_generator = EnhancedFindingsGenerator(self.config)
        self.risk_engine = AdvancedRiskEngine(self.config)
        
    def run_accuracy_tests(self, test_cases: List[TestCase]) -> Dict[str, any]:
        """Run comprehensive accuracy tests"""
        logger.info(f"Running {len(test_cases)} AI accuracy tests")
        
        results = []
        for test_case in test_cases:
            logger.info(f"Running test: {test_case.test_id}")
            result = self._run_single_test(test_case)
            results.append(result)
        
        # Calculate overall metrics
        metrics = self._calculate_test_metrics(results)
        
        # Generate test report
        report = self._generate_test_report(results, metrics)
        
        return {
            "results": results,
            "metrics": metrics,
            "report": report
        }
    
    def _run_single_test(self, test_case: TestCase) -> TestResult:
        """Run a single accuracy test"""
        try:
            # Prepare evidence for analysis
            evidence_texts = {
                f"test_evidence_{test_case.test_id}": {
                    "test_section": test_case.evidence_text
                }
            }
            
            # Run AI analysis
            ai_results = self.workflow.analyze_controls_systematically(
                [test_case.control], 
                evidence_texts, 
                test_case.framework
            )
            
            if not ai_results:
                return TestResult(
                    test_id=test_case.test_id,
                    passed=False,
                    accuracy_score=0.0,
                    compliance_match=False,
                    risk_level_match=False,
                    findings_quality=0.0,
                    issues=["AI analysis failed to produce results"],
                    ai_result=None
                )
            
            ai_result = ai_results[0]
            
            # Evaluate results
            compliance_match = ai_result.compliance_status == test_case.expected_compliance
            risk_level_match = ai_result.risk_level == test_case.expected_risk_level
            
            # Evaluate findings quality
            findings_quality = self._evaluate_findings_quality(
                ai_result.key_findings, test_case.expected_findings
            )
            
            # Calculate overall accuracy score
            accuracy_score = self._calculate_accuracy_score(
                compliance_match, risk_level_match, findings_quality
            )
            
            # Enhanced accuracy calculation
            enhanced_accuracy_score, accuracy_metrics = self._calculate_enhanced_accuracy(
                test_case, ai_result, evidence_texts
            )
            
            # Calculate evidence quality score
            evidence_quality_score = self._calculate_evidence_quality_score(
                test_case, evidence_texts
            )
            
            # Calculate technical specificity score
            technical_specificity_score = self._calculate_technical_specificity_score(ai_result)
            
            # Calculate risk assessment score
            risk_assessment_score = self._calculate_risk_assessment_score(test_case, ai_result)
            
            # Generate enhanced findings
            enhanced_findings = self._generate_enhanced_findings(test_case, ai_result, evidence_quality_score)
            
            # Identify issues
            issues = self._identify_test_issues(test_case, ai_result)
            
            # Determine if test passed (enhanced criteria)
            passed = self._determine_test_passed(
                enhanced_accuracy_score, accuracy_metrics, evidence_quality_score, 
                technical_specificity_score, issues
            )
            
            return TestResult(
                test_id=test_case.test_id,
                passed=passed,
                accuracy_score=enhanced_accuracy_score,
                compliance_match=compliance_match,
                risk_level_match=risk_level_match,
                findings_quality=findings_quality,
                issues=issues,
                ai_result=ai_result,
                accuracy_metrics=accuracy_metrics,
                evidence_quality_score=evidence_quality_score,
                technical_specificity_score=technical_specificity_score,
                risk_assessment_score=risk_assessment_score,
                enhanced_findings=enhanced_findings
            )
            
        except Exception as e:
            logger.error(f"Test {test_case.test_id} failed with error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return TestResult(
                test_id=test_case.test_id,
                passed=False,
                accuracy_score=0.0,
                compliance_match=False,
                risk_level_match=False,
                findings_quality=0.0,
                issues=[f"Test execution error: {str(e)}"],
                ai_result=None
            )
    
    def _evaluate_findings_quality(self, ai_findings: List[str], 
                                  expected_findings: List[str]) -> float:
        """Evaluate quality of AI findings against expected findings"""
        if not expected_findings:
            return 1.0 if ai_findings else 0.0
        
        if not ai_findings:
            return 0.0
        
        # Calculate semantic similarity (simplified)
        matches = 0
        for expected in expected_findings:
            expected_words = set(expected.lower().split())
            for ai_finding in ai_findings:
                ai_words = set(ai_finding.lower().split())
                # Simple word overlap similarity
                overlap = len(expected_words.intersection(ai_words))
                similarity = overlap / len(expected_words.union(ai_words))
                if similarity > 0.3:  # Threshold for match
                    matches += 1
                    break
        
        return matches / len(expected_findings)
    
    def _calculate_accuracy_score(self, compliance_match: bool, 
                                 risk_level_match: bool, 
                                 findings_quality: float) -> float:
        """Calculate overall accuracy score"""
        compliance_weight = 0.4
        risk_weight = 0.3
        findings_weight = 0.3
        
        score = (
            (1.0 if compliance_match else 0.0) * compliance_weight +
            (1.0 if risk_level_match else 0.0) * risk_weight +
            findings_quality * findings_weight
        )
        
        return score
    
    def _identify_test_issues(self, test_case: TestCase, 
                             ai_result: AssessmentResult) -> List[str]:
        """Identify issues with AI analysis"""
        issues = []
        
        # Check for major discrepancies
        if test_case.expected_compliance == "Compliant" and ai_result.compliance_status == "Non-Compliant":
            issues.append("AI incorrectly determined non-compliance")
        elif test_case.expected_compliance == "Non-Compliant" and ai_result.compliance_status == "Compliant":
            issues.append("AI incorrectly determined compliance")
        
        # Check risk level appropriateness
        risk_severity = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
        expected_severity = risk_severity.get(test_case.expected_risk_level, 2)
        ai_severity = risk_severity.get(ai_result.risk_level, 2)
        
        if abs(expected_severity - ai_severity) > 1:
            issues.append(f"Risk level significantly off: expected {test_case.expected_risk_level}, got {ai_result.risk_level}")
        
        # Check confidence appropriateness
        if ai_result.confidence_score > 0.8 and not test_case.evidence_text:
            issues.append("High confidence with no evidence")
        
        if ai_result.confidence_score < 0.3 and len(test_case.evidence_text) > 1000:
            issues.append("Low confidence with substantial evidence")
        
        return issues
    
    def _calculate_test_metrics(self, results: List[TestResult]) -> Dict[str, float]:
        """Calculate overall test metrics"""
        if not results:
            return {}
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        
        avg_accuracy = sum(r.accuracy_score for r in results) / total_tests
        compliance_accuracy = sum(1 for r in results if r.compliance_match) / total_tests
        risk_accuracy = sum(1 for r in results if r.risk_level_match) / total_tests
        avg_findings_quality = sum(r.findings_quality for r in results) / total_tests
        
        return {
            "pass_rate": passed_tests / total_tests,
            "average_accuracy": avg_accuracy,
            "compliance_accuracy": compliance_accuracy,
            "risk_level_accuracy": risk_accuracy,
            "findings_quality": avg_findings_quality,
            "total_tests": total_tests,
            "passed_tests": passed_tests
        }
    
    def _generate_test_report(self, results: List[TestResult], 
                             metrics: Dict[str, float]) -> str:
        """Generate comprehensive test report"""
        report = f"""
AI ACCURACY TEST REPORT
=======================

OVERALL METRICS:
- Pass Rate: {metrics.get('pass_rate', 0):.1%}
- Average Accuracy: {metrics.get('average_accuracy', 0):.1%}
- Compliance Accuracy: {metrics.get('compliance_accuracy', 0):.1%}
- Risk Level Accuracy: {metrics.get('risk_level_accuracy', 0):.1%}
- Findings Quality: {metrics.get('findings_quality', 0):.1%}

DETAILED RESULTS:
"""
        
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            report += f"""
Test {result.test_id}: {status}
- Accuracy Score: {result.accuracy_score:.1%}
- Compliance Match: {'✓' if result.compliance_match else '✗'}
- Risk Level Match: {'✓' if result.risk_level_match else '✗'}
- Findings Quality: {result.findings_quality:.1%}
"""
            if result.issues:
                report += f"- Issues: {'; '.join(result.issues)}\n"
        
        return report

    # Enhanced calculation methods
    def _calculate_enhanced_accuracy(self, test_case: TestCase, ai_result: AssessmentResult, 
                                   evidence_texts: Dict[str, Dict[str, str]]) -> Tuple[float, AccuracyMetrics]:
        """Calculate enhanced accuracy using the accuracy engine"""
        try:
            # Create expected result for comparison
            expected_results = [{
                'expected_compliance': test_case.expected_compliance,
                'expected_risk_level': test_case.expected_risk_level,
                'expected_findings': test_case.expected_findings
            }]
            
            # Calculate evidence quality scores
            evidence_quality_scores = {}
            for filename, sections in evidence_texts.items():
                evidence_quality_scores[filename] = self._calculate_file_evidence_quality(
                    test_case.control, sections
                )
            
            # Calculate enhanced accuracy metrics
            accuracy_metrics = self.accuracy_engine.calculate_enhanced_accuracy(
                [ai_result], expected_results, test_case.framework, evidence_quality_scores
            )
            
            return accuracy_metrics.overall_accuracy, accuracy_metrics
            
        except Exception as e:
            logger.warning(f"Enhanced accuracy calculation failed: {e}")
            return 0.5, None
    
    def _calculate_evidence_quality_score(self, test_case: TestCase, 
                                        evidence_texts: Dict[str, Dict[str, str]]) -> float:
        """Calculate evidence quality score"""
        try:
            # Analyze evidence quality
            evidence_metrics = self.evidence_analyzer.analyze_evidence_quality(
                test_case.control, evidence_texts
            )
            return evidence_metrics.overall_quality_score
        except Exception as e:
            logger.warning(f"Evidence quality calculation failed: {e}")
            return 0.5
    
    def _calculate_technical_specificity_score(self, ai_result: AssessmentResult) -> float:
        """Calculate technical specificity score"""
        if not ai_result.technical_details:
            return 0.2
        
        # Score based on technical details presence and quality
        technical_score = 0.0
        
        # Base score for having technical details
        technical_score += 0.3
        
        # Score based on number of technical details
        technical_score += min(0.4, len(ai_result.technical_details) * 0.1)
        
        # Score based on technical term density
        technical_terms = [
            'configuration', 'implementation', 'protocol', 'algorithm', 'encryption',
            'authentication', 'authorization', 'monitoring', 'logging', 'compliance'
        ]
        
        all_technical_text = ' '.join(ai_result.technical_details).lower()
        term_count = sum(1 for term in technical_terms if term in all_technical_text)
        technical_score += min(0.3, term_count * 0.05)
        
        return min(1.0, technical_score)
    
    def _calculate_risk_assessment_score(self, test_case: TestCase, ai_result: AssessmentResult) -> float:
        """Calculate risk assessment score"""
        try:
            # Use risk engine to analyze risk assessment quality
            risk_profile = self.risk_engine.calculate_dynamic_risk(
                test_case.control, ai_result
            )
            
            # Score based on risk profile quality
            risk_score = 0.5
            
            # Adjust based on risk trajectory
            if risk_profile.risk_trajectory == 'stable':
                risk_score += 0.2
            elif risk_profile.risk_trajectory == 'increasing':
                risk_score += 0.1
            
            # Adjust based on baseline vs adjusted risk alignment
            if abs(risk_profile.baseline_risk - risk_profile.adjusted_risk) < 0.3:
                risk_score += 0.3
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.warning(f"Risk assessment calculation failed: {e}")
            return 0.5
    
    def _generate_enhanced_findings(self, test_case: TestCase, ai_result: AssessmentResult,
                                  evidence_quality_score: float) -> List[dict]:
        """Generate enhanced findings for testing"""
        try:
            # Generate enhanced findings
            enhanced_findings = self.findings_generator.generate_enhanced_findings(
                test_case.control, ai_result, evidence_quality_score, test_case.framework
            )
            
            # Convert to dictionary format
            return [finding.to_dict() for finding in enhanced_findings]
            
        except Exception as e:
            logger.warning(f"Enhanced findings generation failed: {e}")
            return []
    
    def _determine_test_passed(self, accuracy_score: float, accuracy_metrics: AccuracyMetrics,
                             evidence_quality_score: float, technical_specificity_score: float,
                             issues: List[str]) -> bool:
        """Determine if test passed using enhanced criteria"""
        
        # Base criteria
        base_passed = accuracy_score >= 0.7 and len(issues) == 0
        
        # Enhanced criteria
        evidence_passed = evidence_quality_score >= 0.4
        technical_passed = technical_specificity_score >= 0.3
        
        # Multi-dimensional accuracy criteria
        multidimensional_passed = True
        if accuracy_metrics:
            # Check key accuracy dimensions
            key_dimensions = ['compliance_accuracy', 'risk_assessment_accuracy', 'finding_quality']
            for dimension in key_dimensions:
                if dimension in accuracy_metrics.dimension_scores:
                    if accuracy_metrics.dimension_scores[dimension] < 0.5:
                        multidimensional_passed = False
                        break
        
        # Overall pass determination
        return base_passed and evidence_passed and technical_passed and multidimensional_passed
    
    def _calculate_file_evidence_quality(self, control: ControlData, sections: Dict[str, str]) -> float:
        """Calculate evidence quality score for a file"""
        if not sections:
            return 0.0
        
        # Simple quality assessment based on content
        total_score = 0.0
        section_count = 0
        
        for section_name, content in sections.items():
            if "Error" in section_name or not content:
                continue
            
            section_score = 0.5  # Base score
            
            # Check for quality indicators
            content_lower = content.lower()
            quality_indicators = [
                'implemented', 'configured', 'documented', 'tested', 'verified',
                'compliant', 'established', 'maintained', 'monitored', 'audited'
            ]
            
            for indicator in quality_indicators:
                if indicator in content_lower:
                    section_score += 0.1
            
            # Length bonus
            if len(content) > 100:
                section_score += 0.2
            
            total_score += min(1.0, section_score)
            section_count += 1
        
        return total_score / section_count if section_count > 0 else 0.0

def create_sample_test_cases() -> List[TestCase]:
    """Create sample test cases for validation"""
    test_cases = [
        TestCase(
            test_id="access_control_compliant",
            control=ControlData(
                control_id="PR.AC-1",
                control_name="Access Control Policy",
                requirement="Identities and credentials are issued, managed, verified, revoked, and audited for authorized devices, users and processes",
                answer="We have implemented a comprehensive identity and access management system with automated provisioning and regular access reviews."
            ),
            evidence_text="""
            Access Control Policy Document:
            - Multi-factor authentication implemented for all users
            - Role-based access control with least privilege principles
            - Quarterly access reviews conducted with documented results
            - Automated user provisioning and deprovisioning
            - Privileged access management system deployed
            - Access control matrix maintained and updated
            """,
            expected_compliance="Compliant",
            expected_risk_level="Low",
            expected_findings=["Multi-factor authentication implemented", "Regular access reviews conducted", "Automated provisioning in place"],
            framework="NIST",
            description="Well-implemented access control with strong evidence"
        ),
        TestCase(
            test_id="encryption_non_compliant",
            control=ControlData(
                control_id="PR.DS-1",
                control_name="Data Protection",
                requirement="Data-at-rest is protected",
                answer="We encrypt some sensitive data using various methods."
            ),
            evidence_text="""
            Data Protection Assessment:
            - Some databases use basic encryption
            - File shares not encrypted
            - Backup systems use weak encryption (DES)
            - No centralized key management
            - Encryption policies outdated (2019)
            """,
            expected_compliance="Non-Compliant",
            expected_risk_level="High",
            expected_findings=["Weak encryption methods", "No centralized key management", "Outdated policies"],
            framework="NIST",
            description="Poor data protection implementation with significant gaps"
        )
    ]
    
    return test_cases

def main():
    """Main function to run AI accuracy tests"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Create tester
        logger.info("Initializing AI accuracy tester...")
        tester = AIAccuracyTester()

        # Create test cases
        logger.info("Creating test cases...")
        test_cases = create_sample_test_cases()
        logger.info(f"Created {len(test_cases)} test cases")

        # Run tests
        logger.info("Running accuracy tests...")
        results = tester.run_accuracy_tests(test_cases)

        # Print report
        print(results["report"])

    except Exception as e:
        logger.error(f"Failed to initialize or run tests: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        print(f"ERROR: Test execution failed - {e}")
        return
    
    # Save detailed results
    with open("ai_accuracy_test_results.json", "w") as f:
        # Convert results to JSON-serializable format safely
        serializable_results = []
        for result in results["results"]:
            try:
                # Safely convert the main result to dict
                result_dict = safe_dataclass_to_dict(result)

                # Safely handle ai_result field
                if "ai_result" in result_dict:
                    result_dict["ai_result"] = safe_dataclass_to_dict(result_dict["ai_result"])

                serializable_results.append(result_dict)

            except Exception as e:
                logger.error(f"Failed to serialize test result: {e}")
                # Add a minimal error result
                serializable_results.append({
                    "test_id": getattr(result, 'test_id', 'unknown'),
                    "passed": False,
                    "error": f"Serialization failed: {str(e)}",
                    "ai_result": None
                })

        try:
            json.dump({
                "results": serializable_results,
                "metrics": results["metrics"]
            }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save results to JSON: {e}")
            # Save a minimal error report
            json.dump({
                "error": f"Failed to save full results: {str(e)}",
                "results_count": len(serializable_results),
                "metrics": results.get("metrics", {})
            }, f, indent=2)
    
    print(f"\nDetailed results saved to ai_accuracy_test_results.json")
    print(f"Overall pass rate: {results['metrics']['pass_rate']:.1%}")

if __name__ == "__main__":
    main()
