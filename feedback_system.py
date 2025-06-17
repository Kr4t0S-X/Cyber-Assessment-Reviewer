"""
Feedback Collection Framework for Continuous AI Improvement
Captures and stores feedback from validation results, user corrections, and expert reviews
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback sources"""
    VALIDATION_RESULT = "validation_result"
    USER_CORRECTION = "user_correction"
    EXPERT_REVIEW = "expert_review"
    TEST_CASE_OUTCOME = "test_case_outcome"
    AUTOMATED_ANALYSIS = "automated_analysis"

class FeedbackSeverity(Enum):
    """Severity levels for feedback"""
    CRITICAL = "critical"      # Major accuracy issues
    HIGH = "high"             # Significant improvement needed
    MEDIUM = "medium"         # Moderate issues
    LOW = "low"              # Minor improvements
    INFO = "info"            # Informational feedback

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    feedback_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    severity: FeedbackSeverity
    control_id: str
    framework: str
    
    # Original AI response details
    original_compliance_status: str
    original_risk_level: str
    original_confidence_score: float
    original_findings: List[str]
    
    # Feedback details
    corrected_compliance_status: Optional[str] = None
    corrected_risk_level: Optional[str] = None
    corrected_findings: Optional[List[str]] = None
    feedback_comments: str = ""
    
    # Context information
    evidence_quality: Optional[float] = None
    validation_issues: Optional[List[str]] = None
    improvement_suggestions: Optional[List[str]] = None
    
    # Metadata
    user_id: Optional[str] = None
    expert_credentials: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['feedback_type'] = self.feedback_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackEntry':
        """Create instance from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['feedback_type'] = FeedbackType(data['feedback_type'])
        data['severity'] = FeedbackSeverity(data['severity'])
        return cls(**data)

@dataclass
class FeedbackPattern:
    """Identified pattern in feedback data"""
    pattern_id: str
    pattern_type: str  # e.g., "compliance_overestimation", "risk_underestimation"
    frequency: int
    confidence: float
    affected_controls: List[str]
    affected_frameworks: List[str]
    description: str
    suggested_improvements: List[str]
    first_seen: datetime
    last_seen: datetime

class FeedbackCollector:
    """Collects and manages feedback from various sources"""
    
    def __init__(self, feedback_dir: str = "feedback_data"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(exist_ok=True)
        
        # Initialize feedback storage files
        self.feedback_file = self.feedback_dir / "feedback_entries.jsonl"
        self.patterns_file = self.feedback_dir / "feedback_patterns.json"
        self.metrics_file = self.feedback_dir / "feedback_metrics.json"
        
        logger.info(f"Feedback collector initialized with directory: {self.feedback_dir}")
    
    def collect_validation_feedback(self, control_id: str, framework: str,
                                  original_result: Dict[str, Any],
                                  validation_result: Dict[str, Any]) -> str:
        """Collect feedback from AI validation results"""
        
        feedback_id = f"val_{control_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine severity based on validation quality
        quality_score = validation_result.get('quality_score', 1.0)
        if quality_score < 0.4:
            severity = FeedbackSeverity.CRITICAL
        elif quality_score < 0.6:
            severity = FeedbackSeverity.HIGH
        elif quality_score < 0.8:
            severity = FeedbackSeverity.MEDIUM
        else:
            severity = FeedbackSeverity.LOW
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.VALIDATION_RESULT,
            severity=severity,
            control_id=control_id,
            framework=framework,
            original_compliance_status=original_result.get('compliance_status', ''),
            original_risk_level=original_result.get('risk_level', ''),
            original_confidence_score=original_result.get('confidence_score', 0.0),
            original_findings=original_result.get('key_findings', []),
            validation_issues=validation_result.get('validation_issues', []),
            improvement_suggestions=validation_result.get('recommendations', []),
            evidence_quality=validation_result.get('quality_score', 0.0),
            feedback_comments=f"Validation quality score: {quality_score:.2f}"
        )
        
        self._store_feedback(feedback)
        logger.info(f"Collected validation feedback for {control_id}: {severity.value}")
        return feedback_id
    
    def collect_user_correction(self, control_id: str, framework: str,
                              original_result: Dict[str, Any],
                              user_corrections: Dict[str, Any],
                              user_id: str = None,
                              comments: str = "") -> str:
        """Collect feedback from user corrections"""
        
        feedback_id = f"user_{control_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine severity based on magnitude of corrections
        severity = self._assess_correction_severity(original_result, user_corrections)
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.USER_CORRECTION,
            severity=severity,
            control_id=control_id,
            framework=framework,
            original_compliance_status=original_result.get('compliance_status', ''),
            original_risk_level=original_result.get('risk_level', ''),
            original_confidence_score=original_result.get('confidence_score', 0.0),
            original_findings=original_result.get('key_findings', []),
            corrected_compliance_status=user_corrections.get('compliance_status'),
            corrected_risk_level=user_corrections.get('risk_level'),
            corrected_findings=user_corrections.get('key_findings'),
            feedback_comments=comments,
            user_id=user_id
        )
        
        self._store_feedback(feedback)
        logger.info(f"Collected user correction for {control_id}: {severity.value}")
        return feedback_id
    
    def collect_expert_review(self, control_id: str, framework: str,
                            original_result: Dict[str, Any],
                            expert_assessment: Dict[str, Any],
                            expert_credentials: str,
                            review_comments: str = "") -> str:
        """Collect feedback from expert cybersecurity professional reviews"""
        
        feedback_id = f"expert_{control_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Expert feedback is typically high priority
        severity = FeedbackSeverity.HIGH
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.EXPERT_REVIEW,
            severity=severity,
            control_id=control_id,
            framework=framework,
            original_compliance_status=original_result.get('compliance_status', ''),
            original_risk_level=original_result.get('risk_level', ''),
            original_confidence_score=original_result.get('confidence_score', 0.0),
            original_findings=original_result.get('key_findings', []),
            corrected_compliance_status=expert_assessment.get('compliance_status'),
            corrected_risk_level=expert_assessment.get('risk_level'),
            corrected_findings=expert_assessment.get('key_findings'),
            feedback_comments=review_comments,
            expert_credentials=expert_credentials,
            improvement_suggestions=expert_assessment.get('improvement_suggestions', [])
        )
        
        self._store_feedback(feedback)
        logger.info(f"Collected expert review for {control_id}: {expert_credentials}")
        return feedback_id
    
    def collect_test_case_feedback(self, test_id: str, control_id: str,
                                 framework: str, test_result: Dict[str, Any]) -> str:
        """Collect feedback from test case outcomes"""
        
        feedback_id = f"test_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine severity based on test performance
        accuracy_score = test_result.get('accuracy_score', 0.0)
        if accuracy_score < 0.3:
            severity = FeedbackSeverity.CRITICAL
        elif accuracy_score < 0.6:
            severity = FeedbackSeverity.HIGH
        elif accuracy_score < 0.8:
            severity = FeedbackSeverity.MEDIUM
        else:
            severity = FeedbackSeverity.LOW
        
        ai_result = test_result.get('ai_result', {})
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=datetime.now(),
            feedback_type=FeedbackType.TEST_CASE_OUTCOME,
            severity=severity,
            control_id=control_id,
            framework=framework,
            original_compliance_status=ai_result.get('compliance_status', ''),
            original_risk_level=ai_result.get('risk_level', ''),
            original_confidence_score=ai_result.get('confidence_score', 0.0),
            original_findings=ai_result.get('key_findings', []),
            validation_issues=test_result.get('issues', []),
            feedback_comments=f"Test accuracy: {accuracy_score:.2f}, Passed: {test_result.get('passed', False)}"
        )
        
        self._store_feedback(feedback)
        logger.info(f"Collected test case feedback for {control_id}: {severity.value}")
        return feedback_id
    
    def _assess_correction_severity(self, original: Dict[str, Any], 
                                  corrected: Dict[str, Any]) -> FeedbackSeverity:
        """Assess severity of user corrections"""
        severity_score = 0
        
        # Check compliance status change
        if (original.get('compliance_status') != corrected.get('compliance_status') and
            corrected.get('compliance_status') is not None):
            severity_score += 3
        
        # Check risk level change
        risk_levels = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        orig_risk = risk_levels.get(original.get('risk_level', 'Medium'), 2)
        corr_risk = risk_levels.get(corrected.get('risk_level', 'Medium'), 2)
        risk_diff = abs(orig_risk - corr_risk)
        severity_score += risk_diff
        
        # Check findings changes
        if corrected.get('key_findings') and len(corrected.get('key_findings', [])) > 0:
            severity_score += 1
        
        # Map to severity levels
        if severity_score >= 5:
            return FeedbackSeverity.CRITICAL
        elif severity_score >= 3:
            return FeedbackSeverity.HIGH
        elif severity_score >= 1:
            return FeedbackSeverity.MEDIUM
        else:
            return FeedbackSeverity.LOW
    
    def _store_feedback(self, feedback: FeedbackEntry):
        """Store feedback entry to persistent storage"""
        try:
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
    
    def get_recent_feedback(self, days: int = 7,
                          feedback_type: Optional[FeedbackType] = None) -> List[FeedbackEntry]:
        """Get recent feedback entries"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)
        
        feedback_entries = []
        
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            entry = FeedbackEntry.from_dict(data)
                            
                            if entry.timestamp >= cutoff_date:
                                if feedback_type is None or entry.feedback_type == feedback_type:
                                    feedback_entries.append(entry)
                        except Exception as e:
                            logger.warning(f"Failed to parse feedback entry: {e}")
        except Exception as e:
            logger.error(f"Failed to read feedback file: {e}")
        
        return sorted(feedback_entries, key=lambda x: x.timestamp, reverse=True)
    
    def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of feedback over specified period"""
        feedback_entries = self.get_recent_feedback(days)
        
        summary = {
            'total_feedback': len(feedback_entries),
            'by_type': {},
            'by_severity': {},
            'by_framework': {},
            'common_issues': [],
            'improvement_areas': []
        }
        
        for entry in feedback_entries:
            # Count by type
            type_key = entry.feedback_type.value
            summary['by_type'][type_key] = summary['by_type'].get(type_key, 0) + 1
            
            # Count by severity
            severity_key = entry.severity.value
            summary['by_severity'][severity_key] = summary['by_severity'].get(severity_key, 0) + 1
            
            # Count by framework
            framework_key = entry.framework
            summary['by_framework'][framework_key] = summary['by_framework'].get(framework_key, 0) + 1
        
        return summary
