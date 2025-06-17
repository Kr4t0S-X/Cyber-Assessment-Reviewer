"""
Automated Learning Pipeline for Continuous AI Improvement
Processes feedback and automatically improves AI components
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

from feedback_system import FeedbackCollector, FeedbackEntry, FeedbackType, FeedbackSeverity

logger = logging.getLogger(__name__)

@dataclass
class ImprovementAction:
    """Represents an improvement action to be taken"""
    action_id: str
    action_type: str  # 'prompt_update', 'knowledge_update', 'validation_rule', 'threshold_adjustment'
    priority: int  # 1-5, 5 being highest
    description: str
    target_component: str
    proposed_changes: Dict[str, Any]
    evidence: List[str]  # Feedback IDs that support this action
    confidence: float  # 0.0-1.0
    estimated_impact: str  # 'low', 'medium', 'high'

@dataclass
class LearningInsight:
    """Insights derived from feedback analysis"""
    insight_id: str
    insight_type: str
    description: str
    affected_controls: List[str]
    affected_frameworks: List[str]
    frequency: int
    confidence: float
    supporting_evidence: List[str]

class FeedbackAnalyzer:
    """Analyzes feedback patterns to identify improvement opportunities"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        
    def analyze_feedback_patterns(self, days: int = 30) -> List[LearningInsight]:
        """Analyze feedback to identify patterns and insights"""
        feedback_entries = self.feedback_collector.get_recent_feedback(days)
        insights = []
        
        # Analyze compliance status patterns
        insights.extend(self._analyze_compliance_patterns(feedback_entries))
        
        # Analyze risk level patterns
        insights.extend(self._analyze_risk_patterns(feedback_entries))
        
        # Analyze validation issue patterns
        insights.extend(self._analyze_validation_patterns(feedback_entries))
        
        # Analyze framework-specific patterns
        insights.extend(self._analyze_framework_patterns(feedback_entries))
        
        return insights
    
    def _analyze_compliance_patterns(self, feedback_entries: List[FeedbackEntry]) -> List[LearningInsight]:
        """Analyze patterns in compliance status corrections"""
        insights = []
        
        # Group by compliance status changes
        compliance_changes = defaultdict(list)
        
        for entry in feedback_entries:
            if (entry.corrected_compliance_status and 
                entry.original_compliance_status != entry.corrected_compliance_status):
                
                change_key = f"{entry.original_compliance_status} -> {entry.corrected_compliance_status}"
                compliance_changes[change_key].append(entry)
        
        # Identify significant patterns
        for change_pattern, entries in compliance_changes.items():
            if len(entries) >= 3:  # Minimum threshold for pattern
                frameworks = [e.framework for e in entries]
                controls = [e.control_id for e in entries]
                
                insight = LearningInsight(
                    insight_id=f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="compliance_pattern",
                    description=f"Frequent compliance status corrections: {change_pattern}",
                    affected_controls=list(set(controls)),
                    affected_frameworks=list(set(frameworks)),
                    frequency=len(entries),
                    confidence=min(1.0, len(entries) / 10.0),
                    supporting_evidence=[e.feedback_id for e in entries]
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_risk_patterns(self, feedback_entries: List[FeedbackEntry]) -> List[LearningInsight]:
        """Analyze patterns in risk level corrections"""
        insights = []
        
        risk_changes = defaultdict(list)
        
        for entry in feedback_entries:
            if (entry.corrected_risk_level and 
                entry.original_risk_level != entry.corrected_risk_level):
                
                change_key = f"{entry.original_risk_level} -> {entry.corrected_risk_level}"
                risk_changes[change_key].append(entry)
        
        for change_pattern, entries in risk_changes.items():
            if len(entries) >= 3:
                insight = LearningInsight(
                    insight_id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="risk_pattern",
                    description=f"Frequent risk level corrections: {change_pattern}",
                    affected_controls=[e.control_id for e in entries],
                    affected_frameworks=list(set([e.framework for e in entries])),
                    frequency=len(entries),
                    confidence=min(1.0, len(entries) / 8.0),
                    supporting_evidence=[e.feedback_id for e in entries]
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_validation_patterns(self, feedback_entries: List[FeedbackEntry]) -> List[LearningInsight]:
        """Analyze patterns in validation issues"""
        insights = []
        
        # Collect all validation issues
        validation_issues = []
        for entry in feedback_entries:
            if entry.validation_issues:
                validation_issues.extend(entry.validation_issues)
        
        # Count issue frequency
        issue_counts = Counter(validation_issues)
        
        for issue, count in issue_counts.items():
            if count >= 5:  # Minimum threshold for validation pattern
                insight = LearningInsight(
                    insight_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    insight_type="validation_pattern",
                    description=f"Frequent validation issue: {issue}",
                    affected_controls=[],
                    affected_frameworks=[],
                    frequency=count,
                    confidence=min(1.0, count / 20.0),
                    supporting_evidence=[]
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_framework_patterns(self, feedback_entries: List[FeedbackEntry]) -> List[LearningInsight]:
        """Analyze framework-specific patterns"""
        insights = []
        
        # Group feedback by framework
        framework_feedback = defaultdict(list)
        for entry in feedback_entries:
            framework_feedback[entry.framework].append(entry)
        
        for framework, entries in framework_feedback.items():
            if len(entries) >= 5:  # Minimum threshold
                # Calculate error rate
                error_entries = [e for e in entries if e.severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH]]
                error_rate = len(error_entries) / len(entries)
                
                if error_rate > 0.3:  # High error rate threshold
                    insight = LearningInsight(
                        insight_id=f"framework_{framework}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        insight_type="framework_pattern",
                        description=f"High error rate in {framework} framework: {error_rate:.1%}",
                        affected_controls=[e.control_id for e in error_entries],
                        affected_frameworks=[framework],
                        frequency=len(error_entries),
                        confidence=error_rate,
                        supporting_evidence=[e.feedback_id for e in error_entries]
                    )
                    insights.append(insight)
        
        return insights

class ImprovementEngine:
    """Generates improvement actions based on feedback analysis"""
    
    def __init__(self):
        self.improvement_history = []
    
    def generate_improvements(self, insights: List[LearningInsight]) -> List[ImprovementAction]:
        """Generate improvement actions from insights"""
        improvements = []
        
        for insight in insights:
            if insight.insight_type == "compliance_pattern":
                improvements.extend(self._generate_compliance_improvements(insight))
            elif insight.insight_type == "risk_pattern":
                improvements.extend(self._generate_risk_improvements(insight))
            elif insight.insight_type == "validation_pattern":
                improvements.extend(self._generate_validation_improvements(insight))
            elif insight.insight_type == "framework_pattern":
                improvements.extend(self._generate_framework_improvements(insight))
        
        # Prioritize improvements
        improvements.sort(key=lambda x: (x.priority, x.confidence), reverse=True)
        
        return improvements
    
    def _generate_compliance_improvements(self, insight: LearningInsight) -> List[ImprovementAction]:
        """Generate improvements for compliance patterns"""
        improvements = []
        
        # Extract the compliance change pattern
        if " -> " in insight.description:
            pattern = insight.description.split(": ")[1]
            original, corrected = pattern.split(" -> ")
            
            # Generate prompt improvement
            improvement = ImprovementAction(
                action_id=f"prompt_compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                action_type="prompt_update",
                priority=4 if insight.confidence > 0.7 else 3,
                description=f"Update prompts to better distinguish {original} vs {corrected} compliance",
                target_component="ai_backend.py",
                proposed_changes={
                    "compliance_guidance": f"Enhanced guidance for {original} vs {corrected} determination",
                    "evidence_requirements": f"Specific evidence requirements for {corrected} status"
                },
                evidence=insight.supporting_evidence,
                confidence=insight.confidence,
                estimated_impact="high" if insight.frequency > 5 else "medium"
            )
            improvements.append(improvement)
        
        return improvements
    
    def _generate_risk_improvements(self, insight: LearningInsight) -> List[ImprovementAction]:
        """Generate improvements for risk patterns"""
        improvements = []
        
        if " -> " in insight.description:
            pattern = insight.description.split(": ")[1]
            original, corrected = pattern.split(" -> ")
            
            improvement = ImprovementAction(
                action_id=f"risk_criteria_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                action_type="knowledge_update",
                priority=4,
                description=f"Update risk assessment criteria for {original} to {corrected} transitions",
                target_component="cybersecurity_knowledge.py",
                proposed_changes={
                    "risk_criteria_update": f"Refined criteria for {corrected} risk level",
                    "risk_indicators": f"Updated indicators that distinguish {original} from {corrected}"
                },
                evidence=insight.supporting_evidence,
                confidence=insight.confidence,
                estimated_impact="high"
            )
            improvements.append(improvement)
        
        return improvements
    
    def _generate_validation_improvements(self, insight: LearningInsight) -> List[ImprovementAction]:
        """Generate improvements for validation patterns"""
        improvements = []
        
        issue = insight.description.split(": ")[1]
        
        improvement = ImprovementAction(
            action_id=f"validation_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            action_type="validation_rule",
            priority=3,
            description=f"Update validation rules to address: {issue}",
            target_component="ai_validation.py",
            proposed_changes={
                "validation_rule": f"Enhanced validation for: {issue}",
                "threshold_adjustment": "Adjusted validation thresholds"
            },
            evidence=insight.supporting_evidence,
            confidence=insight.confidence,
            estimated_impact="medium"
        )
        improvements.append(improvement)
        
        return improvements
    
    def _generate_framework_improvements(self, insight: LearningInsight) -> List[ImprovementAction]:
        """Generate improvements for framework patterns"""
        improvements = []
        
        framework = insight.affected_frameworks[0] if insight.affected_frameworks else "Unknown"
        
        improvement = ImprovementAction(
            action_id=f"framework_{framework}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            action_type="prompt_update",
            priority=5,  # High priority for framework-specific issues
            description=f"Enhance {framework} framework-specific guidance",
            target_component="ai_backend.py",
            proposed_changes={
                "framework_guidance": f"Enhanced {framework} specific guidance",
                "control_mappings": f"Updated {framework} control mappings"
            },
            evidence=insight.supporting_evidence,
            confidence=insight.confidence,
            estimated_impact="high"
        )
        improvements.append(improvement)
        
        return improvements

class AutomatedLearningPipeline:
    """Main pipeline for automated learning and improvement"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.analyzer = FeedbackAnalyzer(feedback_collector)
        self.improvement_engine = ImprovementEngine()
        
        # Pipeline configuration
        self.analysis_interval_days = 7
        self.min_feedback_threshold = 5
        self.auto_apply_threshold = 0.8  # Confidence threshold for auto-applying improvements
        
        logger.info("Automated learning pipeline initialized")
    
    def run_learning_cycle(self, days: int = None) -> Dict[str, Any]:
        """Run a complete learning cycle"""
        days = days or self.analysis_interval_days
        
        logger.info(f"Starting learning cycle for last {days} days")
        
        # Step 1: Analyze feedback patterns
        insights = self.analyzer.analyze_feedback_patterns(days)
        logger.info(f"Identified {len(insights)} insights")
        
        # Step 2: Generate improvement actions
        improvements = self.improvement_engine.generate_improvements(insights)
        logger.info(f"Generated {len(improvements)} improvement actions")
        
        # Step 3: Apply high-confidence improvements automatically
        auto_applied = self._auto_apply_improvements(improvements)
        
        # Step 4: Generate report
        report = {
            'cycle_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'insights_found': len(insights),
            'improvements_generated': len(improvements),
            'auto_applied_improvements': len(auto_applied),
            'insights': [self._insight_to_dict(i) for i in insights],
            'improvements': [self._improvement_to_dict(i) for i in improvements],
            'auto_applied': [self._improvement_to_dict(i) for i in auto_applied]
        }
        
        # Save report
        self._save_learning_report(report)
        
        logger.info(f"Learning cycle complete. Auto-applied {len(auto_applied)} improvements")
        return report
    
    def _auto_apply_improvements(self, improvements: List[ImprovementAction]) -> List[ImprovementAction]:
        """Automatically apply high-confidence improvements"""
        auto_applied = []
        
        for improvement in improvements:
            if (improvement.confidence >= self.auto_apply_threshold and 
                improvement.priority >= 4):
                
                try:
                    if self._apply_improvement(improvement):
                        auto_applied.append(improvement)
                        logger.info(f"Auto-applied improvement: {improvement.action_id}")
                except Exception as e:
                    logger.error(f"Failed to auto-apply improvement {improvement.action_id}: {e}")
        
        return auto_applied
    
    def _apply_improvement(self, improvement: ImprovementAction) -> bool:
        """Apply a specific improvement action"""
        # This is a placeholder for actual improvement application
        # In practice, this would modify prompts, update knowledge base, etc.
        
        logger.info(f"Applying improvement: {improvement.description}")
        
        # For now, just log the improvement
        # TODO: Implement actual improvement application logic
        
        return True
    
    def _insight_to_dict(self, insight: LearningInsight) -> Dict[str, Any]:
        """Convert insight to dictionary"""
        return {
            'insight_id': insight.insight_id,
            'type': insight.insight_type,
            'description': insight.description,
            'frequency': insight.frequency,
            'confidence': insight.confidence,
            'affected_controls': insight.affected_controls[:5],  # Limit for readability
            'affected_frameworks': insight.affected_frameworks
        }
    
    def _improvement_to_dict(self, improvement: ImprovementAction) -> Dict[str, Any]:
        """Convert improvement to dictionary"""
        return {
            'action_id': improvement.action_id,
            'type': improvement.action_type,
            'priority': improvement.priority,
            'description': improvement.description,
            'target': improvement.target_component,
            'confidence': improvement.confidence,
            'impact': improvement.estimated_impact,
            'evidence_count': len(improvement.evidence)
        }
    
    def _save_learning_report(self, report: Dict[str, Any]):
        """Save learning cycle report"""
        reports_dir = Path("learning_reports")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"learning_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Learning report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save learning report: {e}")
