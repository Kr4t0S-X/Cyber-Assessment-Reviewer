#!/usr/bin/env python3
"""
Feedback Integration System
Main integration script for the iterative feedback loop and continuous improvement system
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from feedback_system import FeedbackCollector, FeedbackType, FeedbackSeverity
from learning_pipeline import AutomatedLearningPipeline
from adaptive_prompts import AdaptivePromptManager
from continuous_improvement import ContinuousImprovementWorkflow
from performance_analytics import PerformanceAnalytics
from ai_backend import CyberAssessmentReviewer
from config import Config

logger = logging.getLogger(__name__)

class FeedbackIntegrationManager:
    """Main manager for the feedback integration system"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize core components
        self.feedback_collector = FeedbackCollector()
        self.learning_pipeline = AutomatedLearningPipeline(self.feedback_collector)
        self.adaptive_prompts = AdaptivePromptManager()
        self.continuous_improvement = ContinuousImprovementWorkflow(self.feedback_collector)
        self.analytics = PerformanceAnalytics(self.feedback_collector)
        
        # Initialize AI reviewer with feedback integration
        self.ai_reviewer = CyberAssessmentReviewer(self.config)
        
        logger.info("Feedback integration system initialized")
    
    def start_continuous_improvement(self):
        """Start the continuous improvement monitoring"""
        logger.info("Starting continuous improvement monitoring...")
        self.continuous_improvement.start_monitoring()
        logger.info("Continuous improvement monitoring started")
    
    def stop_continuous_improvement(self):
        """Stop the continuous improvement monitoring"""
        logger.info("Stopping continuous improvement monitoring...")
        self.continuous_improvement.stop_monitoring()
        logger.info("Continuous improvement monitoring stopped")
    
    def collect_user_feedback(self, control_id: str, framework: str,
                            original_result: Dict[str, Any],
                            user_corrections: Dict[str, Any],
                            user_id: str = None,
                            comments: str = "") -> str:
        """Collect user feedback on AI assessment results"""
        
        logger.info(f"Collecting user feedback for control {control_id}")
        
        feedback_id = self.feedback_collector.collect_user_correction(
            control_id, framework, original_result, user_corrections, user_id, comments
        )
        
        # Trigger immediate learning if high-severity feedback
        severity = self._assess_feedback_severity(original_result, user_corrections)
        if severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH]:
            logger.info("High-severity feedback detected, triggering immediate learning cycle")
            self._trigger_immediate_learning()
        
        return feedback_id
    
    def collect_expert_feedback(self, control_id: str, framework: str,
                              original_result: Dict[str, Any],
                              expert_assessment: Dict[str, Any],
                              expert_credentials: str,
                              review_comments: str = "") -> str:
        """Collect expert feedback on AI assessment results"""
        
        logger.info(f"Collecting expert feedback for control {control_id}")
        
        feedback_id = self.feedback_collector.collect_expert_review(
            control_id, framework, original_result, expert_assessment,
            expert_credentials, review_comments
        )
        
        # Expert feedback always triggers learning
        logger.info("Expert feedback received, triggering immediate learning cycle")
        self._trigger_immediate_learning()
        
        return feedback_id
    
    def run_manual_improvement_cycle(self, reason: str = "Manual trigger") -> Dict[str, Any]:
        """Manually run an improvement cycle"""
        
        logger.info(f"Running manual improvement cycle: {reason}")
        
        # Run learning pipeline
        learning_report = self.learning_pipeline.run_learning_cycle()
        
        # Apply prompt adaptations if insights found
        if learning_report['insights']:
            insights = [self._dict_to_insight(i) for i in learning_report['insights']]
            improvements = [self._dict_to_improvement(i) for i in learning_report['improvements']]
            
            adaptations = self.adaptive_prompts.adapt_prompt_from_feedback(insights, improvements)
            learning_report['prompt_adaptations'] = len(adaptations)
        
        # Force improvement cycle in continuous improvement workflow
        cycle_id = self.continuous_improvement.force_improvement_cycle(reason)
        learning_report['improvement_cycle_id'] = cycle_id
        
        logger.info(f"Manual improvement cycle completed: {learning_report}")
        return learning_report
    
    def generate_analytics_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        logger.info(f"Generating analytics report for {days} days")
        
        report = self.analytics.generate_comprehensive_report(days)
        
        # Convert to dictionary for JSON serialization
        report_dict = {
            'report_id': report.report_id,
            'generated_at': report.generated_at.isoformat(),
            'period_start': report.period_start.isoformat(),
            'period_end': report.period_end.isoformat(),
            'summary_metrics': report.summary_metrics,
            'trends': [
                {
                    'metric_name': t.metric_name,
                    'trend_direction': t.trend_direction,
                    'trend_strength': t.trend_strength,
                    'current_value': t.current_value,
                    'change_percentage': t.change_percentage
                } for t in report.trends
            ],
            'insights': report.insights,
            'recommendations': report.recommendations,
            'improvement_impact': report.improvement_impact
        }
        
        logger.info(f"Analytics report generated: {report.report_id}")
        return report_dict
    
    def get_feedback_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary of recent feedback"""
        
        summary = self.feedback_collector.get_feedback_summary(days)
        
        # Add improvement summary
        improvement_summary = self.continuous_improvement.get_improvement_summary(days)
        summary['improvement_summary'] = improvement_summary
        
        # Add analytics dashboard data
        dashboard_data = self.analytics.get_dashboard_data()
        summary['analytics'] = dashboard_data
        
        return summary
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        
        status = {
            'feedback_system': {
                'active': True,
                'total_feedback_entries': len(self.feedback_collector.get_recent_feedback(365)),
                'recent_feedback_count': len(self.feedback_collector.get_recent_feedback(7))
            },
            'continuous_improvement': {
                'monitoring_active': self.continuous_improvement.is_running,
                'total_cycles': len(self.continuous_improvement.improvement_cycles),
                'recent_cycles': len([c for c in self.continuous_improvement.improvement_cycles 
                                    if (datetime.now() - c.start_time).days <= 30])
            },
            'adaptive_prompts': {
                'total_templates': len(self.adaptive_prompts.templates),
                'total_adaptations': len(self.adaptive_prompts.adaptations),
                'last_adaptation': (self.adaptive_prompts.adaptations[-1].applied_date.isoformat() 
                                  if self.adaptive_prompts.adaptations else None)
            },
            'analytics': {
                'total_reports': len(self.analytics.reports_history),
                'last_report': (self.analytics.reports_history[-1]['generated_at'] 
                              if self.analytics.reports_history else None)
            }
        }
        
        return status
    
    def _assess_feedback_severity(self, original: Dict[str, Any], 
                                corrected: Dict[str, Any]) -> FeedbackSeverity:
        """Assess severity of feedback"""
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
        
        # Map to severity levels
        if severity_score >= 5:
            return FeedbackSeverity.CRITICAL
        elif severity_score >= 3:
            return FeedbackSeverity.HIGH
        elif severity_score >= 1:
            return FeedbackSeverity.MEDIUM
        else:
            return FeedbackSeverity.LOW
    
    def _trigger_immediate_learning(self):
        """Trigger immediate learning cycle for high-priority feedback"""
        try:
            # Run a quick learning cycle
            learning_report = self.learning_pipeline.run_learning_cycle(days=3)  # Short period for immediate response
            
            if learning_report['insights_found'] > 0:
                logger.info(f"Immediate learning triggered: {learning_report['insights_found']} insights, "
                          f"{learning_report['auto_applied_improvements']} improvements applied")
        except Exception as e:
            logger.error(f"Failed to trigger immediate learning: {e}")
    
    def _dict_to_insight(self, insight_dict: Dict[str, Any]):
        """Convert dictionary to LearningInsight object"""
        from learning_pipeline import LearningInsight
        return LearningInsight(
            insight_id=insight_dict['insight_id'],
            insight_type=insight_dict['type'],
            description=insight_dict['description'],
            affected_controls=insight_dict['affected_controls'],
            affected_frameworks=insight_dict['affected_frameworks'],
            frequency=insight_dict['frequency'],
            confidence=insight_dict['confidence'],
            supporting_evidence=[]
        )
    
    def _dict_to_improvement(self, improvement_dict: Dict[str, Any]):
        """Convert dictionary to ImprovementAction object"""
        from learning_pipeline import ImprovementAction
        return ImprovementAction(
            action_id=improvement_dict['action_id'],
            action_type=improvement_dict['type'],
            priority=improvement_dict['priority'],
            description=improvement_dict['description'],
            target_component=improvement_dict['target'],
            proposed_changes={},
            evidence=[],
            confidence=improvement_dict['confidence'],
            estimated_impact=improvement_dict['impact']
        )

def main():
    """Main function for testing the feedback integration system"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the feedback integration system
    feedback_manager = FeedbackIntegrationManager()
    
    # Start continuous improvement monitoring
    feedback_manager.start_continuous_improvement()
    
    # Generate initial analytics report
    report = feedback_manager.generate_analytics_report(30)
    print(f"Generated analytics report: {report['report_id']}")
    
    # Get system status
    status = feedback_manager.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
    
    # Get feedback summary
    summary = feedback_manager.get_feedback_summary(7)
    print(f"Feedback summary: {json.dumps(summary, indent=2)}")
    
    print("Feedback integration system is running. Press Ctrl+C to stop.")
    
    try:
        import time
        while True:
            time.sleep(60)  # Keep running
    except KeyboardInterrupt:
        print("Stopping feedback integration system...")
        feedback_manager.stop_continuous_improvement()
        print("Feedback integration system stopped.")

if __name__ == "__main__":
    main()
