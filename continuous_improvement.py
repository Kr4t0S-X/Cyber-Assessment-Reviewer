"""
Continuous Improvement Workflow
Monitors AI response quality and implements systematic improvements
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading

from feedback_system import FeedbackCollector, FeedbackType, FeedbackSeverity
from learning_pipeline import AutomatedLearningPipeline, FeedbackAnalyzer, ImprovementEngine
from adaptive_prompts import AdaptivePromptManager
from ai_validation import AIResponseValidator

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for monitoring AI performance"""
    timestamp: datetime
    period_days: int
    total_assessments: int
    average_quality_score: float
    average_confidence_score: float
    compliance_accuracy: float
    risk_accuracy: float
    validation_pass_rate: float
    user_satisfaction_score: float
    improvement_trend: str  # 'improving', 'stable', 'declining'

@dataclass
class ImprovementCycle:
    """Represents a complete improvement cycle"""
    cycle_id: str
    start_time: datetime
    end_time: Optional[datetime]
    trigger_reason: str
    metrics_before: QualityMetrics
    metrics_after: Optional[QualityMetrics]
    insights_found: int
    improvements_applied: int
    success_rate: float
    status: str  # 'running', 'completed', 'failed'

class QualityMonitor:
    """Monitors AI response quality and triggers improvements"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.metrics_history = []
        self.quality_thresholds = {
            'min_quality_score': 0.6,
            'min_confidence_score': 0.5,
            'min_compliance_accuracy': 0.7,
            'min_validation_pass_rate': 0.8,
            'decline_threshold': 0.1  # Trigger if metrics decline by this amount
        }
        
    def calculate_current_metrics(self, days: int = 7) -> QualityMetrics:
        """Calculate current quality metrics"""
        
        feedback_entries = self.feedback_collector.get_recent_feedback(days)
        
        if not feedback_entries:
            return QualityMetrics(
                timestamp=datetime.now(),
                period_days=days,
                total_assessments=0,
                average_quality_score=0.0,
                average_confidence_score=0.0,
                compliance_accuracy=0.0,
                risk_accuracy=0.0,
                validation_pass_rate=0.0,
                user_satisfaction_score=0.0,
                improvement_trend='stable'
            )
        
        # Calculate metrics
        total_assessments = len(feedback_entries)
        
        # Quality scores from validation feedback
        validation_entries = [e for e in feedback_entries if e.feedback_type == FeedbackType.VALIDATION_RESULT]
        avg_quality = sum(e.evidence_quality or 0.5 for e in validation_entries) / max(1, len(validation_entries))
        
        # Confidence scores
        avg_confidence = sum(e.original_confidence_score for e in feedback_entries) / total_assessments
        
        # Compliance accuracy (from user corrections and expert reviews)
        correction_entries = [e for e in feedback_entries if e.corrected_compliance_status]
        compliance_matches = sum(1 for e in correction_entries 
                               if e.original_compliance_status == e.corrected_compliance_status)
        compliance_accuracy = compliance_matches / max(1, len(correction_entries))
        
        # Risk accuracy
        risk_correction_entries = [e for e in feedback_entries if e.corrected_risk_level]
        risk_matches = sum(1 for e in risk_correction_entries 
                          if e.original_risk_level == e.corrected_risk_level)
        risk_accuracy = risk_matches / max(1, len(risk_correction_entries))
        
        # Validation pass rate
        high_severity_count = sum(1 for e in feedback_entries 
                                if e.severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH])
        validation_pass_rate = 1.0 - (high_severity_count / total_assessments)
        
        # User satisfaction (based on feedback severity distribution)
        satisfaction_score = 1.0 - (high_severity_count / total_assessments) * 0.8
        
        # Determine trend
        trend = self._calculate_trend(avg_quality, compliance_accuracy, validation_pass_rate)
        
        metrics = QualityMetrics(
            timestamp=datetime.now(),
            period_days=days,
            total_assessments=total_assessments,
            average_quality_score=avg_quality,
            average_confidence_score=avg_confidence,
            compliance_accuracy=compliance_accuracy,
            risk_accuracy=risk_accuracy,
            validation_pass_rate=validation_pass_rate,
            user_satisfaction_score=satisfaction_score,
            improvement_trend=trend
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_trend(self, current_quality: float, current_compliance: float, 
                        current_validation: float) -> str:
        """Calculate improvement trend based on historical data"""
        
        if len(self.metrics_history) < 2:
            return 'stable'
        
        previous = self.metrics_history[-1]
        
        quality_change = current_quality - previous.average_quality_score
        compliance_change = current_compliance - previous.compliance_accuracy
        validation_change = current_validation - previous.validation_pass_rate
        
        avg_change = (quality_change + compliance_change + validation_change) / 3
        
        if avg_change > 0.05:
            return 'improving'
        elif avg_change < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def should_trigger_improvement(self, metrics: QualityMetrics) -> tuple[bool, str]:
        """Determine if improvement cycle should be triggered"""
        
        reasons = []
        
        # Check absolute thresholds
        if metrics.average_quality_score < self.quality_thresholds['min_quality_score']:
            reasons.append(f"Quality score below threshold: {metrics.average_quality_score:.2f}")
        
        if metrics.compliance_accuracy < self.quality_thresholds['min_compliance_accuracy']:
            reasons.append(f"Compliance accuracy below threshold: {metrics.compliance_accuracy:.2f}")
        
        if metrics.validation_pass_rate < self.quality_thresholds['min_validation_pass_rate']:
            reasons.append(f"Validation pass rate below threshold: {metrics.validation_pass_rate:.2f}")
        
        # Check declining trend
        if metrics.improvement_trend == 'declining':
            reasons.append("Declining performance trend detected")
        
        # Check for significant decline
        if len(self.metrics_history) >= 2:
            previous = self.metrics_history[-2]
            quality_decline = previous.average_quality_score - metrics.average_quality_score
            if quality_decline > self.quality_thresholds['decline_threshold']:
                reasons.append(f"Significant quality decline: {quality_decline:.2f}")
        
        should_trigger = len(reasons) > 0
        trigger_reason = "; ".join(reasons) if reasons else ""
        
        return should_trigger, trigger_reason

class ContinuousImprovementWorkflow:
    """Main workflow for continuous improvement"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.quality_monitor = QualityMonitor(feedback_collector)
        self.learning_pipeline = AutomatedLearningPipeline(feedback_collector)
        self.prompt_manager = AdaptivePromptManager()
        
        self.improvement_cycles = []
        self.is_running = False
        self.scheduler_thread = None
        
        # Configuration
        self.monitoring_interval_hours = 24  # Check quality daily
        self.improvement_interval_days = 7   # Run improvement cycle weekly
        self.auto_improvement_enabled = True
        
        logger.info("Continuous improvement workflow initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring and improvement"""

        if self.is_running:
            logger.warning("Monitoring already running")
            return

        self.is_running = True

        # Start monitoring in separate thread
        self.scheduler_thread = threading.Thread(target=self._run_monitoring_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("Continuous monitoring started")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("Continuous monitoring stopped")

    def _run_monitoring_loop(self):
        """Run the monitoring loop in a separate thread"""
        last_quality_check = datetime.now()
        last_improvement_cycle = datetime.now()

        while self.is_running:
            try:
                current_time = datetime.now()

                # Check if it's time for quality monitoring
                if (current_time - last_quality_check).total_seconds() >= (self.monitoring_interval_hours * 3600):
                    self._monitor_quality()
                    last_quality_check = current_time

                # Check if it's time for improvement cycle
                if (current_time - last_improvement_cycle).total_seconds() >= (self.improvement_interval_days * 24 * 3600):
                    self._run_improvement_cycle()
                    last_improvement_cycle = current_time

                # Sleep for a minute before checking again
                time.sleep(60)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _monitor_quality(self):
        """Monitor quality metrics and trigger improvements if needed"""
        
        try:
            logger.info("Running quality monitoring check")
            
            # Calculate current metrics
            metrics = self.quality_monitor.calculate_current_metrics()
            
            # Check if improvement is needed
            should_improve, reason = self.quality_monitor.should_trigger_improvement(metrics)
            
            if should_improve and self.auto_improvement_enabled:
                logger.info(f"Triggering improvement cycle: {reason}")
                self._run_improvement_cycle(trigger_reason=reason, metrics_before=metrics)
            else:
                logger.info(f"Quality check complete. Metrics: Quality={metrics.average_quality_score:.2f}, "
                          f"Compliance={metrics.compliance_accuracy:.2f}, Trend={metrics.improvement_trend}")
                
        except Exception as e:
            logger.error(f"Quality monitoring failed: {e}")
    
    def _run_improvement_cycle(self, trigger_reason: str = "Scheduled cycle", 
                             metrics_before: QualityMetrics = None):
        """Run a complete improvement cycle"""
        
        cycle_id = f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"Starting improvement cycle {cycle_id}")
            
            # Get baseline metrics if not provided
            if not metrics_before:
                metrics_before = self.quality_monitor.calculate_current_metrics()
            
            # Create improvement cycle record
            cycle = ImprovementCycle(
                cycle_id=cycle_id,
                start_time=datetime.now(),
                end_time=None,
                trigger_reason=trigger_reason,
                metrics_before=metrics_before,
                metrics_after=None,
                insights_found=0,
                improvements_applied=0,
                success_rate=0.0,
                status='running'
            )
            
            self.improvement_cycles.append(cycle)
            
            # Run learning pipeline
            learning_report = self.learning_pipeline.run_learning_cycle()
            
            cycle.insights_found = learning_report['insights_found']
            cycle.improvements_applied = learning_report['auto_applied_improvements']
            
            # Apply prompt adaptations
            if learning_report['insights']:
                insights = [self._dict_to_insight(i) for i in learning_report['insights']]
                improvements = [self._dict_to_improvement(i) for i in learning_report['improvements']]
                
                adaptations = self.prompt_manager.adapt_prompt_from_feedback(insights, improvements)
                logger.info(f"Applied {len(adaptations)} prompt adaptations")
            
            # Wait for improvements to take effect (in practice, this would be longer)
            time.sleep(5)
            
            # Measure post-improvement metrics
            metrics_after = self.quality_monitor.calculate_current_metrics(days=1)  # Short period for immediate effect
            cycle.metrics_after = metrics_after
            
            # Calculate success rate
            cycle.success_rate = self._calculate_cycle_success_rate(metrics_before, metrics_after)
            
            cycle.end_time = datetime.now()
            cycle.status = 'completed'
            
            logger.info(f"Improvement cycle {cycle_id} completed. Success rate: {cycle.success_rate:.2f}")
            
            # Save cycle results
            self._save_improvement_cycle(cycle)
            
        except Exception as e:
            logger.error(f"Improvement cycle {cycle_id} failed: {e}")
            cycle.status = 'failed'
            cycle.end_time = datetime.now()
    
    def _calculate_cycle_success_rate(self, before: QualityMetrics, 
                                    after: QualityMetrics) -> float:
        """Calculate success rate of improvement cycle"""
        
        improvements = 0
        total_metrics = 4
        
        # Check if key metrics improved
        if after.average_quality_score > before.average_quality_score:
            improvements += 1
        
        if after.compliance_accuracy > before.compliance_accuracy:
            improvements += 1
        
        if after.validation_pass_rate > before.validation_pass_rate:
            improvements += 1
        
        if after.improvement_trend in ['improving', 'stable'] and before.improvement_trend == 'declining':
            improvements += 1
        
        return improvements / total_metrics
    
    def _dict_to_insight(self, insight_dict: Dict[str, Any]):
        """Convert dictionary to LearningInsight object"""
        # This is a simplified conversion - in practice, you'd use proper deserialization
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
    
    def _save_improvement_cycle(self, cycle: ImprovementCycle):
        """Save improvement cycle results"""
        
        cycles_dir = Path("improvement_cycles")
        cycles_dir.mkdir(exist_ok=True)
        
        cycle_file = cycles_dir / f"{cycle.cycle_id}.json"
        
        try:
            # Convert to dictionary for JSON serialization
            cycle_dict = asdict(cycle)
            cycle_dict['start_time'] = cycle.start_time.isoformat()
            if cycle.end_time:
                cycle_dict['end_time'] = cycle.end_time.isoformat()
            
            # Convert metrics to dict
            cycle_dict['metrics_before'] = asdict(cycle.metrics_before)
            cycle_dict['metrics_before']['timestamp'] = cycle.metrics_before.timestamp.isoformat()
            
            if cycle.metrics_after:
                cycle_dict['metrics_after'] = asdict(cycle.metrics_after)
                cycle_dict['metrics_after']['timestamp'] = cycle.metrics_after.timestamp.isoformat()
            
            with open(cycle_file, 'w', encoding='utf-8') as f:
                json.dump(cycle_dict, f, indent=2)
                
            logger.info(f"Improvement cycle saved: {cycle_file}")
            
        except Exception as e:
            logger.error(f"Failed to save improvement cycle: {e}")
    
    def get_improvement_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of improvement activities"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_cycles = [c for c in self.improvement_cycles if c.start_time >= cutoff_date]
        
        if not recent_cycles:
            return {
                'total_cycles': 0,
                'average_success_rate': 0.0,
                'total_improvements': 0,
                'trend': 'no_data'
            }
        
        total_cycles = len(recent_cycles)
        completed_cycles = [c for c in recent_cycles if c.status == 'completed']
        
        avg_success_rate = sum(c.success_rate for c in completed_cycles) / max(1, len(completed_cycles))
        total_improvements = sum(c.improvements_applied for c in recent_cycles)
        
        # Determine overall trend
        if len(completed_cycles) >= 2:
            recent_success = completed_cycles[-1].success_rate
            previous_success = sum(c.success_rate for c in completed_cycles[:-1]) / max(1, len(completed_cycles) - 1)
            
            if recent_success > previous_success + 0.1:
                trend = 'improving'
            elif recent_success < previous_success - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'total_cycles': total_cycles,
            'completed_cycles': len(completed_cycles),
            'average_success_rate': avg_success_rate,
            'total_improvements': total_improvements,
            'trend': trend,
            'last_cycle_date': recent_cycles[-1].start_time.isoformat() if recent_cycles else None
        }
    
    def force_improvement_cycle(self, reason: str = "Manual trigger") -> str:
        """Manually trigger an improvement cycle"""
        
        logger.info(f"Manually triggering improvement cycle: {reason}")
        
        metrics = self.quality_monitor.calculate_current_metrics()
        self._run_improvement_cycle(trigger_reason=reason, metrics_before=metrics)
        
        return self.improvement_cycles[-1].cycle_id if self.improvement_cycles else "failed"
