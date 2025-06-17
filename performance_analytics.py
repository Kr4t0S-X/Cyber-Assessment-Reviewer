"""
Performance Tracking and Analytics Dashboard
Comprehensive metrics tracking and analytics for monitoring AI improvement over time
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import statistics
from collections import defaultdict

from feedback_system import FeedbackCollector, FeedbackType, FeedbackSeverity
from continuous_improvement import QualityMetrics, ImprovementCycle

logger = logging.getLogger(__name__)

@dataclass
class PerformanceTrend:
    """Represents a performance trend over time"""
    metric_name: str
    time_period: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0.0 to 1.0
    current_value: float
    previous_value: float
    change_percentage: float
    confidence: float

@dataclass
class AnalyticsReport:
    """Comprehensive analytics report"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    summary_metrics: Dict[str, float]
    trends: List[PerformanceTrend]
    insights: List[str]
    recommendations: List[str]
    improvement_impact: Dict[str, float]

class PerformanceAnalytics:
    """Analytics engine for AI performance tracking"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
        self.analytics_dir = Path("analytics_data")
        self.analytics_dir.mkdir(exist_ok=True)
        
        self.reports_file = self.analytics_dir / "analytics_reports.json"
        self.metrics_history_file = self.analytics_dir / "metrics_history.json"
        
        # Load historical data
        self.metrics_history = self._load_metrics_history()
        self.reports_history = self._load_reports_history()
        
        logger.info("Performance analytics initialized")
    
    def generate_comprehensive_report(self, days: int = 30) -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Generating analytics report for {days} days")
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(days)
        
        # Analyze trends
        trends = self._analyze_performance_trends(days)
        
        # Generate insights
        insights = self._generate_insights(summary_metrics, trends)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trends, insights)
        
        # Calculate improvement impact
        improvement_impact = self._calculate_improvement_impact(days)
        
        report = AnalyticsReport(
            report_id=report_id,
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            summary_metrics=summary_metrics,
            trends=trends,
            insights=insights,
            recommendations=recommendations,
            improvement_impact=improvement_impact
        )
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Analytics report generated: {report_id}")
        return report
    
    def _calculate_summary_metrics(self, days: int) -> Dict[str, float]:
        """Calculate summary metrics for the period"""
        
        feedback_entries = self.feedback_collector.get_recent_feedback(days)
        
        if not feedback_entries:
            return {
                'total_assessments': 0,
                'average_quality_score': 0.0,
                'compliance_accuracy': 0.0,
                'risk_accuracy': 0.0,
                'user_satisfaction': 0.0,
                'improvement_rate': 0.0
            }
        
        total_assessments = len(feedback_entries)
        
        # Quality scores
        validation_entries = [e for e in feedback_entries if e.feedback_type == FeedbackType.VALIDATION_RESULT]
        avg_quality = sum(e.evidence_quality or 0.5 for e in validation_entries) / max(1, len(validation_entries))
        
        # Accuracy metrics
        correction_entries = [e for e in feedback_entries if e.corrected_compliance_status]
        compliance_accuracy = sum(1 for e in correction_entries 
                                if e.original_compliance_status == e.corrected_compliance_status) / max(1, len(correction_entries))
        
        risk_correction_entries = [e for e in feedback_entries if e.corrected_risk_level]
        risk_accuracy = sum(1 for e in risk_correction_entries 
                          if e.original_risk_level == e.corrected_risk_level) / max(1, len(risk_correction_entries))
        
        # User satisfaction (inverse of high-severity feedback)
        high_severity_count = sum(1 for e in feedback_entries 
                                if e.severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH])
        user_satisfaction = 1.0 - (high_severity_count / total_assessments)
        
        # Improvement rate (based on trend over time)
        improvement_rate = self._calculate_improvement_rate(days)
        
        return {
            'total_assessments': total_assessments,
            'average_quality_score': avg_quality,
            'compliance_accuracy': compliance_accuracy,
            'risk_accuracy': risk_accuracy,
            'user_satisfaction': user_satisfaction,
            'improvement_rate': improvement_rate
        }
    
    def _analyze_performance_trends(self, days: int) -> List[PerformanceTrend]:
        """Analyze performance trends over time"""
        
        trends = []
        
        # Get historical metrics
        current_period = self._get_period_metrics(days)
        previous_period = self._get_period_metrics(days, offset_days=days)
        
        # Analyze each metric
        metrics_to_analyze = [
            'average_quality_score',
            'compliance_accuracy', 
            'risk_accuracy',
            'user_satisfaction'
        ]
        
        for metric in metrics_to_analyze:
            current_value = current_period.get(metric, 0.0)
            previous_value = previous_period.get(metric, 0.0)
            
            if previous_value > 0:
                change_percentage = ((current_value - previous_value) / previous_value) * 100
            else:
                change_percentage = 0.0
            
            # Determine trend direction and strength
            if abs(change_percentage) < 2:  # Less than 2% change
                direction = 'stable'
                strength = 0.1
            elif change_percentage > 0:
                direction = 'improving'
                strength = min(1.0, abs(change_percentage) / 20)  # Normalize to 0-1
            else:
                direction = 'declining'
                strength = min(1.0, abs(change_percentage) / 20)
            
            # Calculate confidence based on data points
            confidence = min(1.0, current_period.get('total_assessments', 0) / 50)
            
            trend = PerformanceTrend(
                metric_name=metric,
                time_period=f"{days} days",
                trend_direction=direction,
                trend_strength=strength,
                current_value=current_value,
                previous_value=previous_value,
                change_percentage=change_percentage,
                confidence=confidence
            )
            
            trends.append(trend)
        
        return trends
    
    def _get_period_metrics(self, days: int, offset_days: int = 0) -> Dict[str, float]:
        """Get metrics for a specific time period"""
        
        end_date = datetime.now() - timedelta(days=offset_days)
        start_date = end_date - timedelta(days=days)
        
        # Filter feedback for this period
        all_feedback = self.feedback_collector.get_recent_feedback(days + offset_days)
        period_feedback = [f for f in all_feedback 
                          if start_date <= f.timestamp <= end_date]
        
        if not period_feedback:
            return {'total_assessments': 0}
        
        # Calculate metrics for this period
        total_assessments = len(period_feedback)
        
        validation_entries = [e for e in period_feedback if e.feedback_type == FeedbackType.VALIDATION_RESULT]
        avg_quality = sum(e.evidence_quality or 0.5 for e in validation_entries) / max(1, len(validation_entries))
        
        correction_entries = [e for e in period_feedback if e.corrected_compliance_status]
        compliance_accuracy = sum(1 for e in correction_entries 
                                if e.original_compliance_status == e.corrected_compliance_status) / max(1, len(correction_entries))
        
        risk_correction_entries = [e for e in period_feedback if e.corrected_risk_level]
        risk_accuracy = sum(1 for e in risk_correction_entries 
                          if e.original_risk_level == e.corrected_risk_level) / max(1, len(risk_correction_entries))
        
        high_severity_count = sum(1 for e in period_feedback 
                                if e.severity in [FeedbackSeverity.CRITICAL, FeedbackSeverity.HIGH])
        user_satisfaction = 1.0 - (high_severity_count / total_assessments)
        
        return {
            'total_assessments': total_assessments,
            'average_quality_score': avg_quality,
            'compliance_accuracy': compliance_accuracy,
            'risk_accuracy': risk_accuracy,
            'user_satisfaction': user_satisfaction
        }
    
    def _calculate_improvement_rate(self, days: int) -> float:
        """Calculate overall improvement rate"""
        
        # Get metrics for multiple periods to calculate trend
        periods = []
        for i in range(4):  # Look at 4 periods
            period_metrics = self._get_period_metrics(days // 4, offset_days=i * (days // 4))
            if period_metrics.get('total_assessments', 0) > 0:
                # Calculate composite score
                composite = (
                    period_metrics.get('average_quality_score', 0) * 0.3 +
                    period_metrics.get('compliance_accuracy', 0) * 0.3 +
                    period_metrics.get('risk_accuracy', 0) * 0.2 +
                    period_metrics.get('user_satisfaction', 0) * 0.2
                )
                periods.append(composite)
        
        if len(periods) < 2:
            return 0.0
        
        # Calculate linear trend
        x_values = list(range(len(periods)))
        y_values = periods
        
        # Simple linear regression slope
        n = len(periods)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalize slope to improvement rate (0-1 scale)
        return max(0.0, min(1.0, slope + 0.5))
    
    def _generate_insights(self, metrics: Dict[str, float], trends: List[PerformanceTrend]) -> List[str]:
        """Generate insights from metrics and trends"""
        
        insights = []
        
        # Overall performance insights
        if metrics['average_quality_score'] > 0.8:
            insights.append("AI model is performing at high quality levels")
        elif metrics['average_quality_score'] < 0.6:
            insights.append("AI model quality needs improvement - consider additional training")
        
        # Accuracy insights
        if metrics['compliance_accuracy'] < 0.7:
            insights.append("Compliance determination accuracy is below target - review prompt engineering")
        
        if metrics['risk_accuracy'] < 0.7:
            insights.append("Risk assessment accuracy needs improvement - update risk criteria")
        
        # Trend insights
        improving_trends = [t for t in trends if t.trend_direction == 'improving']
        declining_trends = [t for t in trends if t.trend_direction == 'declining']
        
        if len(improving_trends) > len(declining_trends):
            insights.append("Overall performance trend is positive - continuous improvement is working")
        elif len(declining_trends) > len(improving_trends):
            insights.append("Performance decline detected - immediate intervention needed")
        
        # Specific metric insights
        for trend in trends:
            if trend.trend_direction == 'declining' and trend.trend_strength > 0.5:
                insights.append(f"{trend.metric_name} showing significant decline ({trend.change_percentage:.1f}%)")
            elif trend.trend_direction == 'improving' and trend.trend_strength > 0.5:
                insights.append(f"{trend.metric_name} showing strong improvement ({trend.change_percentage:.1f}%)")
        
        return insights
    
    def _generate_recommendations(self, trends: List[PerformanceTrend], insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on trends
        declining_metrics = [t for t in trends if t.trend_direction == 'declining']
        
        for trend in declining_metrics:
            if trend.metric_name == 'average_quality_score':
                recommendations.append("Increase validation threshold and review prompt templates")
            elif trend.metric_name == 'compliance_accuracy':
                recommendations.append("Update compliance determination criteria and add more training examples")
            elif trend.metric_name == 'risk_accuracy':
                recommendations.append("Refine risk assessment guidelines and update knowledge base")
            elif trend.metric_name == 'user_satisfaction':
                recommendations.append("Gather more user feedback and implement user-requested improvements")
        
        # Based on insights
        if any("quality needs improvement" in insight for insight in insights):
            recommendations.append("Run comprehensive model retraining with recent feedback data")
        
        if any("accuracy is below target" in insight for insight in insights):
            recommendations.append("Implement additional validation rules and expert review processes")
        
        # General recommendations
        if len(declining_metrics) > 2:
            recommendations.append("Consider comprehensive system review and potential model upgrade")
        
        if not recommendations:
            recommendations.append("Continue current improvement strategy - performance is stable")
        
        return recommendations
    
    def _calculate_improvement_impact(self, days: int) -> Dict[str, float]:
        """Calculate impact of improvement cycles"""
        
        # Load improvement cycle data
        cycles_dir = Path("improvement_cycles")
        if not cycles_dir.exists():
            return {}
        
        impact = {}
        
        # Find cycles in the period
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for cycle_file in cycles_dir.glob("*.json"):
            try:
                with open(cycle_file, 'r', encoding='utf-8') as f:
                    cycle_data = json.load(f)
                
                cycle_date = datetime.fromisoformat(cycle_data['start_time'])
                if cycle_date >= cutoff_date:
                    # Calculate impact
                    if cycle_data.get('metrics_before') and cycle_data.get('metrics_after'):
                        before = cycle_data['metrics_before']
                        after = cycle_data['metrics_after']
                        
                        for metric in ['average_quality_score', 'compliance_accuracy', 'validation_pass_rate']:
                            if metric in before and metric in after:
                                improvement = after[metric] - before[metric]
                                impact[f"{metric}_improvement"] = impact.get(f"{metric}_improvement", 0) + improvement
                        
                        impact['total_cycles'] = impact.get('total_cycles', 0) + 1
                        impact['success_rate'] = impact.get('success_rate', 0) + cycle_data.get('success_rate', 0)
            
            except Exception as e:
                logger.warning(f"Failed to process cycle file {cycle_file}: {e}")
        
        # Average the success rate
        if impact.get('total_cycles', 0) > 0:
            impact['success_rate'] = impact['success_rate'] / impact['total_cycles']
        
        return impact
    
    def _save_report(self, report: AnalyticsReport):
        """Save analytics report"""
        
        try:
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
                        'previous_value': t.previous_value,
                        'change_percentage': t.change_percentage,
                        'confidence': t.confidence
                    } for t in report.trends
                ],
                'insights': report.insights,
                'recommendations': report.recommendations,
                'improvement_impact': report.improvement_impact
            }
            
            # Save individual report
            report_file = self.analytics_dir / f"{report.report_id}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2)
            
            # Update reports history
            self.reports_history.append(report_dict)
            with open(self.reports_file, 'w', encoding='utf-8') as f:
                json.dump(self.reports_history, f, indent=2)
            
            logger.info(f"Analytics report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analytics report: {e}")
    
    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical metrics data"""
        try:
            if self.metrics_history_file.exists():
                with open(self.metrics_history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metrics history: {e}")
        return []
    
    def _load_reports_history(self) -> List[Dict[str, Any]]:
        """Load historical reports data"""
        try:
            if self.reports_file.exists():
                with open(self.reports_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load reports history: {e}")
        return []
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for analytics dashboard"""
        
        # Get latest report
        latest_report = self.reports_history[-1] if self.reports_history else None
        
        # Get recent trends
        recent_metrics = self._calculate_summary_metrics(7)  # Last 7 days
        
        # Get improvement summary
        improvement_summary = self._calculate_improvement_impact(30)
        
        return {
            'latest_report': latest_report,
            'recent_metrics': recent_metrics,
            'improvement_summary': improvement_summary,
            'total_reports': len(self.reports_history),
            'last_updated': datetime.now().isoformat()
        }
