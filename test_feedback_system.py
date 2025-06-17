#!/usr/bin/env python3
"""
Test Script for Feedback System
Demonstrates the iterative feedback loop and continuous improvement capabilities
"""

import logging
import json
import time
from datetime import datetime

from feedback_integration import FeedbackIntegrationManager
from models import ControlData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_feedback_collection():
    """Test feedback collection from various sources"""
    
    print("🧪 Testing Feedback Collection System")
    print("=" * 50)
    
    # Initialize feedback manager
    feedback_manager = FeedbackIntegrationManager()
    
    # Test 1: User Feedback Collection
    print("\n1. Testing User Feedback Collection...")
    
    original_result = {
        'compliance_status': 'Compliant',
        'risk_level': 'Low',
        'confidence_score': 0.85,
        'key_findings': ['Multi-factor authentication implemented', 'Access reviews conducted']
    }
    
    user_corrections = {
        'compliance_status': 'Partially Compliant',
        'risk_level': 'Medium',
        'key_findings': ['MFA implemented but access reviews incomplete']
    }
    
    feedback_id = feedback_manager.collect_user_feedback(
        control_id="PR.AC-1",
        framework="NIST",
        original_result=original_result,
        user_corrections=user_corrections,
        user_id="test_analyst",
        comments="Access reviews are only conducted annually, not quarterly as required"
    )
    
    print(f"✅ User feedback collected: {feedback_id}")
    
    # Test 2: Expert Feedback Collection
    print("\n2. Testing Expert Feedback Collection...")
    
    expert_assessment = {
        'compliance_status': 'Non-Compliant',
        'risk_level': 'High',
        'key_findings': ['Weak encryption algorithms detected', 'Key management inadequate'],
        'improvement_suggestions': ['Upgrade to AES-256', 'Implement proper key rotation']
    }
    
    expert_feedback_id = feedback_manager.collect_expert_feedback(
        control_id="PR.DS-1",
        framework="NIST",
        original_result={
            'compliance_status': 'Compliant',
            'risk_level': 'Low',
            'confidence_score': 0.9
        },
        expert_assessment=expert_assessment,
        expert_credentials="CISSP, CISM, 15+ years cybersecurity experience",
        review_comments="Current encryption implementation uses deprecated algorithms"
    )
    
    print(f"✅ Expert feedback collected: {expert_feedback_id}")
    
    # Test 3: Get Feedback Summary
    print("\n3. Testing Feedback Summary...")
    
    summary = feedback_manager.get_feedback_summary(days=1)
    print(f"✅ Feedback summary generated:")
    print(f"   - Total feedback: {summary['total_feedback']}")
    print(f"   - By type: {summary['by_type']}")
    print(f"   - By severity: {summary['by_severity']}")
    
    return feedback_manager

def test_learning_pipeline(feedback_manager):
    """Test the automated learning pipeline"""
    
    print("\n🤖 Testing Automated Learning Pipeline")
    print("=" * 50)
    
    # Test 1: Manual Learning Cycle
    print("\n1. Running Manual Learning Cycle...")
    
    learning_report = feedback_manager.run_manual_improvement_cycle(
        reason="Test learning cycle with collected feedback"
    )
    
    print(f"✅ Learning cycle completed:")
    print(f"   - Insights found: {learning_report['insights_found']}")
    print(f"   - Improvements applied: {learning_report['auto_applied_improvements']}")
    print(f"   - Cycle ID: {learning_report.get('improvement_cycle_id', 'N/A')}")
    
    # Test 2: Check Adaptive Prompts
    print("\n2. Testing Adaptive Prompt System...")
    
    # Get adaptive prompt for a control
    adaptive_prompt = feedback_manager.adaptive_prompts.get_adaptive_prompt(
        framework="NIST",
        control_id="PR.AC-1",
        control_category="access_control"
    )
    
    print(f"✅ Adaptive prompt generated (length: {len(adaptive_prompt)} chars)")
    print(f"   Preview: {adaptive_prompt[:200]}...")
    
    return learning_report

def test_analytics_system(feedback_manager):
    """Test the performance analytics system"""
    
    print("\n📊 Testing Performance Analytics System")
    print("=" * 50)
    
    # Test 1: Generate Analytics Report
    print("\n1. Generating Analytics Report...")
    
    analytics_report = feedback_manager.generate_analytics_report(days=30)
    
    print(f"✅ Analytics report generated:")
    print(f"   - Report ID: {analytics_report['report_id']}")
    print(f"   - Period: {analytics_report['period_start']} to {analytics_report['period_end']}")
    print(f"   - Summary metrics: {analytics_report['summary_metrics']}")
    print(f"   - Insights: {len(analytics_report['insights'])}")
    print(f"   - Recommendations: {len(analytics_report['recommendations'])}")
    
    # Test 2: Get System Status
    print("\n2. Checking System Status...")
    
    status = feedback_manager.get_system_status()
    
    print(f"✅ System status retrieved:")
    print(f"   - Feedback entries: {status['feedback_system']['total_feedback_entries']}")
    print(f"   - Monitoring active: {status['continuous_improvement']['monitoring_active']}")
    print(f"   - Prompt templates: {status['adaptive_prompts']['total_templates']}")
    print(f"   - Analytics reports: {status['analytics']['total_reports']}")
    
    return analytics_report

def test_continuous_improvement(feedback_manager):
    """Test the continuous improvement workflow"""
    
    print("\n🔄 Testing Continuous Improvement Workflow")
    print("=" * 50)
    
    # Test 1: Start Monitoring
    print("\n1. Starting Continuous Monitoring...")
    
    feedback_manager.start_continuous_improvement()
    print("✅ Continuous improvement monitoring started")
    
    # Test 2: Check Quality Metrics
    print("\n2. Checking Quality Metrics...")
    
    metrics = feedback_manager.continuous_improvement.quality_monitor.calculate_current_metrics(days=1)
    
    print(f"✅ Quality metrics calculated:")
    print(f"   - Total assessments: {metrics.total_assessments}")
    print(f"   - Average quality: {metrics.average_quality_score:.2f}")
    print(f"   - Compliance accuracy: {metrics.compliance_accuracy:.2f}")
    print(f"   - Trend: {metrics.improvement_trend}")
    
    # Test 3: Check Improvement Summary
    print("\n3. Getting Improvement Summary...")
    
    improvement_summary = feedback_manager.continuous_improvement.get_improvement_summary(days=30)
    
    print(f"✅ Improvement summary:")
    print(f"   - Total cycles: {improvement_summary['total_cycles']}")
    print(f"   - Average success rate: {improvement_summary['average_success_rate']:.2f}")
    print(f"   - Trend: {improvement_summary['trend']}")
    
    # Stop monitoring for test
    feedback_manager.stop_continuous_improvement()
    print("✅ Continuous improvement monitoring stopped")
    
    return improvement_summary

def demonstrate_integration():
    """Demonstrate the complete feedback integration"""
    
    print("\n🎯 Demonstrating Complete Feedback Integration")
    print("=" * 60)
    
    # Simulate a complete workflow
    feedback_manager = FeedbackIntegrationManager()
    
    # Simulate AI assessment with feedback
    print("\n1. Simulating AI Assessment with Feedback...")
    
    # Original AI result
    ai_result = {
        'control_id': 'PR.AC-3',
        'compliance_status': 'Compliant',
        'risk_level': 'Low',
        'confidence_score': 0.8,
        'key_findings': ['Access control matrix maintained', 'Regular access reviews']
    }
    
    # User provides correction
    user_correction = {
        'compliance_status': 'Partially Compliant',
        'risk_level': 'Medium',
        'key_findings': ['Access matrix outdated', 'Reviews not comprehensive']
    }
    
    # Collect feedback
    feedback_id = feedback_manager.collect_user_feedback(
        control_id="PR.AC-3",
        framework="NIST",
        original_result=ai_result,
        user_corrections=user_correction,
        comments="Access matrix last updated 6 months ago"
    )
    
    print(f"✅ Feedback collected: {feedback_id}")
    
    # Trigger learning
    print("\n2. Triggering Learning and Adaptation...")
    
    learning_report = feedback_manager.run_manual_improvement_cycle(
        reason="User correction indicates access control assessment issues"
    )
    
    print(f"✅ Learning completed: {learning_report['insights_found']} insights")
    
    # Generate analytics
    print("\n3. Generating Performance Analytics...")
    
    analytics = feedback_manager.generate_analytics_report(days=1)
    
    print(f"✅ Analytics generated: {analytics['report_id']}")
    
    # Show improvement impact
    if analytics['improvement_impact']:
        print(f"   - Improvement impact: {analytics['improvement_impact']}")
    
    print("\n🎉 Complete feedback integration demonstrated successfully!")
    
    return feedback_manager

def main():
    """Main test function"""
    
    print("🚀 Feedback System Integration Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    try:
        # Test 1: Feedback Collection
        feedback_manager = test_feedback_collection()
        
        # Test 2: Learning Pipeline
        learning_report = test_learning_pipeline(feedback_manager)
        
        # Test 3: Analytics System
        analytics_report = test_analytics_system(feedback_manager)
        
        # Test 4: Continuous Improvement
        improvement_summary = test_continuous_improvement(feedback_manager)
        
        # Test 5: Complete Integration
        demonstrate_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        print(f"\nTest Summary:")
        print(f"- Feedback collection: ✅ Working")
        print(f"- Learning pipeline: ✅ Working")
        print(f"- Analytics system: ✅ Working")
        print(f"- Continuous improvement: ✅ Working")
        print(f"- Complete integration: ✅ Working")
        
        print(f"\nThe iterative feedback loop system is fully operational!")
        print(f"The AI will now continuously improve based on user feedback and validation results.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
