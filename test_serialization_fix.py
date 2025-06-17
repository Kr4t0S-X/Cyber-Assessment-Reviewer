#!/usr/bin/env python3
"""
Test script to verify the JSON serialization fix in the AI accuracy testing framework
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

# Import the safe serialization function
from test_ai_accuracy import safe_dataclass_to_dict, TestResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MockAssessmentResult:
    """Mock assessment result for testing"""
    control_id: str
    compliance_status: str
    confidence_score: float

def test_serialization_scenarios():
    """Test various serialization scenarios"""
    print("🧪 Testing JSON Serialization Fix")
    print("=" * 50)
    
    # Test 1: Normal dataclass serialization
    print("\n1. Testing normal dataclass serialization...")
    mock_result = MockAssessmentResult("TEST-1", "Compliant", 0.85)
    serialized = safe_dataclass_to_dict(mock_result)
    print(f"✅ Success: {serialized}")
    
    # Test 2: None value serialization
    print("\n2. Testing None value serialization...")
    none_result = safe_dataclass_to_dict(None)
    print(f"✅ Success: {none_result}")
    
    # Test 3: Dictionary serialization
    print("\n3. Testing dictionary serialization...")
    dict_result = {"test": "value", "number": 42}
    serialized_dict = safe_dataclass_to_dict(dict_result)
    print(f"✅ Success: {serialized_dict}")
    
    # Test 4: TestResult with None ai_result
    print("\n4. Testing TestResult with None ai_result...")
    test_result_none = TestResult(
        test_id="test_none",
        passed=False,
        accuracy_score=0.0,
        compliance_match=False,
        risk_level_match=False,
        findings_quality=0.0,
        issues=["Test failed"],
        ai_result=None
    )
    
    try:
        serialized_test = safe_dataclass_to_dict(test_result_none)
        print(f"✅ Success: TestResult with None ai_result serialized")
        
        # Test JSON serialization
        json_str = json.dumps(serialized_test, indent=2)
        print(f"✅ Success: JSON serialization works")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 5: TestResult with valid ai_result
    print("\n5. Testing TestResult with valid ai_result...")
    test_result_valid = TestResult(
        test_id="test_valid",
        passed=True,
        accuracy_score=0.85,
        compliance_match=True,
        risk_level_match=True,
        findings_quality=0.9,
        issues=[],
        ai_result=mock_result
    )
    
    try:
        serialized_test_valid = safe_dataclass_to_dict(test_result_valid)
        # Also serialize the nested ai_result
        if serialized_test_valid and "ai_result" in serialized_test_valid:
            serialized_test_valid["ai_result"] = safe_dataclass_to_dict(serialized_test_valid["ai_result"])
        
        print(f"✅ Success: TestResult with valid ai_result serialized")
        
        # Test JSON serialization
        json_str = json.dumps(serialized_test_valid, indent=2)
        print(f"✅ Success: JSON serialization works")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 6: Edge case - corrupted object
    print("\n6. Testing edge case - non-serializable object...")
    class NonSerializable:
        def __init__(self):
            self.data = lambda x: x  # Lambda functions can't be serialized
    
    non_serializable = NonSerializable()
    try:
        result = safe_dataclass_to_dict(non_serializable)
        print(f"✅ Success: Non-serializable object handled gracefully: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All serialization tests completed!")

def test_full_workflow():
    """Test the full workflow that was failing before"""
    print("\n🔄 Testing Full Workflow (Previous Failure Scenario)")
    print("=" * 60)
    
    # Simulate the scenario that was failing
    test_results = [
        TestResult(
            test_id="test_1",
            passed=True,
            accuracy_score=0.8,
            compliance_match=True,
            risk_level_match=True,
            findings_quality=0.7,
            issues=[],
            ai_result=MockAssessmentResult("CTRL-1", "Compliant", 0.8)
        ),
        TestResult(
            test_id="test_2",
            passed=False,
            accuracy_score=0.0,
            compliance_match=False,
            risk_level_match=False,
            findings_quality=0.0,
            issues=["AI analysis failed"],
            ai_result=None  # This was causing the original error
        )
    ]
    
    # Simulate the original serialization code (fixed version)
    try:
        serializable_results = []
        for result in test_results:
            result_dict = safe_dataclass_to_dict(result)
            
            # Handle ai_result field safely
            if "ai_result" in result_dict:
                result_dict["ai_result"] = safe_dataclass_to_dict(result_dict["ai_result"])
            
            serializable_results.append(result_dict)
        
        # Test JSON serialization
        final_data = {
            "results": serializable_results,
            "metrics": {"pass_rate": 0.5, "total_tests": 2}
        }
        
        json_output = json.dumps(final_data, indent=2)
        print("✅ Success: Full workflow serialization works!")
        print(f"📄 JSON output length: {len(json_output)} characters")
        
        # Save to test file
        with open("test_serialization_output.json", "w") as f:
            json.dump(final_data, f, indent=2)
        print("✅ Success: Test file saved as 'test_serialization_output.json'")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_serialization_scenarios()
    test_full_workflow()
    print("\n🎯 Serialization fix verification complete!")
