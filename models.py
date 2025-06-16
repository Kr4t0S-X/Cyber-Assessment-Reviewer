"""
Data models for Cyber Assessment Reviewer
Contains dataclasses and model definitions for the application
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class EvidenceValidity(Enum):
    """Enumeration for evidence validity levels"""
    VALID = "Valid"
    PARTIALLY_VALID = "Partially Valid"
    INVALID = "Invalid"
    NO_EVIDENCE = "No Evidence"
    ERROR = "Error"

class ComplianceStatus(Enum):
    """Enumeration for compliance status levels"""
    COMPLIANT = "Compliant"
    PARTIALLY_COMPLIANT = "Partially Compliant"
    NON_COMPLIANT = "Non-Compliant"
    UNKNOWN = "Unknown"

class RiskLevel(Enum):
    """Enumeration for risk levels"""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"

@dataclass
class AssessmentResult:
    """Data class for individual control assessment results"""
    control_id: str
    control_name: str
    requirement: str
    supplier_answer: str
    evidence_validity: str
    compliance_status: str
    risk_level: str
    key_findings: List[str]
    risks: List[str]
    remediation: List[str]
    evidence_references: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssessmentResult':
        """Create instance from dictionary"""
        return cls(**data)
    
    def get_risk_priority(self) -> int:
        """Get numeric priority based on risk level (higher = more critical)"""
        priority_map = {
            RiskLevel.CRITICAL.value: 5,
            RiskLevel.HIGH.value: 4,
            RiskLevel.MEDIUM.value: 3,
            RiskLevel.LOW.value: 2,
            RiskLevel.UNKNOWN.value: 1
        }
        return priority_map.get(self.risk_level, 1)

@dataclass
class ControlData:
    """Data class for control information from assessment files"""
    control_id: str
    control_name: str
    requirement: str
    answer: str
    status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ControlData':
        """Create instance from dictionary"""
        # Handle flexible field mapping
        control_id = data.get('control_id') or data.get('control') or data.get('id', 'N/A')
        control_name = data.get('control_name') or data.get('name') or data.get('title', 'N/A')
        requirement = data.get('requirement') or data.get('description') or data.get('control_description', 'N/A')
        answer = data.get('answer') or data.get('response') or data.get('supplier_answer', 'N/A')
        status = data.get('status') or data.get('implementation_status') or data.get('compliance_status')
        
        return cls(
            control_id=control_id,
            control_name=control_name,
            requirement=requirement,
            answer=answer,
            status=status
        )

@dataclass
class EvidenceFile:
    """Data class for evidence file information"""
    filename: str
    filepath: str
    file_type: str
    size_bytes: int
    sections: Dict[str, str]  # section_name -> text_content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class RiskMetrics:
    """Data class for overall risk assessment metrics"""
    overall_risk_score: float
    risk_distribution: Dict[str, int]
    compliance_distribution: Dict[str, int]
    compliance_percentage: float
    high_priority_controls: List[Dict[str, Any]]
    total_controls_assessed: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_results(cls, results: List[AssessmentResult], risk_weights: Dict[str, int]) -> 'RiskMetrics':
        """Calculate risk metrics from assessment results"""
        if not results:
            return cls(
                overall_risk_score=0.0,
                risk_distribution={},
                compliance_distribution={},
                compliance_percentage=0.0,
                high_priority_controls=[],
                total_controls_assessed=0
            )
        
        # Count by risk level
        risk_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
        compliance_counts = {"Compliant": 0, "Partially Compliant": 0, "Non-Compliant": 0, "Unknown": 0}
        
        for result in results:
            risk_counts[result.risk_level] = risk_counts.get(result.risk_level, 0) + 1
            compliance_counts[result.compliance_status] = compliance_counts.get(result.compliance_status, 0) + 1
        
        # Calculate risk score (0-100, higher is worse)
        total_risk = sum(risk_counts[level] * risk_weights.get(level, 0) for level in risk_counts)
        max_risk = len(results) * max(risk_weights.values())  # If all were highest risk
        risk_score = (total_risk / max_risk * 100) if max_risk > 0 else 0
        
        # Calculate compliance percentage
        compliant = compliance_counts.get("Compliant", 0)
        total = len(results)
        compliance_percentage = (compliant / total * 100) if total > 0 else 0
        
        # Identify high priority controls
        high_priority = [
            {
                "control_id": r.control_id,
                "control_name": r.control_name,
                "risk_level": r.risk_level,
                "compliance_status": r.compliance_status,
                "confidence_score": r.confidence_score
            }
            for r in results 
            if r.risk_level in ["Critical", "High"] and r.compliance_status in ["Non-Compliant", "Partially Compliant"]
        ]
        
        # Sort by risk priority and confidence
        high_priority.sort(key=lambda x: (
            -5 if x["risk_level"] == "Critical" else -4,  # Critical first
            -x["confidence_score"]  # Higher confidence first
        ))
        
        return cls(
            overall_risk_score=round(risk_score, 1),
            risk_distribution=risk_counts,
            compliance_distribution=compliance_counts,
            compliance_percentage=round(compliance_percentage, 1),
            high_priority_controls=high_priority[:10],  # Top 10
            total_controls_assessed=len(results)
        )

@dataclass
class AnalysisSession:
    """Data class for analysis session information"""
    session_id: str
    created_at: datetime
    framework: str
    assessment_file: Optional[str] = None
    evidence_files: List[str] = None
    controls_count: int = 0
    status: str = "created"  # created, processing, completed, error
    
    def __post_init__(self):
        if self.evidence_files is None:
            self.evidence_files = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSession':
        """Create instance from dictionary"""
        data = data.copy()
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

@dataclass
class AnalysisReport:
    """Data class for complete analysis report"""
    session: AnalysisSession
    results: List[AssessmentResult]
    risk_metrics: RiskMetrics
    framework: str
    timestamp: datetime
    total_controls: int
    analyzed_controls: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session': self.session.to_dict(),
            'results': [result.to_dict() for result in self.results],
            'risk_metrics': self.risk_metrics.to_dict(),
            'framework': self.framework,
            'timestamp': self.timestamp.isoformat(),
            'total_controls': self.total_controls,
            'analyzed_controls': self.analyzed_controls
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisReport':
        """Create instance from dictionary"""
        return cls(
            session=AnalysisSession.from_dict(data['session']),
            results=[AssessmentResult.from_dict(r) for r in data['results']],
            risk_metrics=RiskMetrics(**data['risk_metrics']),
            framework=data['framework'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            total_controls=data['total_controls'],
            analyzed_controls=data['analyzed_controls']
        )

# Type aliases for better code readability
ControlDict = Dict[str, Any]
EvidenceDict = Dict[str, Dict[str, str]]  # filename -> {section -> content}
FrameworkDict = Dict[str, str]  # framework_code -> framework_name
