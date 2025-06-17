"""
Adaptive Prompt Engineering System
Dynamic prompt system that evolves based on validation results and feedback patterns
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

from feedback_system import FeedbackCollector, FeedbackType, FeedbackSeverity
from learning_pipeline import LearningInsight, ImprovementAction

logger = logging.getLogger(__name__)

@dataclass
class PromptTemplate:
    """Template for adaptive prompts"""
    template_id: str
    framework: str
    control_category: str
    base_prompt: str
    adaptive_sections: Dict[str, str]
    performance_metrics: Dict[str, float]
    last_updated: datetime
    version: int

@dataclass
class PromptAdaptation:
    """Represents an adaptation to a prompt"""
    adaptation_id: str
    template_id: str
    adaptation_type: str  # 'enhancement', 'correction', 'refinement'
    description: str
    changes: Dict[str, str]
    trigger_feedback: List[str]
    confidence: float
    applied_date: datetime

class AdaptivePromptManager:
    """Manages adaptive prompt templates and their evolution"""
    
    def __init__(self, prompts_dir: str = "adaptive_prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        
        self.templates_file = self.prompts_dir / "prompt_templates.json"
        self.adaptations_file = self.prompts_dir / "prompt_adaptations.json"
        
        # Load existing templates
        self.templates = self._load_templates()
        self.adaptations = self._load_adaptations()
        
        # Initialize base templates if none exist
        if not self.templates:
            self._initialize_base_templates()
        
        logger.info(f"Adaptive prompt manager initialized with {len(self.templates)} templates")
    
    def _initialize_base_templates(self):
        """Initialize base prompt templates for different frameworks and categories"""
        
        base_templates = [
            {
                "template_id": "nist_access_control",
                "framework": "NIST",
                "control_category": "access_control",
                "base_prompt": """You are analyzing an access control implementation for NIST framework compliance.
                
CONTROL ANALYSIS FOCUS:
- Identity and access management systems
- Authentication and authorization mechanisms
- Privileged access controls
- Access review processes

EVIDENCE EVALUATION CRITERIA:
- Look for multi-factor authentication implementation
- Verify role-based access control (RBAC) deployment
- Check for regular access reviews and audits
- Assess privileged access management (PAM) systems

COMPLIANCE DETERMINATION GUIDELINES:
- Compliant: All access control mechanisms properly implemented with evidence
- Partially Compliant: Some mechanisms in place but gaps exist
- Non-Compliant: Significant access control deficiencies identified""",
                "adaptive_sections": {
                    "evidence_focus": "Standard evidence evaluation",
                    "risk_assessment": "Standard risk assessment approach",
                    "compliance_criteria": "Standard compliance criteria"
                }
            },
            {
                "template_id": "nist_data_protection",
                "framework": "NIST",
                "control_category": "data_protection",
                "base_prompt": """You are analyzing data protection controls for NIST framework compliance.
                
CONTROL ANALYSIS FOCUS:
- Data encryption at rest and in transit
- Data classification and handling
- Data loss prevention (DLP) systems
- Backup and recovery procedures

EVIDENCE EVALUATION CRITERIA:
- Verify encryption algorithms and key management
- Check data classification policies and implementation
- Assess DLP system configuration and effectiveness
- Review backup encryption and testing procedures

COMPLIANCE DETERMINATION GUIDELINES:
- Compliant: Comprehensive data protection with strong encryption
- Partially Compliant: Basic protection in place but improvements needed
- Non-Compliant: Inadequate or missing data protection controls""",
                "adaptive_sections": {
                    "evidence_focus": "Standard evidence evaluation",
                    "risk_assessment": "Standard risk assessment approach",
                    "compliance_criteria": "Standard compliance criteria"
                }
            }
        ]
        
        for template_data in base_templates:
            template = PromptTemplate(
                template_id=template_data["template_id"],
                framework=template_data["framework"],
                control_category=template_data["control_category"],
                base_prompt=template_data["base_prompt"],
                adaptive_sections=template_data["adaptive_sections"],
                performance_metrics={"accuracy": 0.7, "confidence": 0.6},
                last_updated=datetime.now(),
                version=1
            )
            self.templates[template.template_id] = template
        
        self._save_templates()
    
    def get_adaptive_prompt(self, framework: str, control_id: str, 
                          control_category: str = None) -> str:
        """Get adaptive prompt for specific control analysis"""
        
        # Determine control category if not provided
        if not control_category:
            control_category = self._categorize_control(control_id, framework)
        
        # Find best matching template
        template = self._find_best_template(framework, control_category)
        
        if not template:
            logger.warning(f"No template found for {framework}/{control_category}, using default")
            return self._get_default_prompt(framework)
        
        # Build adaptive prompt
        adaptive_prompt = self._build_adaptive_prompt(template, control_id)
        
        logger.debug(f"Generated adaptive prompt for {control_id} using template {template.template_id}")
        return adaptive_prompt
    
    def _categorize_control(self, control_id: str, framework: str) -> str:
        """Categorize control based on ID and framework"""
        
        # NIST categorization
        if framework == "NIST":
            if "AC" in control_id or "PR.AC" in control_id:
                return "access_control"
            elif "DS" in control_id or "PR.DS" in control_id:
                return "data_protection"
            elif "DE" in control_id:
                return "detection"
            elif "RS" in control_id:
                return "response"
            elif "RC" in control_id:
                return "recovery"
        
        # ISO 27001 categorization
        elif framework == "ISO27001":
            if "A.9" in control_id:
                return "access_control"
            elif "A.10" in control_id:
                return "cryptography"
            elif "A.12" in control_id:
                return "operations_security"
        
        return "general"
    
    def _find_best_template(self, framework: str, control_category: str) -> Optional[PromptTemplate]:
        """Find the best matching template for framework and category"""
        
        # Exact match
        for template in self.templates.values():
            if template.framework == framework and template.control_category == control_category:
                return template
        
        # Framework match with general category
        for template in self.templates.values():
            if template.framework == framework and template.control_category == "general":
                return template
        
        # General template
        for template in self.templates.values():
            if template.framework == "general" and template.control_category == "general":
                return template
        
        return None
    
    def _build_adaptive_prompt(self, template: PromptTemplate, control_id: str) -> str:
        """Build adaptive prompt from template"""
        
        prompt = template.base_prompt
        
        # Add adaptive sections
        for section_name, section_content in template.adaptive_sections.items():
            if section_content != f"Standard {section_name.replace('_', ' ')}":
                prompt += f"\n\nADAPTIVE {section_name.upper()}:\n{section_content}"
        
        # Add performance-based adjustments
        if template.performance_metrics.get("accuracy", 0.7) < 0.6:
            prompt += "\n\nIMPORTANT: Previous analyses for this category showed accuracy issues. Be extra careful with evidence evaluation and compliance determination."
        
        if template.performance_metrics.get("confidence", 0.6) < 0.5:
            prompt += "\n\nNOTE: Increase confidence by providing more specific evidence references and detailed technical analysis."
        
        return prompt
    
    def _get_default_prompt(self, framework: str) -> str:
        """Get default prompt when no template is available"""
        return f"""You are analyzing cybersecurity controls for {framework} framework compliance.
        
Provide thorough analysis based on the evidence provided, focusing on:
1. Evidence quality and completeness
2. Control implementation effectiveness
3. Risk assessment and impact
4. Specific compliance determination with rationale

Be specific, reference evidence directly, and provide actionable recommendations."""
    
    def adapt_prompt_from_feedback(self, feedback_insights: List[LearningInsight],
                                 improvement_actions: List[ImprovementAction]) -> List[PromptAdaptation]:
        """Adapt prompts based on feedback insights and improvement actions"""
        
        adaptations = []
        
        for action in improvement_actions:
            if action.action_type == "prompt_update":
                adaptation = self._create_prompt_adaptation(action, feedback_insights)
                if adaptation:
                    adaptations.append(adaptation)
                    self._apply_adaptation(adaptation)
        
        # Save adaptations
        self.adaptations.extend(adaptations)
        self._save_adaptations()
        
        logger.info(f"Applied {len(adaptations)} prompt adaptations")
        return adaptations
    
    def _create_prompt_adaptation(self, action: ImprovementAction,
                                insights: List[LearningInsight]) -> Optional[PromptAdaptation]:
        """Create prompt adaptation from improvement action"""
        
        # Find relevant insights
        relevant_insights = [i for i in insights if i.insight_id in action.evidence]
        
        if not relevant_insights:
            return None
        
        # Determine adaptation type and changes
        changes = {}
        
        if "compliance" in action.description.lower():
            changes["compliance_criteria"] = self._generate_compliance_adaptation(relevant_insights)
        
        if "risk" in action.description.lower():
            changes["risk_assessment"] = self._generate_risk_adaptation(relevant_insights)
        
        if "evidence" in action.description.lower():
            changes["evidence_focus"] = self._generate_evidence_adaptation(relevant_insights)
        
        if not changes:
            return None
        
        adaptation = PromptAdaptation(
            adaptation_id=f"adapt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            template_id=self._find_target_template(action, relevant_insights),
            adaptation_type="enhancement",
            description=action.description,
            changes=changes,
            trigger_feedback=action.evidence,
            confidence=action.confidence,
            applied_date=datetime.now()
        )
        
        return adaptation
    
    def _generate_compliance_adaptation(self, insights: List[LearningInsight]) -> str:
        """Generate compliance criteria adaptation"""
        
        compliance_patterns = [i for i in insights if i.insight_type == "compliance_pattern"]
        
        if not compliance_patterns:
            return "Enhanced compliance determination criteria"
        
        adaptations = []
        for pattern in compliance_patterns:
            if "Non-Compliant -> Compliant" in pattern.description:
                adaptations.append("Be more lenient when evidence shows basic implementation")
            elif "Compliant -> Non-Compliant" in pattern.description:
                adaptations.append("Require stronger evidence for compliance determination")
            elif "Partially Compliant" in pattern.description:
                adaptations.append("Consider partial compliance when some controls are implemented")
        
        return "ENHANCED COMPLIANCE CRITERIA:\n" + "\n".join(f"- {a}" for a in adaptations)
    
    def _generate_risk_adaptation(self, insights: List[LearningInsight]) -> str:
        """Generate risk assessment adaptation"""
        
        risk_patterns = [i for i in insights if i.insight_type == "risk_pattern"]
        
        if not risk_patterns:
            return "Enhanced risk assessment approach"
        
        adaptations = []
        for pattern in risk_patterns:
            if "Low -> High" in pattern.description:
                adaptations.append("Consider higher risk levels when evidence shows gaps")
            elif "High -> Low" in pattern.description:
                adaptations.append("Be more conservative with high risk assignments")
        
        return "ENHANCED RISK ASSESSMENT:\n" + "\n".join(f"- {a}" for a in adaptations)
    
    def _generate_evidence_adaptation(self, insights: List[LearningInsight]) -> str:
        """Generate evidence evaluation adaptation"""
        
        validation_patterns = [i for i in insights if i.insight_type == "validation_pattern"]
        
        adaptations = [
            "Focus on specific technical details in evidence",
            "Reference evidence explicitly in findings",
            "Look for testing and validation evidence"
        ]
        
        return "ENHANCED EVIDENCE EVALUATION:\n" + "\n".join(f"- {a}" for a in adaptations)
    
    def _find_target_template(self, action: ImprovementAction, 
                            insights: List[LearningInsight]) -> str:
        """Find target template for adaptation"""
        
        # Extract framework and category from insights
        frameworks = []
        for insight in insights:
            frameworks.extend(insight.affected_frameworks)
        
        most_common_framework = max(set(frameworks), key=frameworks.count) if frameworks else "NIST"
        
        # Find template for this framework
        for template in self.templates.values():
            if template.framework == most_common_framework:
                return template.template_id
        
        # Default to first template
        return list(self.templates.keys())[0] if self.templates else "default"
    
    def _apply_adaptation(self, adaptation: PromptAdaptation):
        """Apply adaptation to template"""
        
        template = self.templates.get(adaptation.template_id)
        if not template:
            logger.warning(f"Template {adaptation.template_id} not found for adaptation")
            return
        
        # Update adaptive sections
        for section, content in adaptation.changes.items():
            template.adaptive_sections[section] = content
        
        # Update metadata
        template.last_updated = datetime.now()
        template.version += 1
        
        # Save updated templates
        self._save_templates()
        
        logger.info(f"Applied adaptation {adaptation.adaptation_id} to template {adaptation.template_id}")
    
    def update_performance_metrics(self, template_id: str, metrics: Dict[str, float]):
        """Update performance metrics for a template"""
        
        template = self.templates.get(template_id)
        if template:
            template.performance_metrics.update(metrics)
            template.last_updated = datetime.now()
            self._save_templates()
            logger.debug(f"Updated metrics for template {template_id}: {metrics}")
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates from storage"""
        templates = {}
        
        try:
            if self.templates_file.exists():
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for template_data in data:
                    template = PromptTemplate(
                        template_id=template_data["template_id"],
                        framework=template_data["framework"],
                        control_category=template_data["control_category"],
                        base_prompt=template_data["base_prompt"],
                        adaptive_sections=template_data["adaptive_sections"],
                        performance_metrics=template_data["performance_metrics"],
                        last_updated=datetime.fromisoformat(template_data["last_updated"]),
                        version=template_data["version"]
                    )
                    templates[template.template_id] = template
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
        
        return templates
    
    def _save_templates(self):
        """Save prompt templates to storage"""
        try:
            data = []
            for template in self.templates.values():
                data.append({
                    "template_id": template.template_id,
                    "framework": template.framework,
                    "control_category": template.control_category,
                    "base_prompt": template.base_prompt,
                    "adaptive_sections": template.adaptive_sections,
                    "performance_metrics": template.performance_metrics,
                    "last_updated": template.last_updated.isoformat(),
                    "version": template.version
                })
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save templates: {e}")
    
    def _load_adaptations(self) -> List[PromptAdaptation]:
        """Load prompt adaptations from storage"""
        adaptations = []
        
        try:
            if self.adaptations_file.exists():
                with open(self.adaptations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for adaptation_data in data:
                    adaptation = PromptAdaptation(
                        adaptation_id=adaptation_data["adaptation_id"],
                        template_id=adaptation_data["template_id"],
                        adaptation_type=adaptation_data["adaptation_type"],
                        description=adaptation_data["description"],
                        changes=adaptation_data["changes"],
                        trigger_feedback=adaptation_data["trigger_feedback"],
                        confidence=adaptation_data["confidence"],
                        applied_date=datetime.fromisoformat(adaptation_data["applied_date"])
                    )
                    adaptations.append(adaptation)
        except Exception as e:
            logger.error(f"Failed to load adaptations: {e}")
        
        return adaptations
    
    def _save_adaptations(self):
        """Save prompt adaptations to storage"""
        try:
            data = []
            for adaptation in self.adaptations:
                data.append({
                    "adaptation_id": adaptation.adaptation_id,
                    "template_id": adaptation.template_id,
                    "adaptation_type": adaptation.adaptation_type,
                    "description": adaptation.description,
                    "changes": adaptation.changes,
                    "trigger_feedback": adaptation.trigger_feedback,
                    "confidence": adaptation.confidence,
                    "applied_date": adaptation.applied_date.isoformat()
                })
            
            with open(self.adaptations_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save adaptations: {e}")
