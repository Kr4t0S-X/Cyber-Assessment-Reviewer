"""
AI Backend module for Cyber Assessment Reviewer
Handles AI model initialization, prompt generation, and analysis logic
"""

import json
import logging
import re
import requests
import time
from typing import Dict, Any, List, Optional

from models import AssessmentResult, ControlData, RiskMetrics
from config import Config
from ai_validation import AIResponseValidator
from feedback_system import FeedbackCollector
from adaptive_prompts import AdaptivePromptManager

logger = logging.getLogger(__name__)

# Optional imports for Transformers mode
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True

    # Try to import BitsAndBytesConfig separately
    try:
        from transformers import BitsAndBytesConfig
        QUANTIZATION_AVAILABLE = True
    except ImportError:
        QUANTIZATION_AVAILABLE = False
        logger.info("Quantization not available. Will use standard precision.")

except ImportError:
    TRANSFORMERS_AVAILABLE = False
    QUANTIZATION_AVAILABLE = False
    logger.info("Transformers not available. Using Ollama mode only.")

class AIBackend:
    """Base class for AI backends"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_response(self, prompt: str, temperature: float = None) -> str:
        """Generate response from AI model"""
        raise NotImplementedError("Subclasses must implement generate_response method")
    
    def is_available(self) -> bool:
        """Check if backend is available"""
        raise NotImplementedError("Subclasses must implement is_available method")

class OllamaBackend(AIBackend):
    """Ollama backend for AI model interaction"""
    
    def __init__(self, config: Config, model_name: str = None):
        super().__init__(config)
        self.model_name = model_name or config.DEFAULT_MODEL_NAME
        self.base_url = config.OLLAMA_BASE_URL
    
    def is_available(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def check_model_available(self) -> bool:
        """Check if the model is available in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(model["name"] == self.model_name for model in models)
        except:
            pass
        return False
    
    def pull_model(self):
        """Pull model using Ollama"""
        try:
            logger.info(f"Pulling {self.model_name} - this may take several minutes...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                stream=True
            )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Ollama: {data['status']}")
                        
            logger.info(f"Model {self.model_name} pulled successfully")
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            raise RuntimeError(f"Could not pull model {self.model_name}. Please run: ollama pull {self.model_name}")
    
    def generate_response(self, prompt: str, temperature: float = None) -> str:
        """Generate response using Ollama API"""
        temperature = temperature or self.config.LLM_TEMPERATURE
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": self.config.LLM_MAX_NEW_TOKENS,
                        "top_p": self.config.LLM_TOP_P,
                        "repeat_penalty": self.config.LLM_REPETITION_PENALTY
                    }
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return ""

class TransformersBackend(AIBackend):
    """Transformers backend for AI model interaction"""
    
    def __init__(self, config: Config, model_name: str = None):
        super().__init__(config)
        self.model_name = model_name or "microsoft/DialoGPT-medium"
        self.device = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
        self.model = None
        self.tokenizer = None
        self.pipe = None
        
        if TRANSFORMERS_AVAILABLE:
            self._initialize_model()
    
    def is_available(self) -> bool:
        """Check if Transformers backend is available"""
        return TRANSFORMERS_AVAILABLE and self.model is not None
    
    def _initialize_model(self):
        """Initialize the Transformers model"""
        logger.info(f"Using device: {self.device}")
        
        # Configure quantization with better error handling
        quantization_config = None
        if self.device == "cuda" and QUANTIZATION_AVAILABLE:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization for efficient memory usage")
            except Exception as e:
                logger.warning(f"Quantization setup failed: {e}. Using standard precision.")
                quantization_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations and error handling
        logger.info(f"Loading {self.model_name} model...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            logger.warning(f"Failed to load model with quantization: {e}")
            logger.info("Retrying without quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.device = "cpu"  # Force CPU if quantization fails
        
        # Create pipeline with optimized settings
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_new_tokens=self.config.LLM_MAX_NEW_TOKENS,
            temperature=self.config.LLM_TEMPERATURE,
            do_sample=True,
            top_p=self.config.LLM_TOP_P,
            repetition_penalty=self.config.LLM_REPETITION_PENALTY
        )
    
    def generate_response(self, prompt: str, temperature: float = None) -> str:
        """Generate response using Transformers pipeline"""
        if not self.pipe:
            return ""
        
        temperature = temperature or self.config.LLM_TEMPERATURE
        
        try:
            response = self.pipe(
                prompt, 
                max_new_tokens=self.config.LLM_MAX_NEW_TOKENS, 
                temperature=temperature
            )[0]['generated_text']
            
            # Extract only the generated part
            generated_text = response[len(prompt):].strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error generating with Transformers: {e}")
            return ""

class CyberAssessmentReviewer:
    """Main class for cyber assessment analysis"""
    
    def __init__(self, config: Config = None, model_name: str = None, use_ollama: bool = None):
        self.config = config or Config()
        self.model_name = model_name or self.config.DEFAULT_MODEL_NAME
        self.use_ollama = use_ollama if use_ollama is not None else self.config.USE_OLLAMA

        logger.info("Initializing Cyber Assessment Reviewer")

        # Initialize AI backend
        self.backend = self._initialize_backend()

        if not self.backend or not self.backend.is_available():
            raise RuntimeError("No AI backend available. Please install Ollama or transformers library.")

        # Initialize response validator
        self.validator = AIResponseValidator()

        # Initialize feedback system and adaptive prompts
        self.feedback_collector = FeedbackCollector()
        self.adaptive_prompts = AdaptivePromptManager()

        logger.info("Model initialization complete with feedback system")
    
    def _initialize_backend(self) -> Optional[AIBackend]:
        """Initialize the appropriate AI backend"""
        if self.use_ollama:
            backend = OllamaBackend(self.config, self.model_name)
            if backend.is_available():
                logger.info("Ollama detected and running")
                if not backend.check_model_available():
                    logger.info(f"Model {self.model_name} not found. Pulling from Ollama...")
                    backend.pull_model()
                logger.info(f"Using Ollama with model: {self.model_name}")
                return backend
            else:
                logger.warning("Ollama not running. Falling back to Transformers mode.")
                self.use_ollama = False
        
        if not self.use_ollama:
            if TRANSFORMERS_AVAILABLE:
                backend = TransformersBackend(self.config, self.model_name)
                if backend.is_available():
                    logger.info("Using Transformers backend")
                    return backend
        
        return None

    def create_cyber_prompt(self, control: ControlData, evidence_text: str, framework: str = "NIST") -> str:
        """Create specialized prompt for cyber assessment analysis with adaptive enhancement"""
        framework_name = self.config.FRAMEWORKS.get(framework, framework)

        # Try to get adaptive prompt first
        try:
            adaptive_prompt = self.adaptive_prompts.get_adaptive_prompt(
                framework, control.control_id
            )
            if adaptive_prompt:
                # Customize adaptive prompt with control-specific information
                return self._customize_adaptive_prompt(adaptive_prompt, control, evidence_text, framework)
        except Exception as e:
            logger.warning(f"Failed to get adaptive prompt, using fallback: {e}")

        # Fallback to original prompt with framework guidance
        framework_guidance = self._get_framework_guidance(framework)

        prompt = f"""You are a senior cybersecurity compliance auditor with 15+ years of experience in {framework_name} assessments.
Your role is to conduct a thorough, evidence-based analysis of cybersecurity control implementations.

=== ASSESSMENT CONTEXT ===
Framework: {framework_name}
Control ID: {control.control_id}
Control Name: {control.control_name}
Control Requirement: {control.requirement}
Supplier's Claimed Implementation: {control.answer}
Declared Status: {control.status or 'Unknown'}

=== FRAMEWORK-SPECIFIC GUIDANCE ===
{framework_guidance}

=== EVIDENCE TO ANALYZE ===
{evidence_text[:4000]}

=== ANALYSIS METHODOLOGY ===
Follow this systematic approach:

1. EVIDENCE EVALUATION:
   - Identify what evidence is provided vs. what is required
   - Assess the quality, completeness, and relevance of evidence
   - Look for specific technical details, configurations, procedures
   - Check for evidence of testing, monitoring, and validation

2. CONTROL IMPLEMENTATION ASSESSMENT:
   - Does the evidence demonstrate the control is actually implemented?
   - Are all aspects of the control requirement addressed?
   - Is the implementation technically sound and effective?
   - Are there any implementation gaps or weaknesses?

3. RISK ANALYSIS:
   - What are the specific security risks if this control fails?
   - How critical is this control to overall security posture?
   - What is the potential business impact of non-compliance?
   - Are there compensating controls that mitigate risks?

4. COMPLIANCE DETERMINATION:
   - Based on evidence quality and implementation effectiveness
   - Consider both technical implementation and operational maturity
   - Account for any identified gaps or weaknesses

=== REQUIRED OUTPUT FORMAT ===
Provide your analysis in this exact JSON format:
{{
    "evidence_validity": "Valid|Partially Valid|Invalid|No Evidence",
    "compliance_status": "Compliant|Partially Compliant|Non-Compliant",
    "risk_level": "Critical|High|Medium|Low",
    "confidence_score": 0.0-1.0,
    "key_findings": ["specific finding 1", "specific finding 2", "specific finding 3"],
    "identified_risks": ["specific risk 1", "specific risk 2", "specific risk 3"],
    "remediation_steps": ["actionable step 1", "actionable step 2", "actionable step 3"],
    "evidence_gaps": ["missing evidence 1", "missing evidence 2"],
    "technical_details": ["technical observation 1", "technical observation 2"],
    "compliance_rationale": "Detailed explanation of compliance determination based on evidence"
}}

=== CRITICAL REQUIREMENTS ===
- Base ALL conclusions on specific evidence provided
- Identify SPECIFIC technical details, not generic statements
- Provide ACTIONABLE remediation steps with clear priorities
- Explain WHY you reached each conclusion
- If evidence is insufficient, clearly state what additional evidence is needed
- Consider the BUSINESS CONTEXT and risk impact
- Be PRECISE about what was observed vs. what was claimed

Provide ONLY the JSON response without additional text or formatting."""

        return prompt

    def _customize_adaptive_prompt(self, adaptive_prompt: str, control: ControlData,
                                 evidence_text: str, framework: str) -> str:
        """Customize adaptive prompt with control-specific information"""

        # Add control-specific context
        customized_prompt = f"""{adaptive_prompt}

=== CONTROL ANALYSIS CONTEXT ===
Framework: {framework}
Control ID: {control.control_id}
Control Name: {control.control_name}
Control Requirement: {control.requirement}
Supplier's Claimed Implementation: {control.answer}
Declared Status: {control.status or 'Unknown'}

=== EVIDENCE TO ANALYZE ===
{evidence_text[:4000]}

=== REQUIRED OUTPUT FORMAT ===
Provide your analysis in this exact JSON format:
{{
    "evidence_validity": "Valid|Partially Valid|Invalid|No Evidence",
    "compliance_status": "Compliant|Partially Compliant|Non-Compliant",
    "risk_level": "Critical|High|Medium|Low",
    "confidence_score": 0.0-1.0,
    "key_findings": ["specific finding 1", "specific finding 2", "specific finding 3"],
    "identified_risks": ["specific risk 1", "specific risk 2", "specific risk 3"],
    "remediation_steps": ["actionable step 1", "actionable step 2", "actionable step 3"],
    "evidence_gaps": ["missing evidence 1", "missing evidence 2"],
    "technical_details": ["technical observation 1", "technical observation 2"],
    "compliance_rationale": "Detailed explanation of compliance determination based on evidence"
}}

Provide ONLY the JSON response without additional text or formatting."""

        return customized_prompt

    def _get_framework_guidance(self, framework: str) -> str:
        """Get framework-specific guidance for analysis"""
        guidance = {
            "NIST": """
NIST Cybersecurity Framework Focus Areas:
- IDENTIFY: Asset management, governance, risk assessment
- PROTECT: Access control, data security, protective technology
- DETECT: Anomaly detection, continuous monitoring
- RESPOND: Response planning, communications, analysis
- RECOVER: Recovery planning, improvements, communications

Key Evidence Types to Look For:
- Policies and procedures documentation
- Technical configuration evidence
- Monitoring and logging capabilities
- Incident response procedures
- Training and awareness programs
- Risk assessment documentation
""",
            "ISO27001": """
ISO 27001:2022 Focus Areas:
- Information Security Management System (ISMS)
- Risk management processes
- Security controls implementation (Annex A)
- Continuous improvement and monitoring
- Management commitment and responsibility

Key Evidence Types to Look For:
- ISMS documentation and scope
- Risk assessment and treatment plans
- Security policies and procedures
- Control implementation evidence
- Internal audit results
- Management review records
""",
            "SOC2": """
SOC 2 Trust Service Criteria:
- Security: Protection against unauthorized access
- Availability: System operation and usability
- Processing Integrity: Complete, valid, accurate processing
- Confidentiality: Information designated as confidential
- Privacy: Personal information collection, use, retention

Key Evidence Types to Look For:
- System descriptions and boundaries
- Control design documentation
- Operating effectiveness evidence
- Monitoring and testing results
- Exception reports and remediation
- Third-party assessments
""",
            "CIS": """
CIS Controls v8 Implementation Groups:
- IG1: Basic cyber hygiene (6 controls)
- IG2: Risk-driven security program (16 controls)
- IG3: Advanced security program (18 controls)

Key Evidence Types to Look For:
- Asset inventory and management
- Software and hardware configurations
- Access control implementations
- Security awareness training
- Vulnerability management processes
- Incident response capabilities
""",
            "PCI-DSS": """
PCI DSS v4.0 Requirements:
- Build and maintain secure networks
- Protect cardholder data
- Maintain vulnerability management
- Implement strong access controls
- Monitor and test networks
- Maintain information security policy

Key Evidence Types to Look For:
- Network segmentation evidence
- Encryption implementation
- Access control matrices
- Vulnerability scan results
- Log monitoring configurations
- Security testing results
"""
        }

        return guidance.get(framework, "General cybersecurity best practices and evidence-based assessment.")

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                # Ensure all required fields exist with enhanced validation
                default_result = {
                    "evidence_validity": "Invalid",
                    "compliance_status": "Non-Compliant",
                    "risk_level": "High",
                    "confidence_score": 0.5,
                    "key_findings": [],
                    "identified_risks": [],
                    "remediation_steps": [],
                    "evidence_gaps": [],
                    "technical_details": [],
                    "compliance_rationale": "Unable to determine compliance due to parsing error"
                }

                # Merge with defaults
                for key, default_value in default_result.items():
                    if key not in result:
                        result[key] = default_value

                return result
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response was: {response}")

            # Return structured default response
            return {
                "evidence_validity": "Error",
                "compliance_status": "Unknown",
                "risk_level": "Unknown",
                "confidence_score": 0.0,
                "key_findings": ["Error parsing AI response"],
                "identified_risks": ["Unable to assess risks due to parsing error"],
                "remediation_steps": ["Manual review required"],
                "evidence_gaps": ["Technical error in assessment"]
            }

    def analyze_control_point(self, control: ControlData, evidence_texts: Dict[str, Dict[str, str]],
                            framework: str = "NIST") -> AssessmentResult:
        """Analyze a single control point with its evidence"""
        # Combine evidence from all files
        combined_evidence = []
        evidence_references = []

        for filename, file_sections in evidence_texts.items():
            for section, text in file_sections.items():
                if "Error" not in section and text.strip():
                    combined_evidence.append(f"[{filename} - {section}]:\n{text}")
                    evidence_references.append(f"{filename} - {section}")

        evidence_text = "\n\n".join(combined_evidence[:self.config.MAX_EVIDENCE_SECTIONS])

        # Create prompt
        prompt = self.create_cyber_prompt(control, evidence_text, framework)

        try:
            # Generate response
            generated_text = self.backend.generate_response(prompt, temperature=self.config.LLM_TEMPERATURE)

            # Parse response
            parsed = self.parse_llm_response(generated_text)

            # Create AssessmentResult with enhanced fields
            result = AssessmentResult(
                control_id=control.control_id,
                control_name=control.control_name,
                requirement=control.requirement,
                supplier_answer=control.answer,
                evidence_validity=parsed['evidence_validity'],
                compliance_status=parsed['compliance_status'],
                risk_level=parsed['risk_level'],
                key_findings=parsed.get('key_findings', [])[:3],
                risks=parsed.get('identified_risks', [])[:3],
                remediation=parsed.get('remediation_steps', [])[:3],
                evidence_references=evidence_references[:3],
                confidence_score=float(parsed.get('confidence_score', 0.5)),
                technical_details=parsed.get('technical_details', [])[:5],
                compliance_rationale=parsed.get('compliance_rationale', ''),
                evidence_gaps=parsed.get('evidence_gaps', [])[:3]
            )

            # Validate the AI response
            validation_result = self.validator.validate_response(result, evidence_text, control)

            # Collect validation feedback
            try:
                original_result_dict = {
                    'compliance_status': result.compliance_status,
                    'risk_level': result.risk_level,
                    'confidence_score': result.confidence_score,
                    'key_findings': result.key_findings
                }

                validation_result_dict = {
                    'quality_score': validation_result.quality_score,
                    'validation_issues': validation_result.validation_issues,
                    'recommendations': validation_result.recommendations
                }

                self.feedback_collector.collect_validation_feedback(
                    control.control_id, framework, original_result_dict, validation_result_dict
                )
            except Exception as e:
                logger.warning(f"Failed to collect validation feedback: {e}")

            # Adjust confidence score based on validation
            result.confidence_score *= validation_result.confidence_adjustment

            # Log validation issues if any
            if validation_result.validation_issues:
                logger.warning(f"Validation issues for control {control.control_id}: {validation_result.validation_issues}")

            # Add validation quality to technical details
            if validation_result.quality_score < 0.7:
                result.technical_details.append(f"AI Response Quality Score: {validation_result.quality_score:.2f}")
                if validation_result.recommendations:
                    result.technical_details.extend(validation_result.recommendations[:2])

            return result

        except Exception as e:
            logger.error(f"Error analyzing control {control.control_id}: {e}")

            # Return error result
            return AssessmentResult(
                control_id=control.control_id,
                control_name=control.control_name,
                requirement=control.requirement,
                supplier_answer=control.answer,
                evidence_validity="Error",
                compliance_status="Unknown",
                risk_level="Unknown",
                key_findings=[f"Analysis error: {str(e)}"],
                risks=["Unable to assess due to error"],
                remediation=["Manual review required"],
                evidence_references=[],
                confidence_score=0.0
            )

    def analyze_multiple_controls(self, controls: List[ControlData],
                                evidence_texts: Dict[str, Dict[str, str]],
                                framework: str = "NIST",
                                max_controls: int = None) -> List[AssessmentResult]:
        """Analyze multiple controls"""
        max_controls = max_controls or self.config.MAX_CONTROLS_DEFAULT
        controls_to_analyze = controls[:max_controls]

        results = []
        for i, control in enumerate(controls_to_analyze):
            logger.info(f"Analyzing control {i+1}/{len(controls_to_analyze)}: {control.control_id}")

            try:
                result = self.analyze_control_point(control, evidence_texts, framework)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing control {i+1}: {e}")
                continue

        return results

    def calculate_risk_metrics(self, results: List[AssessmentResult]) -> RiskMetrics:
        """Calculate overall risk metrics from assessment results"""
        return RiskMetrics.from_results(results, self.config.RISK_WEIGHTS)
