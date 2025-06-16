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
        
        logger.info("Model initialization complete")
    
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
        """Create specialized prompt for cyber assessment analysis"""
        framework_name = self.config.FRAMEWORKS.get(framework, framework)

        prompt = f"""You are a senior cybersecurity compliance auditor reviewing control assessments.
Analyze the following control against the provided evidence using the {framework_name} framework.

CONTROL INFORMATION:
- Control ID: {control.control_id}
- Control Name: {control.control_name}
- Requirement: {control.requirement}
- Supplier Answer: {control.answer}
- Implementation Status: {control.status or 'Unknown'}

EVIDENCE PROVIDED:
{evidence_text[:3000]}

Provide your analysis in the following JSON format:
{{
    "evidence_validity": "Valid|Partially Valid|Invalid|No Evidence",
    "compliance_status": "Compliant|Partially Compliant|Non-Compliant",
    "risk_level": "Critical|High|Medium|Low",
    "confidence_score": 0.0-1.0,
    "key_findings": ["finding1", "finding2", "finding3"],
    "identified_risks": ["risk1", "risk2", "risk3"],
    "remediation_steps": ["step1", "step2", "step3"],
    "evidence_gaps": ["gap1", "gap2"]
}}

Focus on:
1. Whether evidence adequately demonstrates the control implementation
2. Specific security vulnerabilities or compliance gaps
3. Practical, prioritized remediation recommendations
4. Risk impact on the organization

Provide only the JSON response without additional text."""

        return prompt

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                # Ensure all required fields exist
                default_result = {
                    "evidence_validity": "Invalid",
                    "compliance_status": "Non-Compliant",
                    "risk_level": "High",
                    "confidence_score": 0.5,
                    "key_findings": [],
                    "identified_risks": [],
                    "remediation_steps": [],
                    "evidence_gaps": []
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

            # Create AssessmentResult
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
                confidence_score=float(parsed.get('confidence_score', 0.5))
            )

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
