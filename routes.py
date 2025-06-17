"""
Flask routes for Cyber Assessment Reviewer
Contains all web endpoints and request handling logic
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, session, send_file
import pandas as pd

from config import Config
from models import ControlData, AnalysisSession
from file_processors import AssessmentFileProcessor, EvidenceProcessor, is_supported_file_type
from ai_backend import CyberAssessmentReviewer
from analysis_workflow import OptimizedAnalysisWorkflow
from templates import get_html_template
from utils import (
    allowed_file, secure_save_file, clean_sample_control, 
    safe_json_dump, safe_json_dumps
)

logger = logging.getLogger(__name__)

def create_routes(app: Flask, config: Config, reviewer: CyberAssessmentReviewer):
    """Create and register all Flask routes"""
    
    # Initialize processors and workflow
    assessment_processor = AssessmentFileProcessor(config)
    evidence_processor = EvidenceProcessor(config)
    optimized_workflow = OptimizedAnalysisWorkflow(reviewer)
    
    @app.route('/')
    def index():
        """Render the main page"""
        backend = "Ollama" if reviewer.use_ollama else "Transformers"
        html_content = get_html_template(backend)
        return html_content
    
    @app.route('/upload_assessment', methods=['POST'])
    def upload_assessment():
        """Handle assessment file upload"""
        if 'assessment' not in request.files:
            return jsonify({'error': 'No assessment file provided'}), 400
        
        file = request.files['assessment']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file or not allowed_file(file.filename, config.ALLOWED_EXTENSIONS):
            return jsonify({'error': 'Invalid file type'}), 400
        
        try:
            # Create session
            session_id = session.get('session_id')
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
            
            session_path = Path(f"{config.SESSION_FOLDER}/{session_id}")
            session_path.mkdir(exist_ok=True)
            
            # Save assessment file
            filepath = secure_save_file(file, session_path, "assessment")
            
            # Process assessment file
            controls = assessment_processor.process_assessment_file(str(filepath))
            
            # Convert to dictionaries for JSON serialization
            controls_dict = [control.to_dict() for control in controls]
            
            # Save controls to session
            safe_json_dump(controls_dict, session_path / 'controls.json')
            
            # Save session info
            framework = request.form.get('framework', 'NIST')
            session['framework'] = framework
            session['controls_count'] = len(controls)
            
            # Create analysis session
            analysis_session = AnalysisSession(
                session_id=session_id,
                created_at=datetime.now(),
                framework=framework,
                assessment_file=filepath.name,
                controls_count=len(controls)
            )
            
            safe_json_dump(analysis_session.to_dict(), session_path / 'session.json')
            
            # Clean sample control for response
            sample_control = clean_sample_control(controls_dict[0]) if controls_dict else None
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'controls_count': len(controls),
                'sample_control': sample_control
            })
            
        except Exception as e:
            logger.error(f"Error processing assessment: {e}")
            return jsonify({'error': f'Error processing assessment: {str(e)}'}), 400
    
    @app.route('/upload_evidence', methods=['POST'])
    def upload_evidence():
        """Handle evidence files upload"""
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        session_path = Path(f"{config.SESSION_FOLDER}/{session_id}")
        evidence_path = session_path / 'evidence'
        evidence_path.mkdir(exist_ok=True)
        
        uploaded_files = []
        for key in request.files:
            file = request.files[key]
            if file and allowed_file(file.filename, config.ALLOWED_EXTENSIONS):
                filepath = secure_save_file(file, evidence_path)
                uploaded_files.append({
                    'filename': filepath.name,
                    'size': filepath.stat().st_size
                })
        
        return jsonify({
            'success': True,
            'uploaded_files': uploaded_files,
            'total_files': len(uploaded_files)
        })
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        """Run the analysis on uploaded assessment and evidence"""
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No active session'}), 400
        
        session_path = Path(f"{config.SESSION_FOLDER}/{session_id}")
        
        try:
            # Load controls
            import json
            with open(session_path / 'controls.json', 'r') as f:
                controls_data = json.load(f)
            
            controls = [ControlData.from_dict(data) for data in controls_data]
            
            # Process evidence files
            evidence_texts = {}
            evidence_path = session_path / 'evidence'
            if evidence_path.exists():
                evidence_files = list(evidence_path.iterdir())
                evidence_file_objects = evidence_processor.process_multiple_evidence_files(
                    [str(f) for f in evidence_files if f.is_file()]
                )
                evidence_texts = evidence_processor.extract_combined_evidence_text(evidence_file_objects)
            
            # Get framework and analysis parameters
            framework = session.get('framework', 'NIST')
            max_controls = min(len(controls), int(request.json.get('max_controls', config.MAX_CONTROLS_DEFAULT)))
            analysis_depth = request.json.get('analysis_depth', 'standard')  # basic, standard, comprehensive

            logger.info(f"Starting optimized analysis of {max_controls} controls with framework {framework}")

            # Use optimized workflow for systematic analysis
            results = optimized_workflow.analyze_controls_systematically(
                controls[:max_controls],
                evidence_texts,
                framework,
                analysis_depth
            )
            
            # Calculate risk metrics
            risk_metrics = reviewer.calculate_risk_metrics(results)
            
            # Convert results to dict format
            results_dict = [result.to_dict() for result in results]
            
            # Save results
            analysis_results = {
                'results': results_dict,
                'risk_metrics': risk_metrics.to_dict(),
                'framework': framework,
                'timestamp': datetime.now().isoformat(),
                'total_controls': len(controls),
                'analyzed_controls': len(results)
            }
            
            safe_json_dump(analysis_results, session_path / 'results.json', indent=2)
            
            return jsonify({
                'success': True,
                'results': results_dict,
                'risk_metrics': risk_metrics.to_dict(),
                'summary': {
                    'total_controls': len(controls),
                    'analyzed_controls': len(results),
                    'framework': framework
                }
            })
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/system_status')
    def system_status():
        """Get system and model status"""
        status = {
            'backend': 'Ollama' if reviewer.use_ollama else 'Transformers',
            'model': reviewer.model_name,
            'frameworks': list(config.FRAMEWORKS.keys()),
            'ollama_available': False
        }
        
        if reviewer.use_ollama and hasattr(reviewer.backend, 'is_available'):
            status['ollama_available'] = reviewer.backend.is_available()
            
            if status['ollama_available']:
                try:
                    import requests
                    response = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags")
                    if response.status_code == 200:
                        models = [m['name'] for m in response.json().get('models', [])]
                        status['available_models'] = models
                except:
                    status['available_models'] = []
        
        return jsonify(status)
    
    @app.route('/download_report')
    @app.route('/download_report/<session_id>')
    def download_report(session_id=None):
        """Download the analysis report"""
        if not session_id:
            session_id = session.get('session_id')
            
        if not session_id:
            return jsonify({'error': 'No session found'}), 404
        
        session_path = Path(f"{config.SESSION_FOLDER}/{session_id}")
        results_path = session_path / 'results.json'
        
        if not results_path.exists():
            return jsonify({'error': 'Report not found'}), 404
        
        try:
            # Load results
            import json
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            # Create Excel report
            report_path = session_path / 'cyber_assessment_report.xlsx'
            create_excel_report(data, report_path)
            
            return send_file(
                str(report_path), 
                as_attachment=True,
                download_name=f'cyber_assessment_report_{datetime.now().strftime("%Y%m%d")}.xlsx'
            )
            
        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return jsonify({'error': f'Failed to create report: {str(e)}'}), 500

def create_excel_report(data: dict, report_path: Path):
    """Create Excel report from analysis data"""
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # Executive Summary sheet
        summary_data = {
            'Metric': [
                'Assessment Date',
                'Framework Used',
                'Total Controls',
                'Controls Analyzed',
                'Overall Risk Score',
                'Compliance Percentage',
                'Critical Risk Controls',
                'High Risk Controls'
            ],
            'Value': [
                data['timestamp'][:10],
                data['framework'],
                data['total_controls'],
                data['analyzed_controls'],
                f"{data['risk_metrics']['overall_risk_score']}%",
                f"{data['risk_metrics']['compliance_percentage']}%",
                data['risk_metrics']['risk_distribution'].get('Critical', 0),
                data['risk_metrics']['risk_distribution'].get('High', 0)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Detailed Results sheet
        results_df = pd.DataFrame(data['results'])
        
        # Convert lists to strings for Excel
        for col in ['key_findings', 'risks', 'remediation', 'evidence_references']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: '\n'.join(x) if isinstance(x, list) else x
                )
        
        results_df.to_excel(writer, sheet_name='Detailed Analysis', index=False)
        
        # High Priority Controls sheet
        if data['risk_metrics']['high_priority_controls']:
            priority_df = pd.DataFrame(data['risk_metrics']['high_priority_controls'])
            priority_df.to_excel(writer, sheet_name='High Priority Controls', index=False)
