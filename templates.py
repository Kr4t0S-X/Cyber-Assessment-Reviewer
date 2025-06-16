"""
HTML template generation for Cyber Assessment Reviewer
Contains HTML templates and template generation functions
"""

def get_html_template(backend: str) -> str:
    """Return the HTML template for the main page"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cyber Assessment Reviewer - {backend} Backend</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        
        .backend-indicator {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        @media (max-width: 1024px) {{
            .main-content {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .card h2 {{
            color: #5a67d8;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }}
        
        .upload-area {{
            border: 3px dashed #cbd5e0;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 20px;
        }}
        
        .upload-area:hover {{
            border-color: #5a67d8;
            background-color: #f7fafc;
        }}
        
        .upload-area.dragover {{
            border-color: #5a67d8;
            background-color: #ebf4ff;
        }}
        
        .upload-icon {{
            font-size: 3rem;
            color: #a0aec0;
            margin-bottom: 15px;
        }}
        
        .file-input {{
            display: none;
        }}
        
        .btn {{
            background: linear-gradient(135deg, #5a67d8, #667eea);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(90,103,216,0.3);
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(90,103,216,0.4);
        }}
        
        .btn:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }}
        
        .progress-container {{
            margin-top: 20px;
            display: none;
        }}
        
        .progress-bar {{
            background-color: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            height: 10px;
            margin-bottom: 10px;
        }}
        
        .progress-fill {{
            background: linear-gradient(90deg, #5a67d8, #667eea);
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        
        .status-message {{
            text-align: center;
            font-weight: 500;
            color: #4a5568;
        }}
        
        .file-list {{
            margin-top: 15px;
        }}
        
        .file-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            background-color: #f7fafc;
            border-radius: 8px;
            margin-bottom: 8px;
        }}
        
        .file-icon {{
            margin-right: 10px;
            color: #5a67d8;
        }}
        
        .framework-selector {{
            margin-bottom: 20px;
        }}
        
        .framework-selector label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #4a5568;
        }}
        
        .framework-selector select {{
            width: 100%;
            padding: 10px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1rem;
            background-color: white;
        }}
        
        .framework-selector select:focus {{
            outline: none;
            border-color: #5a67d8;
        }}
        
        .results-section {{
            grid-column: 1 / -1;
            margin-top: 20px;
        }}
        
        .results-container {{
            display: none;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .results-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e2e8f0;
        }}
        
        .results-title {{
            font-size: 1.8rem;
            color: #2d3748;
        }}
        
        .download-btn {{
            background: linear-gradient(135deg, #48bb78, #68d391);
            box-shadow: 0 4px 15px rgba(72,187,120,0.3);
        }}
        
        .download-btn:hover {{
            box-shadow: 0 6px 20px rgba(72,187,120,0.4);
        }}
        
        .analysis-grid {{
            display: grid;
            gap: 20px;
        }}
        
        .control-analysis {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #5a67d8;
        }}
        
        .control-header {{
            display: flex;
            justify-content: between;
            align-items: flex-start;
            margin-bottom: 15px;
        }}
        
        .control-title {{
            font-size: 1.2rem;
            color: #2d3748;
            margin-bottom: 5px;
        }}
        
        .confidence-score {{
            background: linear-gradient(135deg, #ed8936, #f6ad55);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-left: auto;
        }}
        
        .risk-level {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .risk-critical {{
            background-color: #fed7d7;
            color: #c53030;
        }}
        
        .risk-high {{
            background-color: #fed7d7;
            color: #c53030;
        }}
        
        .risk-medium {{
            background-color: #feebc8;
            color: #dd6b20;
        }}
        
        .risk-low {{
            background-color: #c6f6d5;
            color: #25855a;
        }}
        
        .analysis-content {{
            margin-bottom: 15px;
        }}
        
        .evidence-section {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }}
        
        .evidence-title {{
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 8px;
        }}
        
        .evidence-item {{
            background: #edf2f7;
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 6px;
            font-size: 0.9rem;
        }}
        
        .remediation {{
            background: #e6fffa;
            border: 1px solid #38b2ac;
            border-radius: 8px;
            padding: 15px;
            margin-top: 10px;
        }}
        
        .remediation-title {{
            color: #2c7a7b;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        
        .alert {{
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid;
        }}
        
        .alert-success {{
            background-color: #f0fff4;
            border-color: #9ae6b4;
            color: #2f855a;
        }}
        
        .alert-error {{
            background-color: #fed7d7;
            border-color: #fc8181;
            color: #c53030;
        }}
        
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #5a67d8;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .hidden {{
            display: none !important;
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-in-out;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Cyber Assessment Reviewer</h1>
            <div class="subtitle">AI-Powered Cybersecurity Control Analysis</div>
            <div class="backend-indicator">
                🧠 Backend: {backend}
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="main-content">
            <!-- Assessment Upload Section -->
            <div class="card">
                <h2>📋 Upload Assessment</h2>
                <form id="assessmentForm" enctype="multipart/form-data">
                    <div class="framework-selector">
                        <label for="framework">Cybersecurity Framework:</label>
                        <select id="framework" name="framework" required>
                            <option value="">Select Framework</option>
                            <option value="NIST">NIST Cybersecurity Framework</option>
                            <option value="ISO27001">ISO 27001</option>
                            <option value="SOC2">SOC 2</option>
                            <option value="CIS">CIS Controls</option>
                            <option value="PCI-DSS">PCI DSS</option>
                        </select>
                    </div>

                    <div class="upload-area" id="assessmentUpload">
                        <div class="upload-icon">📄</div>
                        <h3>Drop assessment file here</h3>
                        <p>Or click to select file</p>
                        <p><small>Supports: PDF, DOCX, XLSX, PPTX</small></p>
                        <input type="file" id="assessmentFile" name="assessment" class="file-input"
                               accept=".pdf,.docx,.xlsx,.pptx" required>
                    </div>

                    <div id="assessmentFiles" class="file-list"></div>
                </form>
            </div>

            <!-- Evidence Upload Section -->
            <div class="card">
                <h2>🗂️ Upload Evidence</h2>
                <form id="evidenceForm" enctype="multipart/form-data">
                    <div class="upload-area" id="evidenceUpload">
                        <div class="upload-icon">📁</div>
                        <h3>Drop evidence files here</h3>
                        <p>Or click to select files</p>
                        <p><small>Multiple files supported</small></p>
                        <input type="file" id="evidenceFiles" name="evidence" class="file-input"
                               accept=".pdf,.docx,.xlsx,.pptx" multiple>
                    </div>

                    <div id="evidenceFileList" class="file-list"></div>

                    <button type="button" id="analyzeBtn" class="btn" disabled>
                        🔍 Analyze Assessment
                    </button>

                    <div class="progress-container" id="progressContainer">
                        <div class="progress-bar">
                            <div class="progress-fill" id="progressFill">0%</div>
                        </div>
                        <div class="status-message" id="progressText">Starting analysis...</div>
                    </div>
                </form>
            </div>
        </div>

        <!-- Results Section -->
        <div class="results-section">
            <div class="results-container" id="resultsContainer">
                <div class="results-header">
                    <h2 class="results-title">Analysis Results</h2>
                    <button type="button" id="downloadBtn" class="btn download-btn">
                        📥 Download Report
                    </button>
                </div>
                <div id="analysisResults" class="analysis-grid"></div>
            </div>
        </div>

        <!-- Alert Area -->
        <div id="alertArea"></div>
    </div>

    <script>
        let assessmentUploaded = false;
        let evidenceUploaded = false;

        // File upload handling
        function setupFileUpload(uploadAreaId, fileInputId, fileListId, isMultiple = false) {{
            const uploadArea = document.getElementById(uploadAreaId);
            const fileInput = document.getElementById(fileInputId);
            const fileList = document.getElementById(fileListId);

            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('dragover', (e) => {{
                e.preventDefault();
                uploadArea.classList.add('dragover');
            }});

            uploadArea.addEventListener('dragleave', () => {{
                uploadArea.classList.remove('dragover');
            }});

            uploadArea.addEventListener('drop', (e) => {{
                e.preventDefault();
                uploadArea.classList.remove('dragover');

                const files = Array.from(e.dataTransfer.files);
                handleFiles(files, fileInput, fileList, isMultiple);
            }});

            fileInput.addEventListener('change', (e) => {{
                const files = Array.from(e.target.files);
                handleFiles(files, fileInput, fileList, isMultiple);
            }});
        }}

        function handleFiles(files, fileInput, fileList, isMultiple) {{
            if (!isMultiple && files.length > 1) {{
                showAlert('Please select only one file.', 'error');
                return;
            }}

            // Clear existing files if single file mode
            if (!isMultiple) {{
                fileList.innerHTML = '';
            }}

            files.forEach(file => {{
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <span class="file-icon">📄</span>
                    <span>${{file.name}} (${{(file.size / 1024 / 1024).toFixed(2)}} MB)</span>
                `;
                fileList.appendChild(fileItem);
            }});

            // Update upload status
            if (fileInput.id === 'assessmentFile') {{
                assessmentUploaded = files.length > 0;
                uploadAssessment(files[0]);
            }} else {{
                evidenceUploaded = files.length > 0;
                uploadEvidence(files);
            }}

            updateAnalyzeButton();
        }}

        function updateAnalyzeButton() {{
            const analyzeBtn = document.getElementById('analyzeBtn');
            const framework = document.getElementById('framework').value;
            analyzeBtn.disabled = !(assessmentUploaded && framework);
        }}

        async function uploadAssessment(file) {{
            const formData = new FormData();
            const framework = document.getElementById('framework').value;

            if (!framework) {{
                showAlert('Please select a cybersecurity framework first.', 'error');
                return;
            }}

            formData.append('assessment', file);
            formData.append('framework', framework);

            try {{
                const response = await fetch('/upload_assessment', {{
                    method: 'POST',
                    body: formData
                }});

                const result = await response.json();

                if (result.success) {{
                    showAlert('Assessment uploaded successfully!', 'success');
                }} else {{
                    showAlert('Error uploading assessment: ' + result.error, 'error');
                    assessmentUploaded = false;
                    updateAnalyzeButton();
                }}
            }} catch (error) {{
                showAlert('Error uploading assessment: ' + error.message, 'error');
                assessmentUploaded = false;
                updateAnalyzeButton();
            }}
        }}

        async function uploadEvidence(files) {{
            const formData = new FormData();

            files.forEach(file => {{
                formData.append('evidence', file);
            }});

            try {{
                const response = await fetch('/upload_evidence', {{
                    method: 'POST',
                    body: formData
                }});

                const result = await response.json();

                if (result.success) {{
                    showAlert(`${{files.length}} evidence file(s) uploaded successfully!`, 'success');
                }} else {{
                    showAlert('Error uploading evidence: ' + result.error, 'error');
                    evidenceUploaded = false;
                    updateAnalyzeButton();
                }}
            }} catch (error) {{
                showAlert('Error uploading evidence: ' + error.message, 'error');
                evidenceUploaded = false;
                updateAnalyzeButton();
            }}
        }}

        async function analyzeAssessment() {{
            const progressContainer = document.getElementById('progressContainer');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const resultsContainer = document.getElementById('resultsContainer');

            progressContainer.style.display = 'block';
            analyzeBtn.disabled = true;
            resultsContainer.style.display = 'none';

            try {{
                const response = await fetch('/analyze', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        max_controls: 20
                    }})
                }});

                const result = await response.json();

                if (result.success) {{
                    displayResults(result);
                    showAlert('Analysis completed successfully!', 'success');
                }} else {{
                    showAlert('Error during analysis: ' + result.error, 'error');
                }}
            }} catch (error) {{
                showAlert('Error during analysis: ' + error.message, 'error');
            }} finally {{
                progressContainer.style.display = 'none';
                analyzeBtn.disabled = false;
            }}
        }}

        function displayResults(result) {{
            const resultsContainer = document.getElementById('resultsContainer');
            const analysisResults = document.getElementById('analysisResults');

            // Clear previous results
            analysisResults.innerHTML = '';

            // Display each control analysis
            result.results.forEach(control => {{
                const controlDiv = document.createElement('div');
                controlDiv.className = 'control-analysis fade-in';

                const riskClass = `risk-${{control.risk_level.toLowerCase()}}`;

                controlDiv.innerHTML = `
                    <div class="control-header">
                        <div>
                            <div class="control-title">${{control.control_id}}: ${{control.control_name}}</div>
                            <div class="risk-level ${{riskClass}}">${{control.risk_level}} Risk</div>
                        </div>
                        <div class="confidence-score">${{(control.confidence_score * 100).toFixed(0)}}% Confidence</div>
                    </div>

                    <div class="analysis-content">
                        <p><strong>Compliance Status:</strong> ${{control.compliance_status}}</p>
                        <p><strong>Evidence Validity:</strong> ${{control.evidence_validity}}</p>
                    </div>

                    <div class="evidence-section">
                        <div class="evidence-title">Key Findings</div>
                        ${{control.key_findings.map(finding => `<div class="evidence-item">${{finding}}</div>`).join('')}}
                    </div>

                    <div class="evidence-section">
                        <div class="evidence-title">Identified Risks</div>
                        ${{control.risks.map(risk => `<div class="evidence-item">${{risk}}</div>`).join('')}}
                    </div>

                    <div class="remediation">
                        <div class="remediation-title">Recommended Actions</div>
                        ${{control.remediation.map(action => `<div class="evidence-item">${{action}}</div>`).join('')}}
                    </div>
                `;

                analysisResults.appendChild(controlDiv);
            }});

            resultsContainer.style.display = 'block';
        }}

        function showAlert(message, type) {{
            const alertArea = document.getElementById('alertArea');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${{type}} fade-in`;
            alertDiv.textContent = message;

            alertArea.appendChild(alertDiv);

            // Remove alert after 5 seconds
            setTimeout(() => {{
                alertDiv.remove();
            }}, 5000);
        }}

        // Initialize file upload handlers
        document.addEventListener('DOMContentLoaded', () => {{
            setupFileUpload('assessmentUpload', 'assessmentFile', 'assessmentFiles', false);
            setupFileUpload('evidenceUpload', 'evidenceFiles', 'evidenceFileList', true);

            // Framework change handler
            document.getElementById('framework').addEventListener('change', updateAnalyzeButton);

            // Analyze button handler
            document.getElementById('analyzeBtn').addEventListener('click', analyzeAssessment);

            // Download button handler
            document.getElementById('downloadBtn').addEventListener('click', () => {{
                window.location.href = '/download_report';
            }});
        }});
    </script>
</body>
</html>"""
