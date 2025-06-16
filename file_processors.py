"""
File processing module for Cyber Assessment Reviewer
Handles extraction of text from various file formats (PDF, DOCX, XLSX, PPTX)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import PyPDF2
from docx import Document
from pptx import Presentation

from models import EvidenceFile, ControlData
from config import Config

logger = logging.getLogger(__name__)

class FileProcessor:
    """Base class for file processors"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def process_file(self, filepath: str) -> Dict[str, str]:
        """Process file and return extracted text sections"""
        raise NotImplementedError("Subclasses must implement process_file method")

class PDFProcessor(FileProcessor):
    """Processor for PDF files"""
    
    def process_file(self, filepath: str) -> Dict[str, str]:
        """Extract text from PDF file with page references"""
        pages_text = {}
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = min(len(pdf_reader.pages), self.config.MAX_PAGES_PDF)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        pages_text[f"Page {page_num + 1}"] = text[:self.config.MAX_TEXT_LENGTH_PER_SECTION]
                        
        except Exception as e:
            logger.error(f"Error reading PDF {filepath}: {e}")
            return {"Error": f"Failed to read PDF: {str(e)}"}
        
        return pages_text

class DOCXProcessor(FileProcessor):
    """Processor for DOCX files"""
    
    def process_file(self, filepath: str) -> Dict[str, str]:
        """Extract text from DOCX file with section references"""
        sections_text = {}
        try:
            doc = Document(filepath)
            
            # Extract paragraphs
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            # Group into sections
            if full_text:
                section_size = max(1, len(full_text) // 5)  # Divide into ~5 sections
                for i in range(0, len(full_text), section_size):
                    section_text = '\n'.join(full_text[i:i+section_size])
                    sections_text[f"Section {i//section_size + 1}"] = section_text[:self.config.MAX_TEXT_LENGTH_PER_SECTION]
                    
        except Exception as e:
            logger.error(f"Error reading DOCX {filepath}: {e}")
            return {"Error": f"Failed to read DOCX: {str(e)}"}
        
        return sections_text

class PPTXProcessor(FileProcessor):
    """Processor for PPTX files"""
    
    def process_file(self, filepath: str) -> Dict[str, str]:
        """Extract text from PPTX file with slide references"""
        slides_text = {}
        try:
            prs = Presentation(filepath)
            
            for i, slide in enumerate(prs.slides):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_text.append(shape.text)
                
                if slide_text:
                    slides_text[f"Slide {i + 1}"] = '\n'.join(slide_text)[:self.config.MAX_TEXT_LENGTH_PER_SECTION]
                    
        except Exception as e:
            logger.error(f"Error reading PPTX {filepath}: {e}")
            return {"Error": f"Failed to read PPTX: {str(e)}"}
        
        return slides_text

class XLSXProcessor(FileProcessor):
    """Processor for XLSX files"""
    
    def process_file(self, filepath: str) -> Dict[str, str]:
        """Extract text from XLSX file with sheet references"""
        sheets_text = {}
        try:
            xlsx_file = pd.ExcelFile(filepath)
            
            for sheet_name in xlsx_file.sheet_names[:5]:  # Limit to first 5 sheets
                df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
                
                # Handle NaN values by replacing with empty strings
                df = df.fillna('')
                
                # Convert to string representation
                text_content = f"Sheet: {sheet_name}\n"
                text_content += f"Columns: {', '.join(str(col) for col in df.columns)}\n"
                text_content += f"Rows: {len(df)}\n\n"
                
                # Add sample data
                if len(df) > 0:
                    text_content += df.head(10).to_string()
                    
                sheets_text[sheet_name] = text_content[:self.config.MAX_TEXT_LENGTH_PER_SECTION]
                
        except Exception as e:
            logger.error(f"Error reading XLSX {filepath}: {e}")
            return {"Error": f"Failed to read XLSX: {str(e)}"}
        
        return sheets_text

class AssessmentFileProcessor:
    """Specialized processor for assessment files (typically XLSX)"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def process_assessment_file(self, filepath: str) -> List[ControlData]:
        """Process assessment file and extract control data"""
        try:
            df = pd.read_excel(filepath)
            
            # Standardize column names using flexible mapping
            self._standardize_columns(df)
            
            # Handle NaN values before converting to dict
            df = df.fillna('')
            
            # Convert to ControlData objects
            controls = []
            for _, row in df.iterrows():
                try:
                    control = ControlData.from_dict(row.to_dict())
                    controls.append(control)
                except Exception as e:
                    logger.warning(f"Error processing control row: {e}")
                    continue
            
            logger.info(f"Processed {len(controls)} controls from assessment file")
            return controls
            
        except Exception as e:
            logger.error(f"Error processing assessment file {filepath}: {e}")
            raise
    
    def _standardize_columns(self, df: pd.DataFrame) -> None:
        """Standardize column names using flexible mapping"""
        for standard_name, possible_names in self.config.COLUMN_MAPPING.items():
            for col in df.columns:
                if col.lower().strip() in [name.lower() for name in possible_names]:
                    df.rename(columns={col: standard_name}, inplace=True)
                    break

class FileProcessorFactory:
    """Factory class for creating appropriate file processors"""
    
    _processors = {
        '.pdf': PDFProcessor,
        '.docx': DOCXProcessor,
        '.pptx': PPTXProcessor,
        '.ppt': PPTXProcessor,
        '.xlsx': XLSXProcessor
    }
    
    @classmethod
    def get_processor(cls, file_extension: str, config: Config = None) -> Optional[FileProcessor]:
        """Get appropriate processor for file extension"""
        processor_class = cls._processors.get(file_extension.lower())
        if processor_class:
            return processor_class(config)
        return None
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions"""
        return list(cls._processors.keys())

class EvidenceProcessor:
    """Main processor for evidence files"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def process_evidence_file(self, filepath: str) -> EvidenceFile:
        """Process evidence file and return EvidenceFile object"""
        file_path = Path(filepath)
        file_extension = file_path.suffix.lower()
        
        # Get appropriate processor
        processor = FileProcessorFactory.get_processor(file_extension, self.config)
        if not processor:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Process file
        sections = processor.process_file(str(file_path))
        
        # Create EvidenceFile object
        evidence_file = EvidenceFile(
            filename=file_path.name,
            filepath=str(file_path),
            file_type=file_extension,
            size_bytes=file_path.stat().st_size,
            sections=sections
        )
        
        logger.info(f"Processed evidence file: {file_path.name} ({len(sections)} sections)")
        return evidence_file
    
    def process_multiple_evidence_files(self, filepaths: List[str]) -> Dict[str, EvidenceFile]:
        """Process multiple evidence files"""
        evidence_files = {}
        
        for filepath in filepaths:
            try:
                evidence_file = self.process_evidence_file(filepath)
                evidence_files[evidence_file.filename] = evidence_file
            except Exception as e:
                logger.error(f"Error processing evidence file {filepath}: {e}")
                continue
        
        return evidence_files
    
    def extract_combined_evidence_text(self, evidence_files: Dict[str, EvidenceFile], 
                                     max_sections: int = None) -> Dict[str, Dict[str, str]]:
        """Extract combined evidence text in the format expected by the AI backend"""
        max_sections = max_sections or self.config.MAX_EVIDENCE_SECTIONS
        
        combined_evidence = {}
        for filename, evidence_file in evidence_files.items():
            # Filter out error sections and limit sections
            valid_sections = {
                section: text for section, text in evidence_file.sections.items()
                if "Error" not in section and text.strip()
            }
            
            # Limit number of sections
            if len(valid_sections) > max_sections:
                valid_sections = dict(list(valid_sections.items())[:max_sections])
            
            combined_evidence[filename] = valid_sections
        
        return combined_evidence

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported"""
    file_extension = Path(filename).suffix.lower()
    return file_extension in FileProcessorFactory.get_supported_extensions()
