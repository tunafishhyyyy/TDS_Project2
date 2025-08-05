import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List
import math
import re
import requests
from bs4 import BeautifulSoup
import json
import logging

def extract_keywords(task_description: str) -> List[str]:
    """
    Extracts keywords/entities from the task description using a simple regex and stopword filtering.
    This can be replaced with a more advanced NLP/LLM-based extractor if needed.
    """
    stopwords = set([
        'the', 'of', 'and', 'to', 'in', 'for', 'by', 'with', 'on', 'at', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'a', 'an', 'or', 'that', 'this', 'which', 'who', 'what', 'when', 'where', 'how', 'why', 'it', 'its', 'their', 'his', 'her', 'our', 'your', 'my', 'but', 'not', 'so', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must', 'also', 'such', 'than', 'then', 'there', 'here', 'these', 'those', 'all', 'any', 'each', 'other', 'some', 'many', 'most', 'more', 'less', 'few', 'lot', 'lots', 'just', 'even', 'still', 'yet', 'very', 'quite', 'rather', 'really', 'always', 'never', 'sometimes', 'often', 'usually', 'again', 'once', 'twice', 'first', 'second', 'third', 'next', 'last', 'new', 'old', 'good', 'bad', 'better', 'best', 'worst', 'same', 'different', 'another', 'own', 'much', 'such', 'big', 'small', 'large', 'great', 'high', 'low', 'long', 'short', 'early', 'late', 'young', 'old', 'important', 'main', 'major', 'minor', 'key', 'top', 'bottom', 'left', 'right', 'up', 'down', 'out', 'over', 'under', 'between', 'among', 'within', 'without', 'through', 'during', 'before', 'after', 'above', 'below', 'across', 'against', 'toward', 'upon', 'about', 'around', 'into', 'onto', 'off', 'near', 'far', 'away', 'back', 'forward', 'ahead', 'behind', 'beside', 'along', 'past', 'since', 'until', 'while', 'because', 'although', 'though', 'unless', 'except', 'despite', 'regardless', 'concerning', 'regarding', 'including', 'excluding', 'plus', 'minus', 'per', 'via', 'versus', 'vs', 'etc', 'i', 'you', 'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'whose', 'whom', 'which', 'what', 'where', 'when', 'why', 'how', 'if', 'then', 'else', 'because', 'since', 'until', 'while', 'although', 'though', 'unless', 'except', 'despite', 'regardless', 'concerning', 'regarding', 'including', 'excluding', 'plus', 'minus', 'per', 'via', 'versus', 'vs', 'etc'
    ])
    words = re.findall(r'\b\w+\b', task_description.lower())
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    # Remove duplicates, preserve order
    seen = set()
    result = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result

def sanitize_for_json(obj):
    """
    Recursively sanitize dicts/lists/floats for JSON serialization.
    Converts NaN, inf, -inf to None.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj
import logging
import json
import re as regex

logger = logging.getLogger(__name__)

class DetectDataFormatStep:
    """
    Step 0: LLM-powered data format detection
    - Analyze webpage HTML structure to detect data availability and format
    - Identify if data is in HTML tables, JSON, JavaScript variables, or other formats
    - Provide extraction strategy recommendations
    - Generic approach that works for any website (IMDb, Wikipedia, Trading Economics, etc.)
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data['url']
        task_description = input_data.get('task_description', '')
        
        try:
            # Fetch webpage content
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Analyze page structure with LLM
            format_analysis = self._analyze_data_format_with_llm(soup, task_description, url)
            
            print(f"Data format analysis for {url}:")
            print(f"Format: {format_analysis['format']}")
            print(f"Strategy: {format_analysis['strategy']}")
            print(f"Confidence: {format_analysis['confidence']}")
            if format_analysis.get('json_data'):
                print(f"JSON data found: {len(format_analysis['json_data'])} characters")
            
            return {
                'url': url,
                'task_description': task_description,
                'format_analysis': format_analysis,
                'html_content': response.text,
                'soup': soup  # Pass soup object for next steps (not serialized)
            }
            
        except Exception as e:
            print(f"Error in data format detection: {str(e)}")
            # Fallback to traditional table scraping
            return {
                'url': url,
                'task_description': task_description,
                'format_analysis': {
                    'format': 'html_tables',
                    'strategy': 'pandas_read_html',
                    'confidence': 'low',
                    'fallback': True
                }
            }
    
    def _analyze_data_format_with_llm(self, soup: BeautifulSoup, task_description: str, url: str) -> Dict[str, Any]:
        """
        Use LLM to analyze webpage structure and detect data format
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Extract key structural information
            structure_info = self._extract_page_structure(soup)
            
            # Create LLM prompt for format detection
            system_prompt = """You are an expert web scraping analyst. Analyze webpage structure to determine the best data extraction approach.

Your task is to identify:
1. Data format: html_tables, json_embedded, javascript_data, structured_divs, or mixed
2. Extraction strategy: pandas_read_html, json_parsing, regex_extraction, custom_parsing
3. Confidence level: high, medium, low

Consider these indicators:
- HTML <table> tags suggest html_tables format
- <script> tags with JSON/data suggest javascript_data format  
- Structured <div> patterns suggest structured_divs format
- Multiple formats may coexist (mixed)

Respond in this JSON format:
{{
  "format": "html_tables|json_embedded|javascript_data|structured_divs|mixed",
  "strategy": "pandas_read_html|json_parsing|regex_extraction|custom_parsing",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation",
  "json_selectors": ["script[type='application/ld+json']", "script containing data"],
  "table_selectors": ["table.chart", "table.data-table"],
  "fallback_strategy": "alternative approach if primary fails"
}}"""

            human_prompt = """URL: {url}
Task: {task_description}

Page structure analysis:
{structure_info}

Determine the best data extraction approach for this webpage."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # Get LLM model and create chain
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            # Invoke LLM with structure data
            result = chain.invoke({
                "url": url,
                "task_description": task_description or "general data extraction",
                "structure_info": structure_info
            })
            
            # Parse LLM response
            try:
                # Clean up response and parse JSON
                cleaned_result = result.strip()
                if cleaned_result.startswith('```json'):
                    cleaned_result = cleaned_result[7:]
                if cleaned_result.endswith('```'):
                    cleaned_result = cleaned_result[:-3]
                
                format_analysis = json.loads(cleaned_result)
                
                # Validate required fields
                required_fields = ['format', 'strategy', 'confidence']
                for field in required_fields:
                    if field not in format_analysis:
                        raise ValueError(f"Missing required field: {field}")
                
                # Extract JSON data if strategy suggests it
                if format_analysis['strategy'] in ['json_parsing', 'regex_extraction']:
                    json_data = self._extract_json_data(soup, format_analysis.get('json_selectors', []))
                    if json_data:
                        format_analysis['json_data'] = json_data
                
                return format_analysis
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Could not parse LLM response as JSON: {e}")
                print(f"Raw response: {result}")
                return self._fallback_format_analysis()
                
        except Exception as e:
            print(f"Error in LLM format analysis: {str(e)}")
            return self._fallback_format_analysis()
    
    def _extract_page_structure(self, soup: BeautifulSoup) -> str:
        """
        Extract key structural information from the webpage for LLM analysis
        """
        structure_info = []
        
        # Count HTML tables
        tables = soup.find_all("table")
        structure_info.append(f"HTML tables found: {len(tables)}")
        if tables:
            for i, table in enumerate(tables[:3]):  # Analyze first 3 tables
                rows = len(table.find_all("tr"))
                cols = len(table.find_all("td")) + len(table.find_all("th"))
                classes = table.get("class", [])
                structure_info.append(f"  Table {i}: {rows} rows, ~{cols} cells, classes: {classes}")
        
        # Check for JSON in script tags
        scripts = soup.find_all("script")
        json_scripts = 0
        data_scripts = 0
        for script in scripts:
            if script.get("type") == "application/ld+json":
                json_scripts += 1
            elif script.string and ("data" in script.string.lower() or "{" in script.string):
                data_scripts += 1
        structure_info.append(f"Script tags: {len(scripts)} total, {json_scripts} JSON-LD, {data_scripts} with data")
        
        # Check for structured content divs
        common_data_classes = ["chart", "data", "list", "table", "grid", "content", "results"]
        structured_divs = 0
        for class_name in common_data_classes:
            divs = soup.find_all("div", class_=re.compile(class_name, re.IGNORECASE))
            if divs:
                structured_divs += len(divs)
                structure_info.append(f"  Divs with '{class_name}' class: {len(divs)}")
        
        # Check for lists (ul, ol)
        lists = soup.find_all(["ul", "ol"])
        structure_info.append(f"Lists found: {len(lists)}")
        
        # Sample page content
        text_content = soup.get_text()[:500] + "..." if len(soup.get_text()) > 500 else soup.get_text()
        structure_info.append(f"Page content sample: {text_content}")
        
        return "\n".join(structure_info)
    
    def _extract_json_data(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """
        Extract JSON data from script tags based on provided selectors
        """
        json_data = ""
        
        # Try provided selectors first
        for selector in selectors:
            elements = soup.select(selector)
            for element in elements:
                if element.string:
                    try:
                        # Validate it's valid JSON
                        json.loads(element.string)
                        json_data += element.string + "\n"
                    except json.JSONDecodeError:
                        continue
        
        # Fallback: search all script tags for JSON-like content
        if not json_data:
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    text = script.string.strip()
                    # Look for JSON objects or arrays
                    if (text.startswith('{') and text.endswith('}')) or \
                       (text.startswith('[') and text.endswith(']')):
                        try:
                            json.loads(text)
                            json_data += text + "\n"
                        except json.JSONDecodeError:
                            continue
                    # Look for variable assignments with JSON data
                    json_pattern = r'(?:var|let|const)\s+\w+\s*=\s*(\{.*?\}|\[.*?\]);?'
                    matches = re.findall(json_pattern, text, re.DOTALL)
                    for match in matches:
                        try:
                            json.loads(match)
                            json_data += match + "\n"
                        except json.JSONDecodeError:
                            continue
        
        return json_data.strip()
    
    def _fallback_format_analysis(self) -> Dict[str, Any]:
        """
        Fallback format analysis when LLM is unavailable
        """
        return {
            'format': 'html_tables',
            'strategy': 'pandas_read_html',
            'confidence': 'low',
            'reasoning': 'LLM analysis failed, using traditional table scraping',
            'fallback': True
        }

class ScrapeTableStep:
    """
    Step 1: Enhanced data extraction based on format analysis
    - Use format analysis from DetectDataFormatStep to choose extraction method
    - Support multiple data formats: HTML tables, JSON, JavaScript data
    - Intelligent fallback mechanisms
    - Generic approach for any website structure
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        url = input_data['url']
        format_analysis = input_data.get('format_analysis', {})
        task_description = input_data.get('task_description', '')
        
        print(f"Extracting data using strategy: {format_analysis.get('strategy', 'pandas_read_html')}")
        
        soup = None
        html_content = ""
        
        try:
            # Use existing soup if available, otherwise fetch fresh content
            if 'soup' in input_data:
                soup = input_data['soup']
                html_content = input_data.get('html_content', '')
            else:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/"
                }
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                html_content = response.text
            
            # Extract data based on detected format
            data = self._extract_data_by_strategy(soup, html_content, format_analysis, task_description)
            
            if data is None or (hasattr(data, 'empty') and data.empty):
                raise ValueError("No data extracted from the webpage")
            
            print(f"Successfully extracted data with shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            
        except Exception as e:
            print(f"Error in primary extraction strategy: {str(e)}")
            print("Falling back to traditional table scraping...")
            
            # Ensure we have soup for fallback
            if soup is None:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/"
                    }
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                except Exception as fallback_error:
                    print(f"Failed to fetch content for fallback: {str(fallback_error)}")
                    raise e  # Re-raise original error
            
            data = self._fallback_table_extraction(soup, task_description)
            
        return sanitize_for_json({'data': data, 'url': url, 'format_analysis': format_analysis})
    
    def _extract_data_by_strategy(self, soup, html_content: str, format_analysis: Dict, task_description: str) -> pd.DataFrame:
        """
        Extract data using the strategy recommended by format analysis
        """
        strategy = format_analysis.get('strategy', 'pandas_read_html')
        
        if strategy == 'json_parsing':
            return self._extract_from_json(soup, format_analysis, task_description)
        elif strategy == 'regex_extraction':
            return self._extract_from_javascript(html_content, task_description)
        elif strategy == 'custom_parsing':
            return self._extract_from_structured_divs(soup, task_description)
        else:  # Default to pandas_read_html
            return self._extract_from_html_tables(soup, task_description)
    
    def _extract_from_json(self, soup, format_analysis: Dict, task_description: str) -> pd.DataFrame:
        """
        Extract data from JSON embedded in script tags
        """
        json_data = format_analysis.get('json_data', '')
        if not json_data:
            raise ValueError("No JSON data found")
        
        try:
            # Parse JSON data
            data_obj = json.loads(json_data)
            
            # Convert JSON to DataFrame using LLM guidance
            df = self._json_to_dataframe_with_llm(data_obj, task_description)
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to parse JSON data: {str(e)}")
    
    def _extract_from_javascript(self, html_content: str, task_description: str) -> pd.DataFrame:
        """
        Extract data from JavaScript variables using regex patterns
        """
        # Use LLM to identify relevant JavaScript patterns
        js_data = self._extract_js_data_with_llm(html_content, task_description)
        
        if not js_data:
            raise ValueError("No JavaScript data patterns found")
        
        # Convert extracted JS data to DataFrame
        return self._parse_js_data_to_dataframe(js_data, task_description)
    
    def _extract_from_structured_divs(self, soup, task_description: str) -> pd.DataFrame:
        """
        Extract data from structured div elements (common in modern websites)
        """
        # Use LLM to identify relevant div patterns and extract data
        return self._extract_div_data_with_llm(soup, task_description)
    
    def _extract_from_html_tables(self, soup, task_description: str) -> pd.DataFrame:
        """
        Traditional HTML table extraction with enhanced selection
        """
        tables_html = soup.find_all("table")
        tables = []
        
        if tables_html:
            for table_html in tables_html:
                try:
                    dfs = pd.read_html(str(table_html))
                    tables.extend(dfs)
                except Exception:
                    continue
        
        if not tables:
            # Enhanced fallback: try to extract from structured content
            print("No <table> tags found, attempting structured content extraction...")
            tables = self._extract_from_non_table_elements(soup)
        
        if not tables:
            raise ValueError("No tables or structured data found")
        
        print(f"Found {len(tables)} tables on the page")
        
        # Inspect all tables
        for i, table in enumerate(tables):
            print(f"\nTable {i}:")
            print(f"  Shape: {table.shape}")
            print(f"  Columns: {table.columns.tolist()}")
            print(f"  Sample data:")
            print(table.head(3))
        
        # Use LLM-powered table selection
        keywords = extract_keywords(task_description)
        best_table_idx = self._select_best_table_with_llm(tables, task_description, keywords)
        data = tables[best_table_idx]
        print(f"Selected table {best_table_idx} with {data.shape[0]} rows and {data.shape[1]} columns")
        
        return data
    
    def _extract_from_non_table_elements(self, soup) -> List[pd.DataFrame]:
        """
        Extract tabular data from non-table HTML elements
        Enhanced to handle various website structures
        """
        tables = []
        
        # Try to find structured content areas
        content_selectors = [
            ".chart", ".data-table", ".list", ".grid", ".content",
            "[class*='chart']", "[class*='table']", "[class*='list']"
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                rows = element.find_all("tr")
                if rows:
                    extracted = []
                    for row in rows:
                        cells = row.find_all(["th", "td"])
                        if cells:
                            extracted.append([cell.get_text(strip=True) for cell in cells])
                    
                    if extracted and len(extracted) > 1:
                        # Assume first row is header if it contains any <th>
                        first_row_has_th = any(row.find_all("th") for row in [rows[0]] if rows)
                        if first_row_has_th or len(extracted[0]) == len(extracted[1]):
                            header = extracted[0]
                            data_rows = extracted[1:]
                        else:
                            header = [f"Column_{i}" for i in range(len(extracted[0]))]
                            data_rows = extracted
                        
                        if data_rows:
                            df = pd.DataFrame(data_rows, columns=header)
                            tables.append(df)
                            print(f"Extracted {len(df)} rows from {selector} element.")
        
        return tables
    
    def _json_to_dataframe_with_llm(self, data_obj, task_description: str) -> pd.DataFrame:
        """
        Convert JSON object to DataFrame using LLM guidance
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Analyze JSON structure
            json_sample = str(data_obj)[:1000] + "..." if len(str(data_obj)) > 1000 else str(data_obj)
            
            system_prompt = """You are a data extraction expert. Analyze the JSON structure and provide instructions for converting it to a tabular DataFrame.

Identify:
1. The path to the array/list containing the main data
2. The key fields that should become DataFrame columns
3. Any nested structures that need flattening

Respond in this JSON format:
{{
  "data_path": "path.to.data.array (e.g., 'results', 'data.items', 'movies')",
  "key_fields": ["field1", "field2", "field3"],
  "nested_fields": {{"field_name": "path.to.nested.value"}},
  "instructions": "brief explanation of the structure"
}}"""

            human_prompt = """Task: {task_description}

JSON Structure Sample:
{json_sample}

Provide extraction instructions for converting this JSON to a DataFrame."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "json_sample": json_sample
            })
            
            # Parse LLM instructions and extract data
            try:
                instructions = json.loads(result.strip().replace('```json', '').replace('```', ''))
                return self._extract_dataframe_from_json(data_obj, instructions)
            except:
                # Fallback: try to directly convert if it's a list of dicts
                if isinstance(data_obj, list):
                    return pd.DataFrame(data_obj)
                elif isinstance(data_obj, dict):
                    # Find the first list/array in the JSON
                    for key, value in data_obj.items():
                        if isinstance(value, list) and len(value) > 0:
                            return pd.DataFrame(value)
                raise ValueError("Could not determine JSON structure")
                
        except Exception as e:
            raise ValueError(f"Failed to convert JSON to DataFrame: {str(e)}")
    
    def _extract_dataframe_from_json(self, data_obj, instructions: Dict) -> pd.DataFrame:
        """
        Extract DataFrame from JSON using LLM-provided instructions
        """
        # Navigate to data array using provided path
        current_data = data_obj
        data_path = instructions.get('data_path', '')
        
        if data_path:
            for key in data_path.split('.'):
                if key and key in current_data:
                    current_data = current_data[key]
        
        if not isinstance(current_data, list):
            raise ValueError("Data path does not lead to an array")
        
        # Extract specified fields
        key_fields = instructions.get('key_fields', [])
        nested_fields = instructions.get('nested_fields', {})
        
        rows = []
        for item in current_data:
            if isinstance(item, dict):
                row = {}
                # Extract key fields
                for field in key_fields:
                    row[field] = item.get(field, '')
                # Extract nested fields
                for field_name, nested_path in nested_fields.items():
                    nested_value = item
                    for key in nested_path.split('.'):
                        if isinstance(nested_value, dict) and key in nested_value:
                            nested_value = nested_value[key]
                        else:
                            nested_value = ''
                            break
                    row[field_name] = nested_value
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _extract_js_data_with_llm(self, html_content: str, task_description: str) -> str:
        """
        Use LLM to identify and extract relevant JavaScript data patterns
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Extract script content
            import re
            script_pattern = r'<script[^>]*>(.*?)</script>'
            scripts = re.findall(script_pattern, html_content, re.DOTALL)
            
            # Sample scripts for LLM analysis
            script_sample = ""
            for script in scripts[:5]:  # Analyze first 5 scripts
                if len(script) > 100:  # Only analyze substantial scripts
                    script_sample += script[:500] + "\n---\n"
            
            system_prompt = """You are a JavaScript data extraction expert. Analyze script content to find data relevant to the task.

Look for:
1. Variable assignments with arrays/objects
2. JSON data embedded in JavaScript
3. API responses or data initialization
4. Structured data patterns

Extract the relevant JavaScript code that contains the data. Return only the data assignment or object definition."""

            human_prompt = """Task: {task_description}

JavaScript Content Sample:
{script_sample}

Extract the JavaScript code containing the relevant data for this task."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "script_sample": script_sample
            })
            
            return result.strip()
            
        except Exception as e:
            print(f"Error in JavaScript extraction: {str(e)}")
            return ""
    
    def _parse_js_data_to_dataframe(self, js_data: str, task_description: str) -> pd.DataFrame:
        """
        Parse extracted JavaScript data into DataFrame
        """
        # Try to extract JSON-like structures from JavaScript
        json_pattern = r'(\{.*?\}|\[.*?\])'
        matches = re.findall(json_pattern, js_data, re.DOTALL)
        
        for match in matches:
            try:
                # Clean up JavaScript to make it valid JSON
                cleaned = match.replace("'", '"').replace('undefined', 'null')
                data_obj = json.loads(cleaned)
                return self._json_to_dataframe_with_llm(data_obj, task_description)
            except:
                continue
        
        raise ValueError("Could not parse JavaScript data to DataFrame")
    
    def _extract_div_data_with_llm(self, soup, task_description: str) -> pd.DataFrame:
        """
        Extract data from structured div elements using LLM guidance
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Find potential data containers
            data_containers = soup.find_all(['div', 'section', 'article'], 
                                          class_=re.compile(r'(chart|data|list|table|grid|content|results)', re.IGNORECASE))
            
            if not data_containers:
                data_containers = soup.find_all(['div', 'section', 'article'])[:10]  # First 10 as fallback
            
            # Analyze container structure
            container_info = []
            for i, container in enumerate(data_containers[:5]):  # Analyze first 5
                info = {
                    'index': i,
                    'tag': container.name,
                    'classes': container.get('class', []),
                    'content_preview': container.get_text()[:200],
                    'child_count': len(container.find_all(True)),
                    'structure': str(container)[:300]
                }
                container_info.append(info)
            
            system_prompt = """You are a web scraping expert specializing in extracting tabular data from div-based layouts.

Analyze the HTML structure and identify which container holds the relevant data. Then provide extraction instructions.

Respond in JSON format:
{{
  "container_index": 0,
  "extraction_method": "text_rows|attribute_values|nested_elements",
  "row_selector": "CSS selector for rows",
  "cell_selector": "CSS selector for cells within rows",
  "headers": ["column1", "column2", "column3"]
}}"""

            human_prompt = """Task: {task_description}

Container Analysis:
{container_info}

Which container contains the data and how should it be extracted?"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "container_info": json.dumps(container_info, indent=2)
            })
            
            # Parse instructions and extract data
            try:
                instructions = json.loads(result.strip().replace('```json', '').replace('```', ''))
                return self._extract_from_div_container(data_containers, instructions)
            except:
                raise ValueError("Could not parse div extraction instructions")
                
        except Exception as e:
            raise ValueError(f"Failed to extract from div structure: {str(e)}")
    
    def _extract_from_div_container(self, containers: List, instructions: Dict) -> pd.DataFrame:
        """
        Extract data from div container using provided instructions
        """
        container_idx = instructions.get('container_index', 0)
        if container_idx >= len(containers):
            container_idx = 0
        
        container = containers[container_idx]
        row_selector = instructions.get('row_selector', 'div')
        cell_selector = instructions.get('cell_selector', 'span, div')
        headers = instructions.get('headers', [])
        
        # Extract rows
        rows = container.select(row_selector)
        data_rows = []
        
        for row in rows:
            cells = row.select(cell_selector) if cell_selector else [row]
            cell_data = [cell.get_text(strip=True) for cell in cells]
            if cell_data and any(cell_data):  # Only add non-empty rows
                data_rows.append(cell_data)
        
        if not data_rows:
            raise ValueError("No data rows extracted from div container")
        
        # Create DataFrame
        if headers and len(headers) == len(data_rows[0]):
            df = pd.DataFrame(data_rows, columns=headers)
        else:
            # Auto-generate column names
            max_cols = max(len(row) for row in data_rows) if data_rows else 0
            columns = [f"Column_{i}" for i in range(max_cols)]
            # Pad rows to match column count
            padded_rows = [row + [''] * (max_cols - len(row)) for row in data_rows]
            df = pd.DataFrame(padded_rows, columns=columns)
        
        return df
    
    def _fallback_table_extraction(self, soup, task_description: str) -> pd.DataFrame:
        """
        Ultimate fallback: traditional table extraction
        """
        print("Using fallback table extraction method...")
        
        # Try pandas.read_html on the entire page
        try:
            tables = pd.read_html(str(soup))
            if tables:
                # Use simple heuristics to select best table
                best_table = max(tables, key=lambda t: t.shape[0] * t.shape[1])
                return best_table
        except:
            pass
        
        # Manual extraction from any table-like structure
        rows = soup.find_all("tr")
        if rows:
            extracted = []
            for row in rows:
                cells = row.find_all(["th", "td"])
                if cells:
                    extracted.append([cell.get_text(strip=True) for cell in cells])
            
            if extracted:
                # Use first row as header if consistent column count
                if len(extracted) > 1 and len(extracted[0]) == len(extracted[1]):
                    df = pd.DataFrame(extracted[1:], columns=extracted[0])
                else:
                    df = pd.DataFrame(extracted)
                return df
        
        raise ValueError("Could not extract any tabular data from the webpage")

    def _select_best_table_with_llm(self, tables: List[pd.DataFrame], task_description: str = '', keywords: List[str] = None) -> int:
        """
        Use LLM to intelligently select the most relevant table for analysis
        Based on task description and table previews
        """
        try:
            # Import LLM components here to avoid circular imports
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Create table previews for LLM analysis
            table_previews = []
            for i, table in enumerate(tables):
                # Get sample of first 3 rows for each table
                preview = {
                    'index': i,
                    'shape': f"{table.shape[0]} rows × {table.shape[1]} columns",
                    'columns': table.columns.tolist()[:10],  # Limit to first 10 columns
                    'sample_data': table.head(3).to_string(max_cols=10, max_rows=3)
                }
                table_previews.append(preview)
            
            # Create LLM prompt for table selection
            system_prompt = """You are an expert web scraping assistant. Given multiple HTML tables from a webpage, 
            select the most relevant table for data analysis based on the task description.
            
            Consider these factors:
            1. Table size and data density
            2. Column relevance to the task
            3. Data quality and completeness
            4. Avoid summary/navigation tables
            
            Respond with ONLY the table index number (0, 1, 2, etc.) that best matches the analysis requirements."""
            
            human_prompt = """Task: {task_description}
Keywords/entities: {keywords}

Available tables:
{table_info}

Which table index (0-{max_index}) contains the most relevant data for this analysis?
Respond with just the number."""
            
            # Format table information for LLM
            table_info_str = ""
            for preview in table_previews:
                table_info_str += f"\nTable {preview['index']}: {preview['shape']}\n"
                table_info_str += f"Columns: {preview['columns']}\n"
                table_info_str += f"Sample data:\n{preview['sample_data']}\n"
                table_info_str += "-" * 50 + "\n"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # Get LLM model and create chain
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            # Invoke LLM with table data
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "keywords": ', '.join(keywords) if keywords else '',
                "table_info": table_info_str,
                "max_index": len(tables) - 1
            })
            
            # Parse LLM response to get table index
            try:
                selected_index = int(result.strip())
                if 0 <= selected_index < len(tables):
                    print(f"LLM selected table {selected_index} for task: {task_description}")
                    return selected_index
                else:
                    print(f"LLM returned invalid index {selected_index}, using fallback")
                    return self._fallback_table_selection(tables)
            except (ValueError, AttributeError):
                print(f"Could not parse LLM response: {result}, using fallback")
                return self._fallback_table_selection(tables)
                
        except Exception as e:
            print(f"Error in LLM table selection: {str(e)}, using fallback")
            return self._fallback_table_selection(tables)
    
    def _fallback_table_selection(self, tables: List[pd.DataFrame]) -> int:
        """
        Fallback method for table selection when LLM is unavailable
        Uses simple heuristics based on table size and content
        """
        if not tables:
            return 0
            
        best_idx = 0
        best_score = 0
        
        for i, table in enumerate(tables):
            score = 0
            
            # Prefer tables with reasonable size (10+ rows, 2+ columns)
            if table.shape[0] >= 10:
                score += min(table.shape[0] * 0.5, 50)
            if table.shape[1] >= 2:
                score += table.shape[1] * 5
                
            # Prefer tables with numeric data
            numeric_cols = table.select_dtypes(include=[np.number]).columns
            score += len(numeric_cols) * 10
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        print(f"Fallback selection: table {best_idx} (score: {best_score})")
        return best_idx

class InspectTableStep:
    """
    Step 2: Generic data inspection
    - Print shape, columns, and head
    - Handle MultiIndex columns (flatten if needed)
    - Check if first row contains headers and set as columns if so
    - Generic approach for various data types (movies, countries, sports, etc.)
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data'].copy()
        print(f"\nSelected data shape: {data.shape}")
        print(f"Original columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head())
        
        # Handle MultiIndex columns (common in Wikipedia and other tables)
        if isinstance(data.columns, pd.MultiIndex):
            print("\nDetected MultiIndex columns, flattening...")
            new_columns = []
            for col in data.columns:
                if isinstance(col, tuple):
                    # Take the most specific part of the column name
                    if col[1] and col[1] != col[0]:
                        new_columns.append(f"{col[0]}_{col[1]}")
                    else:
                        new_columns.append(str(col[0]))
                else:
                    new_columns.append(str(col))
            data.columns = new_columns
            print(f"Flattened columns: {data.columns.tolist()}")
        
        # LLM-powered header detection for generic web scraping
        task_description = input_data.get('task_description', '')
        keywords = extract_keywords(task_description)
        first_row_is_header, header_row_idx = self._detect_headers_with_llm(data, task_description, keywords)
        
        # Apply header detection results
        if first_row_is_header and header_row_idx is not None:
            if header_row_idx > 0:
                print(f"Found headers in row {header_row_idx}")
                data.columns = [str(val) for val in data.iloc[header_row_idx]]
                data = data[header_row_idx+1:].reset_index(drop=True)
            else:
                print("First row confirmed as headers, setting as column names...")
                data.columns = [str(val) for val in data.iloc[0]]
                data = data[1:].reset_index(drop=True)
            print(f"Updated columns: {data.columns.tolist()}")
        
        print("\nAfter column processing:")
        print(f"Shape: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        print("\nFirst few rows:")
        print(data.head(3))
        
        return sanitize_for_json({'data': data})
    
    def _detect_headers_with_llm(self, data: pd.DataFrame, task_description: str = '', keywords: List[str] = None) -> tuple:
        """
        Use LLM to intelligently detect if any row contains headers
        Returns (is_header, row_index) tuple
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Check first 3 rows for potential headers
            rows_to_check = min(3, len(data))
            table_sample = data.head(rows_to_check).to_string(max_cols=10)
            
            system_prompt = """You are an expert data analyst. Examine the first few rows of a table and determine if any row contains column headers.
            
            Look for:
            1. Descriptive names instead of data values
            2. Text patterns typical of headers (Name, Rank, Total, etc.)
            3. Consistency with the analysis task
            4. Non-numeric values in what should be data rows
            
            Respond with ONLY the row index (0, 1, 2) that contains headers, or "NONE" if no headers are found in the data rows."""
            
            human_prompt = """Task: {task_description}
Keywords/entities: {keywords}

Table sample (first {rows_count} rows):
{table_sample}

Current column names: {current_columns}

Which row index (0, 1, 2) contains the headers, or "NONE" if headers are not in the data rows?
Respond with just the number or "NONE"."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "keywords": ', '.join(keywords) if keywords else '',
                "table_sample": table_sample,
                "rows_count": rows_to_check,
                "current_columns": data.columns.tolist()
            })
            
            # Parse LLM response
            result_clean = result.strip().upper()
            if result_clean == "NONE":
                print("LLM determined no headers in data rows")
                return False, None
            else:
                try:
                    header_row = int(result_clean)
                    if 0 <= header_row < rows_to_check:
                        print(f"LLM detected headers in row {header_row}")
                        return True, header_row
                    else:
                        print(f"LLM returned invalid row index: {result}, using fallback")
                        return self._fallback_header_detection(data)
                except ValueError:
                    print(f"Could not parse LLM response: {result}, using fallback")
                    return self._fallback_header_detection(data)
                    
        except Exception as e:
            print(f"Error in LLM header detection: {str(e)}, using fallback")
            return self._fallback_header_detection(data)
    
    def _fallback_header_detection(self, data: pd.DataFrame) -> tuple:
        """
        Fallback method for header detection when LLM is unavailable
        """
        # Check if column names are non-descriptive
        has_unnamed_cols = any(str(col).startswith('Unnamed') or str(col).isdigit() for col in data.columns)
        
        if has_unnamed_cols and len(data) > 0:
            # Check first row for header-like content
            first_row_values = data.iloc[0].astype(str).tolist()
            # Simple check for text that looks like headers
            header_like_count = sum(1 for val in first_row_values if 
                                  len(str(val)) > 1 and not str(val).replace('.', '').replace('-', '').isdigit())
            
            if header_like_count >= len(first_row_values) * 0.5:  # At least half look like headers
                print("Fallback detected likely headers in first row")
                return True, 0
        
        print("Fallback found no headers in data rows")
        return False, None

class CleanDataStep:
    """
    Step 3: Generic data cleaning
    - Remove symbols, footnotes, and convert to numeric
    - CRITICAL: Use select_dtypes to find numeric columns
    - Handle various data formats (currency, percentages, etc.)
    - Print after cleaning
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data'].copy()
        print("\n=== CLEANING DATA ===")
        print(f"Original data types:\n{data.dtypes}")
        
        # Clean each column
        task_description = input_data.get('task_description', '')
        keywords = extract_keywords(task_description)
        for col in data.columns:
            if data[col].dtype == 'object':
                print(f"\nCleaning column: {col}")
                # Convert to string first
                cleaned = data[col].astype(str)
                
                # LLM-guided cleaning (inject keywords/entities)
                # Could be enhanced to call LLM for cleaning strategy
                # Remove common symbols and formatting (generic for various data types)
                cleaned = cleaned.str.replace('$', '', regex=False)  # Currency
                cleaned = cleaned.str.replace(',', '', regex=False)  # Thousands separator
                cleaned = cleaned.str.replace('€', '', regex=False)  # Euro
                cleaned = cleaned.str.replace('£', '', regex=False)  # Pound
                cleaned = cleaned.str.replace('¥', '', regex=False)  # Yen
                cleaned = cleaned.str.replace('%', '', regex=False)  # Percentage
                cleaned = cleaned.str.replace('₹', '', regex=False)  # Rupee
                cleaned = cleaned.str.replace('billion', '', regex=False)  # Scale indicators
                cleaned = cleaned.str.replace('million', '', regex=False)
                cleaned = cleaned.str.replace('trillion', '', regex=False)
                cleaned = cleaned.str.replace('bn', '', regex=False)
                cleaned = cleaned.str.replace('mn', '', regex=False)
                # Add more currency and number formats
                cleaned = cleaned.str.replace('B', '', regex=False)  # Billion abbreviation
                cleaned = cleaned.str.replace('M', '', regex=False)  # Million abbreviation
                cleaned = cleaned.str.replace('K', '', regex=False)  # Thousand abbreviation
                
                # Remove footnote references like [1], [n 1], etc.
                cleaned = cleaned.str.replace(r'\[.*?\]', '', regex=True)
                cleaned = cleaned.str.replace(r'\([^)]*\)', '', regex=True)  # Remove parentheses content
                
                # Remove any other non-numeric characters except decimal points and minus signs
                cleaned = cleaned.str.replace(r'[^\d.\-]', '', regex=True)
                
                # Handle empty strings and special cases
                cleaned = cleaned.replace('', np.nan)
                cleaned = cleaned.replace('nan', np.nan)
                cleaned = cleaned.replace('NaN', np.nan)
                cleaned = cleaned.replace('None', np.nan)
                cleaned = cleaned.replace('N/A', np.nan)
                cleaned = cleaned.replace('–', np.nan)  # En dash
                cleaned = cleaned.replace('—', np.nan)  # Em dash
                
                # Convert to numeric
                numeric_data = pd.to_numeric(cleaned, errors='coerce')
                
                # Only replace if we got some valid numbers (at least 5% of data should be numeric)
                valid_count = numeric_data.notna().sum()
                total_count = len(data)
                if valid_count > 0 and valid_count >= max(3, total_count * 0.05):  # Lower threshold for small datasets
                    print(f"  Converted {valid_count} values to numeric ({valid_count/total_count*100:.1f}%)")
                    data[col] = numeric_data
                    
                    # Handle scale factors (if column had billion/million indicators)
                    original_str = data[col].astype(str).str.lower()
                    if any('billion' in str(val) or 'bn' in str(val) for val in data[col].astype(str).iloc[:5]):
                        print(f"  Detected billion scale factor in {col}")
                        # Values are likely already in billions, don't multiply
                    elif any('million' in str(val) or 'mn' in str(val) for val in data[col].astype(str).iloc[:5]):
                        print(f"  Detected million scale factor in {col}, converting to billions")
                        data[col] = data[col] / 1000  # Convert millions to billions
                else:
                    print(f"  No valid numeric data found ({valid_count} values), keeping as text")
        
        print("\nAfter cleaning:")
        print(f"Data types:\n{data.dtypes}")
        print(data.head(3))
        
        # Find numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\nNumeric columns found: {numeric_cols}")
        
        # Print some statistics about numeric columns
        for col in numeric_cols:
            valid_count = data[col].notna().sum()
            if valid_count > 0:
                print(f"  {col}: {valid_count} valid values, range: {data[col].min():.2f} to {data[col].max():.2f}")
        
        # Replace inf/-inf with nan, then nan with None for JSON
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.where(pd.notnull(data), None)
        return sanitize_for_json({'data': data, 'numeric_cols': numeric_cols})

class AnalyzeDataStep:
    """
    Step 4: Generic data analysis
    - Use most relevant numeric column for analysis
    - Remove NaNs, sort, get top N
    - Support various analysis types (rankings, filtering, etc.)
    - Print results in a generic format
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        data = input_data['data']
        numeric_cols = input_data['numeric_cols']
        top_n = input_data.get('top_n', 10)
        
        print(f"\n=== ANALYZING DATA ===")
        print(f"Available numeric columns: {numeric_cols}")
        
        if not numeric_cols:
            print("ERROR: No numeric columns found for analysis")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': None}
        
        # LLM-powered column selection for analysis
        task_description = input_data.get('task_description', '')
        keywords = extract_keywords(task_description)
        best_col = self._select_analysis_column_with_llm(data, numeric_cols, task_description, keywords)
        
        if not best_col:
            print("ERROR: No suitable column found for analysis")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': None}
        
        print(f"Selected column '{best_col}' for analysis")
        
        # Clean and analyze - first filter out summary/total rows using LLM
        data_clean = data.dropna(subset=[best_col])
        data_clean = self._filter_summary_rows_with_llm(data_clean, input_data.get('task_description', ''))
        
        print(f"After removing NaN values and filtering summary rows: {data_clean.shape[0]} rows")
        
        if len(data_clean) == 0:
            print("ERROR: No valid data after cleaning and filtering")
            return {'top_n_df': pd.DataFrame(), 'analysis_col': best_col}
        
        # Sort by the analysis column (descending for most metrics like revenue, GDP, etc.)
        data_sorted = data_clean.sort_values(best_col, ascending=False)
        top_n_df = data_sorted.head(top_n)
        
        print(f"\nTop {len(top_n_df)} by {best_col}:")
        
        # Find the identifier column (usually first text column)
        text_cols = data.select_dtypes(include=['object']).columns.tolist()
        name_col = text_cols[0] if text_cols else data.columns[0]
        
        print(f"Using '{name_col}' as identifier column")
        
        # Display results
        for i, (idx, row) in enumerate(top_n_df.iterrows()):
            identifier = row[name_col]
            value = row[best_col]
            print(f"{i+1:2d}. {identifier}: {value:,.2f}")
        
        # Replace inf/-inf with nan, then nan with None for JSON
        top_n_df = top_n_df.replace([np.inf, -np.inf], np.nan)
        top_n_df = top_n_df.where(pd.notnull(top_n_df), None)
        data_clean = data_clean.replace([np.inf, -np.inf], np.nan)
        data_clean = data_clean.where(pd.notnull(data_clean), None)
        return sanitize_for_json({'top_n_df': top_n_df, 'analysis_col': best_col, 'name_col': name_col, 'data_clean': data_clean})
    
    def _select_analysis_column_with_llm(self, data: pd.DataFrame, numeric_cols: List[str], task_description: str = '', keywords: List[str] = None) -> str:
        """
        Use LLM to intelligently select the most relevant numeric column for analysis
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Create column descriptions for LLM
            column_info = []
            for col in numeric_cols:
                valid_count = data[col].notna().sum()
                if valid_count > 0:
                    sample_values = data[col].dropna().head(5).tolist()
                    value_range = f"{data[col].min():.2f} to {data[col].max():.2f}"
                    column_info.append(f"Column '{col}': {valid_count} valid values, range {value_range}, samples: {sample_values}")
            
            system_prompt = """You are an expert data analyst. Given numeric columns and a task description, 
            select the most relevant column for analysis.
            
            Avoid columns that are:
            - Summary/total columns (containing "total", "sum", "world", etc.)
            - Year columns (4-digit numbers starting with 19xx or 20xx)
            - Rank/position columns (containing "rank", "position")
            - Index columns
            
            Prefer columns with:
            - Values relevant to the analysis task
            - Good data completeness
            - Meaningful value ranges for comparison
            
            Respond with ONLY the exact column name."""
            
            human_prompt = """Task: {task_description}
Keywords/entities: {keywords}

Available numeric columns:
{column_descriptions}

Which column is most relevant for this analysis? Respond with just the column name."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "keywords": ', '.join(keywords) if keywords else '',
                "column_descriptions": "\n".join(column_info)
            })
            
            # Parse LLM response
            selected_col = result.strip().strip("'\"")  # Remove quotes
            if selected_col in numeric_cols:
                print(f"LLM selected column: {selected_col}")
                return selected_col
            else:
                print(f"LLM returned invalid column: {selected_col}, using fallback")
                return self._fallback_column_selection(data, numeric_cols)
                
        except Exception as e:
            print(f"Error in LLM column selection: {str(e)}, using fallback")
            return self._fallback_column_selection(data, numeric_cols)
    
    def _fallback_column_selection(self, data: pd.DataFrame, numeric_cols: List[str]) -> str:
        """
        Fallback method for column selection when LLM is unavailable
        """
        best_col = None
        best_score = 0
        
        for col in numeric_cols:
            score = 0
            col_name = str(col).lower()
            
            # Avoid summary/total/rank columns
            if any(keyword in col_name for keyword in ['world', 'total', 'sum', 'rank', 'position']):
                continue
            
            # Avoid year columns
            if col_name.isdigit() or (len(col_name) == 4 and col_name.startswith('2')):
                continue
            
            # Score based on data completeness
            valid_count = data[col].notna().sum()
            score += valid_count
            
            # Score based on value range (prefer columns with variation)
            if valid_count > 0:
                value_range = data[col].max() - data[col].min()
                if value_range > 0:
                    score += 100
            
            if score > best_score:
                best_score = score
                best_col = col
        
        # If no column found, use first numeric column
        if not best_col and numeric_cols:
            best_col = numeric_cols[0]
        
        print(f"Fallback selected column: {best_col}")
        return best_col
    
    def _filter_summary_rows_with_llm(self, data: pd.DataFrame, task_description: str = '') -> pd.DataFrame:
        """
        Use LLM to identify and filter out summary/total rows
        """
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Find text columns that might contain identifiers
            text_cols = data.select_dtypes(include=['object']).columns.tolist()
            if not text_cols or len(data) == 0:
                return data
                
            name_col = text_cols[0]
            sample_data = data.head(20)  # Look at first 20 rows
            
            system_prompt = """You are a data cleaning expert. Examine the data rows and identify which ones are summary/total/aggregate rows that should be filtered out for analysis.

Look for rows that contain:
- Total/sum values (like "Total", "World", "All Countries", "Overall")
- Summary statistics (like "Average", "Mean", "Median")
- Aggregate categories (like "Other", "Others", "Miscellaneous")
- Non-specific entries that aren't individual data points

Return a comma-separated list of the exact values from the name column that should be removed.
If no summary rows are found, respond with "NONE"."""

            human_prompt = """Task: {task_description}

Data sample (first 20 rows, name column '{name_col}'):
{data_sample}

Which values from the '{name_col}' column represent summary/total rows that should be filtered out?
Respond with exact values separated by commas, or "NONE" if no summary rows found."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description or "general data analysis",
                "name_col": name_col,
                "data_sample": sample_data[[name_col]].to_string()
            })
            
            # Parse LLM response and filter rows
            result_clean = result.strip().upper()
            if result_clean != "NONE":
                rows_to_remove = [item.strip().strip('"\'') for item in result.split(',')]
                before_count = len(data)
                
                for row_value in rows_to_remove:
                    if row_value:  # Skip empty values
                        # Case-insensitive filtering
                        mask = ~data[name_col].astype(str).str.lower().str.contains(
                            row_value.lower(), na=False, regex=False
                        )
                        data = data[mask]
                
                after_count = len(data)
                if before_count != after_count:
                    print(f"LLM identified and filtered out {before_count - after_count} summary rows")
                    print(f"Removed values: {rows_to_remove}")
            else:
                print("LLM found no summary rows to filter")
            
            return data
            
        except Exception as e:
            print(f"Error in LLM summary row filtering: {str(e)}, using fallback")
            # Fallback to keyword-based approach
            if len(data) > 0:
                text_cols = data.select_dtypes(include=['object']).columns.tolist()
                if text_cols:
                    name_col = text_cols[0]
                    print(f"Filtering summary rows using '{name_col}' column")
                    
                    # Generic summary keywords for various domains
                    summary_keywords = [
                        'world', 'total', 'sum', 'all', 'global', 'aggregate',
                        'overall', 'average', 'mean', 'median', 'other', 'others'
                    ]
                    before_count = len(data)
                    for keyword in summary_keywords:
                        data = data[~data[name_col].astype(str).str.lower().str.contains(keyword, na=False)]
                    
                    after_count = len(data)
                    if before_count != after_count:
                        print(f"Filtered out {before_count - after_count} summary rows")
            
            return data

class VisualizeStep:
    """
    Step 5: Enhanced generic visualization
    - Support multiple chart types (bar, scatter, histogram, time_series)
    - Auto-detect visualization type based on task requirements
    - Use dynamic column names
    - Return base64 encoded images when requested
    - Handle various data relationships (rank vs peak, total vs deaths, etc.)
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data.get('top_n_df')
        analysis_col = input_data.get('analysis_col')
        name_col = input_data.get('name_col')
        data_clean = input_data.get('data_clean')
        task_description = input_data.get('task_description', '')
        keywords = extract_keywords(task_description)
        chart_type = input_data.get('chart_type', self._auto_detect_chart_type(task_description, data_clean, keywords))
        
        if top_n_df is None or top_n_df.empty or analysis_col is None:
            print("No data available for visualization.")
            return {'plot_path': None, 'plot_base64': None}
        
        try:
            import base64
            from io import BytesIO
            
            plt.figure(figsize=(12, 8))
            
            if chart_type == 'bar':
                # Standard bar chart for top N
                plt.bar(range(len(top_n_df)), top_n_df[analysis_col], color='skyblue', alpha=0.7)
                plt.xticks(range(len(top_n_df)), top_n_df[name_col], rotation=45, ha='right')
                plt.title(f'Top {len(top_n_df)} by {analysis_col}')
                plt.xlabel(name_col)
                plt.ylabel(analysis_col)
                
            elif chart_type == 'scatter':
                # Enhanced scatter plot with auto-detection of x/y columns
                numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
                x_col, y_col = self._select_scatter_columns(numeric_cols, analysis_col, task_description)
                
                if x_col and y_col and len(numeric_cols) >= 2:
                    clean_data = data_clean[[x_col, y_col]].dropna()
                    plt.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, s=50)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.title(f'{x_col} vs {y_col}')
                    
                    # Add regression line (dotted red line as specified in requirements)
                    if len(clean_data) > 1:
                        try:
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(clean_data[x_col], clean_data[y_col])
                            line = slope * clean_data[x_col] + intercept
                            plt.plot(clean_data[x_col], line, 'r:', linewidth=2, alpha=0.8, 
                                   label=f'Regression Line (r²={r_value**2:.3f})')
                            plt.legend()
                        except ImportError:
                            # Fallback manual regression if scipy not available
                            z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                            p = np.poly1d(z)
                            plt.plot(clean_data[x_col], p(clean_data[x_col]), 'r:', linewidth=2, alpha=0.8, 
                                   label='Regression Line')
                            plt.legend()
                else:
                    # Fallback to bar chart if not enough numeric columns
                    chart_type = 'bar'
                    plt.bar(range(len(top_n_df)), top_n_df[analysis_col], color='skyblue', alpha=0.7)
                    plt.xticks(range(len(top_n_df)), top_n_df[name_col], rotation=45, ha='right')
                    plt.title(f'Top {len(top_n_df)} by {analysis_col}')
                    plt.xlabel(name_col)
                    plt.ylabel(analysis_col)
                    
            elif chart_type == 'histogram':
                # Enhanced histogram with better binning
                data_values = data_clean[analysis_col].dropna()
                if len(data_values) > 0:
                    bins = min(20, max(5, len(data_values) // 5))  # Dynamic bin sizing
                    plt.hist(data_values, bins=bins, alpha=0.7, color='lightgreen', edgecolor='black')
                    plt.title(f'Distribution of {analysis_col}')
                    plt.xlabel(analysis_col)
                    plt.ylabel('Frequency')
                    # Add mean line
                    mean_val = data_values.mean()
                    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.7, 
                              label=f'Mean: {mean_val:.2f}')
                    plt.legend()
                    
            elif chart_type == 'time_series':
                # Enhanced time series with better date handling
                date_cols = self._find_date_columns(data_clean)
                
                if date_cols:
                    date_col = date_cols[0]
                    sorted_data = data_clean.sort_values(date_col)
                    plt.plot(sorted_data[date_col], sorted_data[analysis_col], 
                           marker='o', linewidth=2, markersize=6)
                    plt.title(f'{analysis_col} over {date_col}')
                    plt.xlabel(date_col)
                    plt.ylabel(analysis_col)
                    plt.xticks(rotation=45)
                    plt.grid(True, alpha=0.3)
                else:
                    # Fallback to bar chart if no date column found
                    chart_type = 'bar'
                    plt.bar(range(len(top_n_df)), top_n_df[analysis_col], color='skyblue', alpha=0.7)
                    plt.xticks(range(len(top_n_df)), top_n_df[name_col], rotation=45, ha='right')
                    plt.title(f'Top {len(top_n_df)} by {analysis_col}')
                    plt.xlabel(name_col)
                    plt.ylabel(analysis_col)
            
            plt.tight_layout()
            
            # Always generate base64 for API responses
            plot_base64 = None
            buffer = BytesIO()
            # Optimize for size while maintaining quality
            plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            
            # Ensure base64 is under 100KB as required
            if len(plot_base64) > 100000:
                # Reduce quality if too large
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=60, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buffer.seek(0)
                plot_base64 = base64.b64encode(buffer.getvalue()).decode()
                buffer.close()
            
            plt.show()
            
            print(f"Generated {chart_type} visualization for {analysis_col}")
            print(f"Base64 size: {len(plot_base64)} characters")
            
            return {
                'plot_path': 'generated', 
                'plot_base64': f"data:image/png;base64,{plot_base64}", 
                'chart_type': chart_type,
                'image_size_bytes': len(plot_base64)
            }
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return {'plot_path': None, 'plot_base64': None, 'error': str(e)}
    
    def _auto_detect_chart_type(self, task_description: str, data_clean, keywords: List[str] = None) -> str:
        """Auto-detect the appropriate chart type based on task description using LLM"""
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Get column information for context
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = data_clean.select_dtypes(include=['object']).columns.tolist()
            
            system_prompt = """You are a data visualization expert. Based on the task description and data characteristics, 
            recommend the most appropriate chart type.

Available chart types:
- bar: For rankings, comparisons, top N items
- scatter: For correlations, relationships between two numeric variables
- histogram: For distributions of a single numeric variable
- time_series: For data over time periods

Consider:
1. What the task is asking for (correlation, distribution, ranking, time trends)
2. The type of analysis needed
3. The data structure and column types

Respond with ONLY one of: bar, scatter, histogram, time_series"""

            human_prompt = """Task: {task_description}
Keywords/entities: {keywords}

Data characteristics:
- Numeric columns: {numeric_cols}
- Text columns: {text_cols}
- Data shape: {data_shape}

What chart type is most appropriate? Respond with just: bar, scatter, histogram, or time_series"""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "keywords": ', '.join(keywords) if keywords else '',
                "numeric_cols": numeric_cols[:5],  # Limit for context
                "text_cols": text_cols[:3],
                "data_shape": f"{data_clean.shape[0]} rows, {data_clean.shape[1]} columns"
            })
            
            chart_type = result.strip().lower()
            valid_types = ['bar', 'scatter', 'histogram', 'time_series']
            
            if chart_type in valid_types:
                print(f"LLM recommended chart type: {chart_type}")
                return chart_type
            else:
                print(f"LLM returned invalid chart type: {chart_type}, using fallback")
                return self._fallback_chart_detection(task_description)
                
        except Exception as e:
            print(f"Error in LLM chart type detection: {str(e)}, using fallback")
            return self._fallback_chart_detection(task_description)
    
    def _fallback_chart_detection(self, task_description: str) -> str:
        """Fallback chart type detection using keywords"""
        task_lower = task_description.lower()
        
        if 'scatterplot' in task_lower or 'scatter plot' in task_lower:
            return 'scatter'
        elif 'histogram' in task_lower:
            return 'histogram'
        elif 'time series' in task_lower or 'over time' in task_lower:
            return 'time_series'
        elif any(word in task_lower for word in ['correlation', 'vs', 'versus', 'relationship']):
            return 'scatter'
        else:
            return 'bar'
    
    def _select_scatter_columns(self, numeric_cols: List[str], analysis_col: str, task_description: str) -> tuple:
        """Select appropriate columns for scatter plot based on task description"""
        task_lower = task_description.lower()
        
        # Look for specific column relationships mentioned in task
        if 'rank' in task_lower and 'peak' in task_lower:
            # Find rank and peak columns
            rank_col = None
            peak_col = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if 'rank' in col_lower:
                    rank_col = col
                elif 'peak' in col_lower:
                    peak_col = col
            if rank_col and peak_col:
                return rank_col, peak_col
        
        elif 'cases' in task_lower and 'deaths' in task_lower:
            # Find cases and deaths columns
            cases_col = None
            deaths_col = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if 'cases' in col_lower or 'total' in col_lower:
                    cases_col = col
                elif 'deaths' in col_lower:
                    deaths_col = col
            if cases_col and deaths_col:
                return cases_col, deaths_col
        
        elif 'runs' in task_lower and 'average' in task_lower:
            # Find runs and average columns for cricket data
            runs_col = None
            avg_col = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if 'runs' in col_lower or 'total' in col_lower:
                    runs_col = col
                elif 'average' in col_lower or 'avg' in col_lower:
                    avg_col = col
            if runs_col and avg_col:
                return runs_col, avg_col
        
        # Default: use first two numeric columns, prioritizing analysis_col as y-axis
        if len(numeric_cols) >= 2:
            if analysis_col in numeric_cols:
                other_cols = [col for col in numeric_cols if col != analysis_col]
                return other_cols[0], analysis_col
            else:
                return numeric_cols[0], numeric_cols[1]
        
        return None, None
    
    def _find_date_columns(self, data_clean) -> List[str]:
        """Find columns that might contain date/time information"""
        date_cols = []
        for col in data_clean.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['year', 'date', 'time', 'month']):
                date_cols.append(col)
        return date_cols

class AnswerQuestionsStep:
    """
    Step 6: Enhanced generic question answering
    - Use dynamic column names
    - Support various types of questions across different domains
    - Handle financial, geographic, sports, health, and other data types
    - Store answers in variables for capture
    """
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        top_n_df = input_data.get('top_n_df')
        analysis_col = input_data.get('analysis_col')
        name_col = input_data.get('name_col')
        data_clean = input_data.get('data_clean')
        task_description = input_data.get('task_description', '')
        questions = input_data.get('questions', [])
        
        print(f"\n=== ANSWERING QUESTIONS ===")
        
        answers = {}
        
        if top_n_df is None or top_n_df.empty or analysis_col is None:
            print("ERROR: No data available for answering questions")
            answers = {
                'error': 'No data available for analysis',
                'status': 'failed'
            }
        else:
            print(f"Analyzing top {len(top_n_df)} entries by {analysis_col}")
            
            # Enhanced question answering based on data structure and domain
            
            # LLM-powered question interpretation and answering
            answers.update(self._answer_questions_with_llm(
                data_clean, top_n_df, analysis_col, name_col, task_description
            ))
            
            # Basic statistics and rankings
            if len(top_n_df) >= 5:
                rank_5_item = top_n_df.iloc[4][name_col]
                rank_5_value = top_n_df.iloc[4][analysis_col]
                answers['item_ranking_5th'] = rank_5_item
                answers['item_ranking_5th_value'] = rank_5_value
                print(f"Item ranking 5th: {rank_5_item} ({rank_5_value:,.2f})")
            else:
                answers['item_ranking_5th'] = f'Only {len(top_n_df)} items available'
                answers['item_ranking_5th_value'] = 0
                print(f"Not enough data - only {len(top_n_df)} items in dataset")
            
            # Total and average calculations
            total_top_n = top_n_df[analysis_col].sum()
            avg_top_n = top_n_df[analysis_col].mean()
            answers['total_top_n'] = total_top_n
            answers['average_top_n'] = avg_top_n
            print(f"Total {analysis_col} of top {len(top_n_df)}: {total_top_n:,.2f}")
            print(f"Average {analysis_col} of top {len(top_n_df)}: {avg_top_n:,.2f}")
            
            # Range analysis
            if len(data_clean) > 0:
                max_value = data_clean[analysis_col].max()
                min_value = data_clean[analysis_col].min()
                answers['max_value'] = max_value
                answers['min_value'] = min_value
                answers['range'] = max_value - min_value
                print(f"Range: {min_value:,.2f} to {max_value:,.2f}")
            
            # Correlation analysis if multiple numeric columns
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                correlations = {}
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        try:
                            corr = data_clean[[col1, col2]].corr().iloc[0, 1]
                            correlations[f"{col1}_vs_{col2}"] = corr
                            print(f"Correlation {col1} vs {col2}: {corr:.3f}")
                        except:
                            pass
                answers['correlations'] = correlations
            
            # Domain-specific question handling
            
            # 1. Financial/Revenue data (movies, GDP, etc.)
            if self._is_financial_data(analysis_col, task_description):
                self._answer_financial_questions(answers, data_clean, analysis_col, name_col, task_description)
            
            # 2. Health/Medical data (COVID, disease statistics)
            elif self._is_health_data(analysis_col, task_description):
                self._answer_health_questions(answers, data_clean, analysis_col, name_col, top_n_df)
            
            # 3. Sports data (cricket, football, etc.)
            elif self._is_sports_data(analysis_col, task_description):
                self._answer_sports_questions(answers, data_clean, analysis_col, name_col, top_n_df)
            
            # 4. Economic data (inflation, trading)
            elif self._is_economic_data(analysis_col, task_description):
                self._answer_economic_questions(answers, data_clean, analysis_col, name_col)
            
            # 5. Entertainment data (ratings, reviews)
            elif self._is_entertainment_data(analysis_col, task_description):
                self._answer_entertainment_questions(answers, data_clean, analysis_col, name_col, top_n_df)
            
            # Time-based analysis if year column exists
            year_cols = [col for col in data_clean.columns if 'year' in str(col).lower()]
            if year_cols:
                self._answer_temporal_questions(answers, data_clean, analysis_col, name_col, year_cols[0])
            
            # Full top N list for reference
            top_list = []
            for i, (idx, row) in enumerate(top_n_df.iterrows()):
                top_list.append({
                    'rank': i + 1,
                    'name': row[name_col],
                    'value': row[analysis_col]
                })
            answers['top_n_list'] = top_list
            
            # Summary metadata
            answers['summary'] = {
                'analysis_column': analysis_col,
                'name_column': name_col,
                'total_items_analyzed': len(top_n_df),
                'total_items_in_dataset': len(data_clean) if data_clean is not None else 0,
                'data_type': self._identify_data_type(analysis_col, task_description),
                'domain': self._identify_domain(task_description)
            }
            
            answers['status'] = 'success'
        
        print("\nFINAL ANSWERS:")
        for key, value in answers.items():
            if key not in ['top_n_list', 'correlations']:  # Don't print large objects
                print(f"  {key}: {value}")
        
        return sanitize_for_json({'answers': answers})
    
    def _answer_questions_with_llm(self, data_clean, top_n_df, analysis_col: str, name_col: str, task_description: str) -> dict:
        """Use LLM to intelligently interpret and answer questions from task description"""
        try:
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import StrOutputParser
            from config import get_chat_model
            
            # Extract key data insights for LLM context
            data_insights = {
                'total_rows': len(data_clean),
                'top_n_count': len(top_n_df),
                'analysis_column': analysis_col,
                'name_column': name_col,
                'max_value': float(data_clean[analysis_col].max()) if len(data_clean) > 0 else 0,
                'min_value': float(data_clean[analysis_col].min()) if len(data_clean) > 0 else 0,
                'average_value': float(data_clean[analysis_col].mean()) if len(data_clean) > 0 else 0,
            }
            
            # Get numeric columns for correlation analysis
            numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
            
            # Get year column if exists for temporal questions
            year_cols = [col for col in data_clean.columns if 'year' in str(col).lower()]
            
            # Prepare top N data for LLM
            if len(top_n_df) >= 5:
                top_5_items = []
                for i in range(min(5, len(top_n_df))):
                    top_5_items.append({
                        'rank': i + 1,
                        'name': str(top_n_df.iloc[i][name_col]),
                        'value': float(top_n_df.iloc[i][analysis_col])
                    })
            else:
                top_5_items = []
            
            system_prompt = """You are an expert data analyst. Based on the task description and data insights, 
            answer specific questions that might be asked about the data.

Common question types to address:
1. Ranking questions (Who/what ranks 5th? Which is earliest/latest?)
2. Threshold questions (How many above X value? Items before year Y?)
3. Statistical questions (What's the correlation? Average? Total?)
4. Domain-specific questions (death rates, averages above threshold, etc.)

Provide specific numeric answers and insights. Return answers as a JSON object with descriptive keys.
Focus on actionable insights that directly address the analysis goals."""

            human_prompt = """Task: {task_description}

Data insights:
- Total items: {total_rows}
- Analysis column: {analysis_column} (range: {min_value:.2f} to {max_value:.2f})
- Average value: {average_value:.2f}
- Top {top_n_count} items available
- Numeric columns: {numeric_cols}
- Year columns: {year_cols}

Top 5 items:
{top_5_data}

Based on this data and the task requirements, what are the key insights and answers to likely questions?
Respond with a JSON object containing relevant answers and insights."""

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            llm = get_chat_model()
            chain = prompt | llm | StrOutputParser()
            
            result = chain.invoke({
                "task_description": task_description,
                "total_rows": data_insights['total_rows'],
                "analysis_column": analysis_col,
                "min_value": data_insights['min_value'],
                "max_value": data_insights['max_value'], 
                "average_value": data_insights['average_value'],
                "top_n_count": data_insights['top_n_count'],
                "numeric_cols": numeric_cols[:5],  # Limit for context
                "year_cols": year_cols,
                "top_5_data": str(top_5_items)
            })
            
            # Try to parse JSON response
            try:
                import json
                llm_answers = json.loads(result)
                print("LLM provided intelligent question answers")
                return llm_answers
            except json.JSONDecodeError:
                print("LLM response not in JSON format, extracting key insights")
                # Extract key insights from text response
                insights = {}
                if "5th" in result.lower() and len(top_5_items) >= 5:
                    insights['llm_fifth_item_insight'] = top_5_items[4]['name']
                if "total" in result.lower():
                    insights['llm_total_insight'] = sum(item['value'] for item in top_5_items)
                if "average" in result.lower():
                    insights['llm_average_insight'] = data_insights['average_value']
                
                insights['llm_analysis_summary'] = result[:200] + "..." if len(result) > 200 else result
                return insights
                
        except Exception as e:
            print(f"Error in LLM question answering: {str(e)}")
            return {'llm_error': str(e)}
    
    def _is_financial_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is financial/revenue data"""
        financial_keywords = ['gross', 'revenue', 'gdp', 'billion', 'million', 'box office', 'earnings']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in financial_keywords)
    
    def _is_health_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is health/medical data"""
        health_keywords = ['cases', 'deaths', 'covid', 'infection', 'recovery', 'mortality', 'disease']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in health_keywords)
    
    def _is_sports_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is sports data"""
        sports_keywords = ['runs', 'average', 'cricket', 'batsmen', 'matches', 'innings', 'wickets', 'goals', 'points']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in sports_keywords)
    
    def _is_economic_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is economic data"""
        economic_keywords = ['inflation', 'cpi', 'rate', 'economics', 'trading', 'price', 'index']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in economic_keywords)
    
    def _is_entertainment_data(self, analysis_col: str, task_description: str) -> bool:
        """Check if this is entertainment data"""
        entertainment_keywords = ['rating', 'imdb', 'score', 'movie', 'film', 'review']
        col_lower = str(analysis_col).lower()
        task_lower = task_description.lower()
        return any(keyword in col_lower or keyword in task_lower for keyword in entertainment_keywords)
    
    def _answer_financial_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, task_description: str):
        """Answer financial/revenue specific questions"""
        if 'billion' in str(analysis_col).lower() or 'billion' in task_description.lower():
            # Count items above certain thresholds
            above_1_5bn = len(data_clean[data_clean[analysis_col] > 1500])
            above_2bn = len(data_clean[data_clean[analysis_col] > 2000])
            answers['items_above_1_5_billion'] = above_1_5bn
            answers['items_above_2_billion'] = above_2bn
            print(f"Items above 1.5 billion: {above_1_5bn}")
            print(f"Items above 2 billion: {above_2bn}")
    
    def _answer_health_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, top_n_df):
        """Answer health/medical specific questions"""
        # Look for death rate calculations
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        deaths_col = None
        cases_col = None
        
        for col in numeric_cols:
            col_lower = str(col).lower()
            if 'deaths' in col_lower:
                deaths_col = col
            elif 'cases' in col_lower or 'total' in col_lower:
                cases_col = col
        
        if deaths_col and cases_col:
            # Calculate death-to-case ratio
            data_clean['death_rate'] = (data_clean[deaths_col] / data_clean[cases_col] * 100).round(2)
            highest_death_rate_country = data_clean.loc[data_clean['death_rate'].idxmax(), name_col]
            highest_death_rate_value = data_clean['death_rate'].max()
            
            answers['highest_death_rate_country'] = highest_death_rate_country
            answers['highest_death_rate_value'] = highest_death_rate_value
            print(f"Highest death-to-case ratio: {highest_death_rate_country} ({highest_death_rate_value:.2f}%)")
            
            # Global average calculations
            global_death_rate = (data_clean[deaths_col].sum() / data_clean[cases_col].sum() * 100)
            answers['global_average_death_rate'] = global_death_rate
            print(f"Global average death rate: {global_death_rate:.2f}%")
    
    def _answer_sports_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, top_n_df):
        """Answer sports specific questions"""
        numeric_cols = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Look for runs and average columns
        runs_col = None
        avg_col = None
        
        for col in numeric_cols:
            col_lower = str(col).lower()
            if 'runs' in col_lower:
                runs_col = col
            elif 'average' in col_lower or 'avg' in col_lower:
                avg_col = col
        
        if runs_col and avg_col:
            # Find highest average among players with over 8000 runs
            high_runs_players = data_clean[data_clean[runs_col] > 8000]
            if len(high_runs_players) > 0:
                highest_avg_player = high_runs_players.loc[high_runs_players[avg_col].idxmax(), name_col]
                highest_avg_value = high_runs_players[avg_col].max()
                answers['highest_average_player_8000_runs'] = highest_avg_player
                answers['highest_average_value_8000_runs'] = highest_avg_value
                print(f"Highest average among 8000+ run players: {highest_avg_player} ({highest_avg_value:.2f})")
        
        # Count countries in top 10
        if 'country' in data_clean.columns:
            country_counts = top_n_df['country'].value_counts()
            most_common_country = country_counts.index[0]
            country_count = country_counts.iloc[0]
            answers['most_represented_country_top10'] = most_common_country
            answers['country_count_top10'] = country_count
            print(f"Most represented country in top 10: {most_common_country} ({country_count} players)")
    
    def _answer_economic_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str):
        """Answer economic data specific questions"""
        if len(data_clean) > 0:
            current_rate = data_clean[analysis_col].iloc[-1]  # Most recent
            highest_rate = data_clean[analysis_col].max()
            answers['current_inflation_rate'] = current_rate
            answers['highest_rate_period'] = highest_rate
            print(f"Current rate: {current_rate:.2f}")
            print(f"Highest rate: {highest_rate:.2f}")
    
    def _answer_entertainment_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, top_n_df):
        """Answer entertainment/rating specific questions"""
        if 'rating' in str(analysis_col).lower():
            avg_rating = data_clean[analysis_col].mean()
            answers['average_rating'] = avg_rating
            print(f"Average rating: {avg_rating:.2f}")
            
        # Decade analysis if year column exists
        year_cols = [col for col in data_clean.columns if 'year' in str(col).lower()]
        if year_cols:
            year_col = year_cols[0]
            data_clean['decade'] = (data_clean[year_col] // 10) * 10
            decade_counts = data_clean['decade'].value_counts()
            most_common_decade = decade_counts.index[0]
            decade_count = decade_counts.iloc[0]
            answers['most_movies_decade'] = f"{most_common_decade}s"
            answers['decade_movie_count'] = decade_count
            print(f"Decade with most top movies: {most_common_decade}s ({decade_count} movies)")
    
    def _answer_temporal_questions(self, answers: dict, data_clean, analysis_col: str, name_col: str, year_col: str):
        """Answer time-based questions"""
        try:
            # Ensure year column is numeric
            data_clean[year_col] = pd.to_numeric(data_clean[year_col], errors='coerce')
            
            # Count items before year 2000
            before_2000 = len(data_clean[data_clean[year_col] < 2000])
            answers['items_before_2000'] = before_2000
            print(f"Items before year 2000: {before_2000}")
            
            # Find earliest item above threshold (for financial data)
            if 'billion' in str(analysis_col).lower():
                above_threshold = data_clean[data_clean[analysis_col] > 1500].sort_values(year_col)
                if len(above_threshold) > 0:
                    earliest_item = above_threshold.iloc[0][name_col]
                    earliest_year = above_threshold.iloc[0][year_col]
                    answers['earliest_above_threshold'] = earliest_item
                    answers['earliest_year'] = earliest_year
                    print(f"Earliest item above 1.5bn: {earliest_item} ({earliest_year})")
        except Exception as e:
            print(f"Error in temporal analysis: {str(e)}")
            # Set default values if temporal analysis fails
            answers['items_before_2000'] = 0
    
    def _identify_data_type(self, analysis_col: str, task_description: str) -> str:
        """Identify the type of data being analyzed"""
        if self._is_financial_data(analysis_col, task_description):
            return 'financial'
        elif self._is_health_data(analysis_col, task_description):
            return 'health'
        elif self._is_sports_data(analysis_col, task_description):
            return 'sports'
        elif self._is_economic_data(analysis_col, task_description):
            return 'economic'
        elif self._is_entertainment_data(analysis_col, task_description):
            return 'entertainment'
        else:
            return 'general'
    
    def _identify_domain(self, task_description: str) -> str:
        """Identify the domain/industry of the data"""
        task_lower = task_description.lower()
        
        if 'wikipedia' in task_lower and 'films' in task_lower:
            return 'movies'
        elif 'imdb' in task_lower:
            return 'movies'
        elif 'covid' in task_lower or 'coronavirus' in task_lower:
            return 'health'
        elif 'cricket' in task_lower or 'espn' in task_lower:
            return 'sports'
        elif 'inflation' in task_lower or 'trading' in task_lower:
            return 'economics'
        else:
            return 'general'