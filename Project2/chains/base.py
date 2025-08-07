"""
Base classes and utilities for LangChain workflows
"""
import sys
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.callbacks.manager import CallbackManagerForChainRun


# Configuration constants
from config import get_chat_model, EMBEDDING_MODEL, OPENAI_API_KEY

# LangChain components
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain

# Ensure project root is on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseWorkflow(ABC):
    """Base class for LangChain workflows"""

    def __init__(self, llm=None, **kwargs):
        try:
            # Use the provided LLM or get the default one
            self.llm = llm or get_chat_model()

            object.__setattr__(self, "embeddings", OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY))

            # Setup memory
            object.__setattr__(
                self, "memory", ConversationBufferWindowMemory(k=10, return_messages=True)  # Keep last 10 interactions
            )

            logger.info(f"Initialized {self.__class__.__name__} with model {self.llm.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow with given input data"""
        pass

    def add_to_memory(self, human_input: str, ai_response: str):
        """Add interaction to memory"""
        self.memory.chat_memory.add_user_message(human_input)
        self.memory.chat_memory.add_ai_message(ai_response)


class DataAnalysisChain(BaseWorkflow):
    """Chain for data analysis tasks"""

    def __init__(self, llm=None, **kwargs):
        super().__init__(llm=llm)
        self.prompt_template = self._create_analysis_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_analysis_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for data analysis"""
        system_message = """You are a data analysis expert. Your task is to:
1. Understand the analysis request
2. Provide structured analysis approach
3. Suggest appropriate data processing steps
4. Recommend visualizations and insights
5. Identify potential issues or limitations

Be thorough, practical, and provide actionable recommendations."""

        human_message = """
Analysis Request: {task_description}

Data Context: {data_context}

Parameters: {parameters}

Please provide a comprehensive analysis plan including:
- Data preprocessing steps
- Analysis methodology
- Visualization recommendations
- Expected insights
- Potential challenges

Format your response as structured output with clear sections.
"""

        return ChatPromptTemplate.from_messages([("system", system_message), ("human", human_message)])

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute the data analysis chain"""
        try:
            # Prepare inputs
            task_description = inputs.get("task_description", "")
            data_context = inputs.get("data_context", "No data context provided")
            parameters = json.dumps(inputs.get("parameters", {}), indent=2)

            # Run the chain
            result = self.chain.run(task_description=task_description, data_context=data_context, parameters=parameters)

            # Parse and structure the result
            return {
                "analysis_result": result,
                "recommendations": self._extract_recommendations(result),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_used": self.model_name,
                    "input_tokens": len(task_description.split()),
                    "status": "completed",
                },
            }

        except Exception as e:
            logger.error(f"Error in data analysis chain: {e}")
            return {
                "analysis_result": f"Error occurred: {str(e)}",
                "recommendations": [],
                "metadata": {"timestamp": datetime.now().isoformat(), "status": "error", "error": str(e)},
            }

    def _extract_recommendations(self, result: str) -> List[str]:
        """Extract recommendations from the analysis result"""
        # Simple extraction logic - can be enhanced with more sophisticated parsing
        lines = result.split("\n")
        recommendations = []

        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ["recommend", "suggest", "should", "consider"]):
                if line and not line.startswith("#"):
                    recommendations.append(line)

        return recommendations[:5]  # Return top 5 recommendations

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async wrapper for execution"""
        return self._call(input_data)


class CodeGenerationChain(BaseWorkflow):
    """Chain for generating data analysis code"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_code_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_code_prompt(self) -> PromptTemplate:
        from utils.prompts import CODE_GENERATION_PROMPT
        return PromptTemplate(
            input_variables=[
                "task_description",
                "data_context",
                "libraries",
                "output_format"
            ],
            template=CODE_GENERATION_PROMPT
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Python code for data analysis"""
        try:
            result = self.chain.run(
                task_description=input_data.get("task_description", ""),
                data_context=input_data.get("data_context", "General dataset"),
                libraries=input_data.get("libraries", "pandas, matplotlib, seaborn"),
                output_format=input_data.get("output_format", "plots and summary statistics"),
            )

            return {"generated_code": result, "status": "success", "timestamp": datetime.now().isoformat()}

        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return {
                "generated_code": f"# Error generating code: {str(e)}",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class ReportGenerationChain(BaseWorkflow):
    """Chain for generating analysis reports"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt_template = self._create_report_prompt()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def _create_report_prompt(self) -> PromptTemplate:
        """Create prompt for report generation"""
        template = """Create a comprehensive data analysis report based on the following information:

Analysis Results: {analysis_results}
Data Summary: {data_summary}
Key Findings: {key_findings}
Target Audience: {audience}

Generate a professional report with the following structure:
1. Executive Summary
2. Data Overview
3. Key Findings
4. Detailed Analysis
5. Recommendations
6. Limitations and Assumptions
7. Next Steps

Make it clear, actionable, and appropriate for the target audience.
Use markdown formatting for better readability.
"""

        return PromptTemplate(
            input_variables=["analysis_results", "data_summary", "key_findings", "audience"], template=template
        )

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis report"""
        try:
            result = self.chain.run(
                analysis_results=input_data.get("analysis_results", ""),
                data_summary=input_data.get("data_summary", ""),
                key_findings=input_data.get("key_findings", ""),
                audience=input_data.get("audience", "technical team"),
            )

            return {
                "report": result,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "word_count": len(result.split()),
            }

        except Exception as e:
            logger.error(f"Error in report generation: {e}")
            return {
                "report": f"Error generating report: {str(e)}",
                "status": "error",
                "timestamp": datetime.now().isoformat(),
            }


class WorkflowOrchestrator:
    """Orchestrates multiple LangChain workflows"""

    def __init__(self):
        self.workflows = {
            "data_analysis": DataAnalysisChain(),
            "code_generation": CodeGenerationChain(),
            "report_generation": ReportGenerationChain(),
        }
        self.execution_history = []

    async def execute_workflow(self, workflow_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific workflow"""
        if workflow_type not in self.workflows:
            return {
                "error": f"Unknown workflow type: {workflow_type}",
                "available_workflows": list(self.workflows.keys()),
            }

        try:
            workflow = self.workflows[workflow_type]
            result = await workflow.execute(input_data)

            # Store execution history
            self.execution_history.append(
                {
                    "workflow_type": workflow_type,
                    "timestamp": datetime.now().isoformat(),
                    "input_data": input_data,
                    "result": result,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error executing workflow {workflow_type}: {e}")
            return {"error": str(e), "workflow_type": workflow_type, "timestamp": datetime.now().isoformat()}

    async def execute_multi_step_workflow(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a multi-step workflow where outputs can feed into next steps"""
        results = {}
        context = {}

        for i, step in enumerate(steps):
            workflow_type = step.get("workflow_type")
            input_data = step.get("input_data", {})

            # Inject context from previous steps
            if context:
                input_data.update(context)

            step_result = await self.execute_workflow(workflow_type, input_data)
            results[f"step_{i + 1}_{workflow_type}"] = step_result

            # Update context for next step
            if "analysis_result" in step_result:
                context["previous_analysis"] = step_result["analysis_result"]
            if "generated_code" in step_result:
                context["generated_code"] = step_result["generated_code"]

        return {
            "multi_step_results": results,
            "execution_summary": {
                "total_steps": len(steps),
                "completed_steps": len(results),
                "timestamp": datetime.now().isoformat(),
            },
        }

    def get_execution_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
