"""
Iterative reasoning and self - check workflow implementation.
Enhances accuracy with iterative validation loops and cross - model verification.
"""

import logging
from typing import Dict, Any, List
import json
from datetime import datetime
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from chains.base import BaseWorkflow
from .logging_and_benchmarking import workflow_logger, log_workflow_execution

logger = logging.getLogger(__name__)


class IterativeReasoningWorkflow(BaseWorkflow):
    """
    Workflow that implements iterative reasoning with self - check passes
    and cross - model verification for improved accuracy.
    """

    def __init__(self, llm=None, secondary_llm=None, max_iterations: int = 3,
                 confidence_threshold: float = 0.8, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.secondary_llm = secondary_llm  # For cross - model verification
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with iterative reasoning and self - checking"""

        workflow_id = f"iterative_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution = log_workflow_execution(workflow_id, "iterative_reasoning", input_data)

        try:
            task_description = input_data.get('task_description', '')
            data = input_data.get('data')

            # Initial analysis
            current_result = await self._initial_analysis(data, task_description)
            iteration_history = [current_result]

            # Iterative refinement
            for iteration in range(self.max_iterations):
                logger.info(f"Starting iteration {iteration + 1}/{self.max_iterations}")

                # Self - check current result
                self_check_result = await self._perform_self_check(
                    data, current_result, task_description, iteration_history
                )

                # Check if we should continue iterating
                if self_check_result['confidence_score'] >= self.confidence_threshold:
                    logger.info(
                        f"High confidence achieved "
                        f"({self_check_result['confidence_score']:.2f}), stopping iteration"
                    )
                    break

                # Refine the analysis based on self - check feedback
                refined_result = await self._refine_analysis(
                    data, current_result, self_check_result, task_description
                )

                current_result = refined_result
                iteration_history.append(refined_result)

            # Cross - model verification (if secondary LLM available)
            cross_verification = None
            if self.secondary_llm:
                cross_verification = await self._cross_model_verification(
                    data, current_result, task_description
                )

            # Final result compilation
            final_result = {
                'final_analysis': current_result,
                'iteration_history': iteration_history,
                'self_check_results': self_check_result,
                'cross_verification': cross_verification,
                'total_iterations': len(iteration_history),
                'final_confidence_score': self_check_result['confidence_score'],
                'workflow_type': 'iterative_reasoning',
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'max_iterations': self.max_iterations,
                    'confidence_threshold': self.confidence_threshold,
                    'used_cross_verification': self.secondary_llm is not None
                }
            }

            workflow_logger.complete_workflow(final_result, "completed")
            return final_result

        except Exception as e:
            logger.error(f"Iterative reasoning workflow failed: {e}")
            workflow_logger.complete_workflow({}, "failed", str(e))
            return {
                'error': str(e),
                'workflow_type': 'iterative_reasoning',
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }

    async def _initial_analysis(self, data: Any, task_description: str) -> Dict[str, Any]:
        """Perform initial analysis of the data"""

        system_prompt = """You are an expert data analyst. Perform a thorough initial analysis
        of the provided data and task. Be comprehensive but also identify areas where you might
        need additional validation or refinement."""

        human_prompt = """
        Task: {task_description}

        Data Summary: {data_summary}

        Please provide:
        1. Initial analysis and findings
        2. Key insights discovered
        3. Assumptions made during analysis
        4. Areas of uncertainty or potential error
        5. Confidence level in your findings (0 - 1 scale)

        Be honest about limitations and uncertainties.
        """

        data_summary = self._summarize_data(data)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            'task_description': task_description,
            'data_summary': data_summary
        })

        return {
            'analysis_text': result,
            'timestamp': datetime.now().isoformat(),
            'data_summary': data_summary,
            'iteration_number': 0
        }

    async def _perform_self_check(self, data: Any, current_result: Dict[str, Any],
                                task_description: str, iteration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform self - check validation of current results"""

        system_prompt = """You are a critical reviewer of data analysis. Your job is to
        validate analysis results against the original data and identify potential issues,
        inconsistencies, or areas for improvement."""

        human_prompt = """
        Original Task: {task_description}

        Data Summary: {data_summary}

        Current Analysis: {current_analysis}

        Previous Iterations: {iteration_summary}

        Please perform a critical self - check:
        1. Are the findings consistent with the data?
        2. Are there any logical inconsistencies?
        3. What assumptions might be incorrect?
        4. What additional analysis might be needed?
        5. How confident are you in these results? (0 - 1 scale)
        6. What specific improvements would you recommend?

        Provide a confidence score and specific recommendations for improvement.
        """

        iteration_summary = self._summarize_iteration_history(iteration_history)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        self_check_text = chain.invoke({
            'task_description': task_description,
            'data_summary': self._summarize_data(data),
            'current_analysis': current_result.get('analysis_text', ''),
            'iteration_summary': iteration_summary
        })

        # Extract confidence score from the response
        confidence_score = self._extract_confidence_score(self_check_text)

        return {
            'self_check_text': self_check_text,
            'confidence_score': confidence_score,
            'recommendations': self._extract_recommendations(self_check_text),
            'timestamp': datetime.now().isoformat()
        }

    async def _refine_analysis(self, data: Any, current_result: Dict[str, Any],
                             self_check_result: Dict[str, Any], task_description: str) -> Dict[str, Any]:
        """Refine analysis based on self - check feedback"""

        system_prompt = """You are a data analyst improving your analysis based on critical feedback.
        Address the specific concerns and recommendations from the self - check while maintaining
        accuracy and rigor."""

        human_prompt = """
        Original Task: {task_description}

        Data Summary: {data_summary}

        Previous Analysis: {previous_analysis}

        Self - Check Feedback: {self_check_feedback}

        Recommendations to Address: {recommendations}

        Please provide a refined analysis that:
        1. Addresses the specific concerns raised
        2. Incorporates the recommendations
        3. Maintains or improves accuracy
        4. Provides better evidence for conclusions
        5. Acknowledges remaining limitations

        Focus on the areas identified for improvement while building on valid insights from before.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        refined_text = chain.invoke({
            'task_description': task_description,
            'data_summary': self._summarize_data(data),
            'previous_analysis': current_result.get('analysis_text', ''),
            'self_check_feedback': self_check_result.get('self_check_text', ''),
            'recommendations': json.dumps(self_check_result.get('recommendations', []))
        })

        return {
            'analysis_text': refined_text,
            'timestamp': datetime.now().isoformat(),
            'iteration_number': current_result.get('iteration_number', 0) + 1,
            'addressed_feedback': self_check_result.get('recommendations', [])
        }

    async def _cross_model_verification(self, data: Any, final_result: Dict[str, Any],
                                     task_description: str) -> Dict[str, Any]:
        """Perform cross - model verification using a secondary LLM"""

        system_prompt = """You are an independent data analyst reviewing another analyst's work.
        Provide an objective assessment of their analysis and conclusions."""

        human_prompt = """
        Task: {task_description}

        Data Summary: {data_summary}

        Analysis to Review: {analysis_to_review}

        Please provide:
        1. Your independent assessment of the conclusions
        2. Points of agreement with the analysis
        3. Points of disagreement or concern
        4. Alternative interpretations you would consider
        5. Overall confidence in the analysis (0 - 1 scale)

        Be objective and focus on analytical rigor.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.secondary_llm | StrOutputParser()

        verification_text = chain.invoke({
            'task_description': task_description,
            'data_summary': self._summarize_data(data),
            'analysis_to_review': final_result.get('analysis_text', '')
        })

        verification_confidence = self._extract_confidence_score(verification_text)

        return {
            'verification_text': verification_text,
            'verification_confidence': verification_confidence,
            'agreement_level': self._assess_agreement_level(
                final_result.get('analysis_text', ''), verification_text
            ),
            'timestamp': datetime.now().isoformat(),
            'verifier_model': str(self.secondary_llm) if hasattr(self.secondary_llm, '__str__') else 'secondary_llm'
        }

    def _summarize_data(self, data: Any) -> str:
        """Create a summary of the data for prompts"""
        if data is None:
            return "No data provided"

        if hasattr(data, 'shape'):
            return f"DataFrame with shape {data.shape}, columns: {list(data.columns)}"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        elif isinstance(data, dict):
            return f"Dictionary with keys: {list(data.keys())}"
        else:
            return f"Data type: {type(data)}, length: {len(str(data))}"

    def _summarize_iteration_history(self, history: List[Dict[str, Any]]) -> str:
        """Summarize the iteration history for context"""
        if not history:
            return "No previous iterations"

        summary_parts = []
        for i, iteration in enumerate(history):
            summary_parts.append(f"Iteration {i}: {iteration.get('analysis_text', '')[:200]}...")

        return "\n".join(summary_parts)

    def _extract_confidence_score(self, text: str) -> float:
        """Extract confidence score from text (simple regex - based approach)"""
        import re

        # Look for patterns like "confidence: 0.8" or "confidence level: 80%"
        patterns = [
            r"confidence.*?(\d+\.?\d*)",
            r"confident.*?(\d+\.?\d*)%",
            r"(\d+\.?\d*)/1\.?0?",
            r"score.*?(\d+\.?\d*)"
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    score = float(matches[-1])  # Take the last match
                    # Normalize to 0 - 1 scale
                    if score > 1:
                        score = score / 100  # Convert percentage to decimal
                    return min(max(score, 0.0), 1.0)  # Clamp to [0,1]
                except ValueError:
                    continue

        # Default to medium confidence if no explicit score found
        return 0.6

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract specific recommendations from self - check text"""
        import re

        # Look for numbered lists or bullet points with recommendations
        recommendations = []

        # Pattern for numbered recommendations
        numbered_pattern = r"\d+\.?\s*([^.\n]+(?:\.[^.\n]*)?)"
        matches = re.findall(numbered_pattern, text)

        for match in matches:
            if any(keyword in match.lower() for keyword in ['recommend', 'suggest', 'should', 'improve', 'consider']):
                recommendations.append(match.strip())

        # If no numbered recommendations found, look for bullet points
        if not recommendations:
            bullet_pattern = r"[•\-\*]\s*([^.\n]+)"
            matches = re.findall(bullet_pattern, text)
            for match in matches:
                if any(keyword in match.lower() for keyword in ['recommend', 'suggest', 'should', 'improve', 'consider']):
                    recommendations.append(match.strip())

        return recommendations[:5]  # Limit to top 5 recommendations

    def _assess_agreement_level(self, original_analysis: str, verification_analysis: str) -> float:
        """Assess level of agreement between original and verification analysis"""
        # Simple keyword - based similarity assessment
        original_words = set(original_analysis.lower().split())
        verification_words = set(verification_analysis.lower().split())

        if not original_words or not verification_words:
            return 0.0

        intersection = original_words.intersection(verification_words)
        union = original_words.union(verification_words)

        # Jaccard similarity
        similarity = len(intersection) / len(union) if union else 0.0

        return similarity


class RetrievalAugmentedAnalysisWorkflow(BaseWorkflow):
    """
    Workflow that uses retrieval - augmented generation (RAG) to feed the model
    only the most relevant chunks from the data store.
    """

    def __init__(self, llm=None, embedding_model=None, vector_store=None, **kwargs):
        super().__init__(llm=llm, **kwargs)
        self.embedding_model = embedding_model or self.embeddings
        self.vector_store = vector_store

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG - enhanced analysis workflow"""

        try:
            task_description = input_data.get('task_description', '')
            data = input_data.get('data')

            # Step 1: Index the data for retrieval
            if self.vector_store:
                await self._index_data_for_retrieval(data, task_description)

            # Step 2: Retrieve relevant data chunks
            relevant_chunks = await self._retrieve_relevant_chunks(task_description, top_k=5)

            # Step 3: Perform analysis with retrieved context
            analysis_result = await self._analyze_with_retrieved_context(
                task_description, relevant_chunks
            )

            return {
                'analysis_result': analysis_result,
                'relevant_chunks': relevant_chunks,
                'retrieval_enhanced': True,
                'workflow_type': 'retrieval_augmented_analysis',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"RAG workflow failed: {e}")
            return {
                'error': str(e),
                'workflow_type': 'retrieval_augmented_analysis',
                'status': 'failed',
                'timestamp': datetime.now().isoformat()
            }

    async def _index_data_for_retrieval(self, data: Any, context: str):
        """Index data for retrieval (placeholder implementation)"""
        # This would implement actual vector indexing
        # For now, this is a placeholder
        logger.info("Indexing data for retrieval (placeholder implementation)")
        pass

    async def _retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most relevant data chunks (placeholder implementation)"""
        # This would implement actual similarity search
        # For now, return placeholder chunks
        return [
            {
                'content': f"Relevant data chunk {i + 1} for query: {query[:50]}...",
                'relevance_score': 0.9 - (i * 0.1),
                'chunk_id': f"chunk_{i + 1}"
            }
            for i in range(top_k)
        ]

    async def _analyze_with_retrieved_context(self, task_description: str,
                                           relevant_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform analysis using only the most relevant retrieved chunks"""

        system_prompt = """You are a data analyst with access to pre - selected, highly relevant
        data chunks. Focus your analysis on these specific pieces of information rather than
        making broad assumptions."""

        human_prompt = """
        Task: {task_description}

        Relevant Data Chunks:
        {relevant_chunks}

        Based only on the provided relevant data chunks:
        1. Perform focused analysis
        2. Draw evidence - based conclusions
        3. Identify what additional data might be needed
        4. Acknowledge limitations based on available chunks

        Stay grounded in the provided evidence.
        """

        chunks_text = "\n\n".join([
            f"Chunk {i + 1} (relevance: {chunk['relevance_score']:.2f}):\n{chunk['content']}"
            for i, chunk in enumerate(relevant_chunks)
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            'task_description': task_description,
            'relevant_chunks': chunks_text
        })

        return {
            'analysis_text': result,
            'chunks_used': len(relevant_chunks),
            'retrieval_method': 'vector_similarity',
            'timestamp': datetime.now().isoformat()
        }


# Factory functions for creating enhanced workflows
def create_iterative_reasoning_workflow(llm=None, secondary_llm=None, **kwargs):
    """Create an iterative reasoning workflow with optional cross - model verification"""
    return IterativeReasoningWorkflow(
        llm=llm,
        secondary_llm=secondary_llm,
        **kwargs
    )


def create_rag_workflow(llm=None, vector_store=None, **kwargs):
    """Create a retrieval - augmented analysis workflow"""
    return RetrievalAugmentedAnalysisWorkflow(
        llm=llm,
        vector_store=vector_store,
        **kwargs
    )
