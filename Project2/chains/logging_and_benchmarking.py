"""
Logging, benchmarking, and accuracy tracking infrastructure for data analysis workflows.
Implements comprehensive logging at each step and accuracy benchmarking capabilities.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class WorkflowStepLog:
    """Log entry for a single workflow step"""
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed
    input_summary: str = ""
    output_summary: str = ""
    error_message: str = ""
    performance_metrics: Dict[str, Any] = None
    memory_usage_mb: float = 0.0
    data_shape_before: Optional[tuple] = None
    data_shape_after: Optional[tuple] = None

    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}

    @property
    def duration_seconds(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def complete(self, status: str = "completed", error_message: str = ""):
        """Mark the step as completed"""
        self.end_time = datetime.now()
        self.status = status
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        result['duration_seconds'] = self.duration_seconds
        return result


@dataclass
class WorkflowExecution:
    """Complete workflow execution log"""
    workflow_id: str
    workflow_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    steps: List[WorkflowStepLog] = None
    input_data_summary: Dict[str, Any] = None
    output_data_summary: Dict[str, Any] = None
    error_message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.input_data_summary is None:
            self.input_data_summary = {}
        if self.output_data_summary is None:
            self.output_data_summary = {}
        if self.metadata is None:
            self.metadata = {}

    @property
    def total_duration_seconds(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def add_step(self, step_log: WorkflowStepLog):
        """Add a step log to the execution"""
        self.steps.append(step_log)

    def complete(self, status: str = "completed", error_message: str = ""):
        """Mark the workflow execution as completed"""
        self.end_time = datetime.now()
        self.status = status
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        result['total_duration_seconds'] = self.total_duration_seconds
        result['steps'] = [step.to_dict() for step in self.steps]
        return result


class WorkflowLogger:
    """
    Enhanced logging system for data analysis workflows.
    Tracks input, intermediate, and output at each step.
    """

    def __init__(self, log_directory: str = "logs", enable_file_logging: bool = True):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)

        self.enable_file_logging = enable_file_logging
        self.current_execution: Optional[WorkflowExecution] = None
        self.logger = logging.getLogger(__name__)

        # Setup detailed logging format
        self._setup_logging()

    def _setup_logging(self):
        """Setup detailed logging configuration"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(workflow_id)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if self.enable_file_logging:
            # Create daily log files
            today = datetime.now().strftime("%Y%m%d")
            log_file = self.log_directory / f"workflow_{today}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)

            self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)

    def start_workflow(self, workflow_id: str, workflow_type: str, input_data: Dict[str, Any]) -> WorkflowExecution:
        """Start logging a new workflow execution"""
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=datetime.now(),
            input_data_summary=self._summarize_input_data(input_data)
        )

        self.current_execution = execution

        self.logger.info(
            f"Started workflow: {workflow_type}",
            extra={'workflow_id': workflow_id}
        )

        return execution

    def start_step(self, step_name: str, input_data: Dict[str, Any]) -> WorkflowStepLog:
        """Start logging a workflow step"""
        step_log = WorkflowStepLog(
            step_name=step_name,
            start_time=datetime.now(),
            input_summary=self._summarize_step_input(input_data),
            memory_usage_mb=self._get_memory_usage(),
            data_shape_before=self._get_data_shape(input_data.get('data'))
        )

        if self.current_execution:
            self.current_execution.add_step(step_log)

        self.logger.info(
            f"Started step: {step_name}",
            extra={'workflow_id': self.current_execution.workflow_id if self.current_execution else 'unknown'}
        )

        return step_log

    def complete_step(self, step_log: WorkflowStepLog, output_data: Dict[str, Any],
                     status: str = "completed", error_message: str = ""):
        """Complete logging for a workflow step"""
        step_log.complete(status, error_message)
        step_log.output_summary = self._summarize_step_output(output_data)
        step_log.data_shape_after = self._get_data_shape(output_data.get('data'))
        step_log.memory_usage_mb = self._get_memory_usage()

        # Add performance metrics
        step_log.performance_metrics.update({
            'duration_seconds': step_log.duration_seconds,
            'memory_delta_mb': step_log.memory_usage_mb,
            'data_size_change': self._calculate_data_size_change(
                step_log.data_shape_before,
                step_log.data_shape_after
            )
        })

        self.logger.info(
            f"Completed step: {step_log.step_name} ({step_log.duration_seconds:.2f}s)",
            extra={'workflow_id': self.current_execution.workflow_id if self.current_execution else 'unknown'}
        )

        if status == "failed":
            self.logger.error(
                f"Step failed: {step_log.step_name} - {error_message}",
                extra={'workflow_id': self.current_execution.workflow_id if self.current_execution else 'unknown'}
            )

    def complete_workflow(self, output_data: Dict[str, Any], status: str = "completed",
                         error_message: str = ""):
        """Complete logging for the entire workflow"""
        if not self.current_execution:
            return

        self.current_execution.complete(status, error_message)
        self.current_execution.output_data_summary = self._summarize_output_data(output_data)

        # Save execution log to file
        if self.enable_file_logging:
            self._save_execution_log(self.current_execution)

        self.logger.info(
            f"Completed workflow: {self.current_execution.workflow_type} "
            f"({self.current_execution.total_duration_seconds:.2f}s)",
            extra={'workflow_id': self.current_execution.workflow_id}
        )

        if status == "failed":
            self.logger.error(
                f"Workflow failed: {self.current_execution.workflow_type} - {error_message}",
                extra={'workflow_id': self.current_execution.workflow_id}
            )

        # Reset for next workflow
        completed_execution = self.current_execution
        self.current_execution = None

        return completed_execution

    def _summarize_input_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize input data for logging"""
        summary = {
            'keys': list(input_data.keys()),
            'task_description': input_data.get('task_description', '')[:200] + '...'
                              if len(input_data.get('task_description', '')) > 200 else input_data.get('task_description', ''),
            'source_type': input_data.get('source_type', 'unknown'),
            'has_files': bool(input_data.get('files')),
            'has_url': bool(input_data.get('url')),
            'timestamp': datetime.now().isoformat()
        }

        # Add data shape if available
        if 'data' in input_data:
            summary['data_shape'] = self._get_data_shape(input_data['data'])

        return summary

    def _summarize_step_input(self, input_data: Dict[str, Any]) -> str:
        """Summarize step input for logging"""
        keys = list(input_data.keys())
        data_shape = self._get_data_shape(input_data.get('data'))

        summary_parts = [f"Keys: {keys}"]
        if data_shape:
            summary_parts.append(f"Data shape: {data_shape}")

        return " | ".join(summary_parts)

    def _summarize_step_output(self, output_data: Dict[str, Any]) -> str:
        """Summarize step output for logging"""
        keys = list(output_data.keys())
        data_shape = self._get_data_shape(output_data.get('data'))

        summary_parts = [f"Output keys: {keys}"]
        if data_shape:
            summary_parts.append(f"Data shape: {data_shape}")

        # Add any errors or warnings
        if 'error' in output_data:
            summary_parts.append(f"Error: {str(output_data['error'])[:100]}")
        if 'issues_found' in output_data:
            issues = output_data['issues_found']
            if issues:
                summary_parts.append(f"Issues: {len(issues)} found")

        return " | ".join(summary_parts)

    def _summarize_output_data(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize final output data for logging"""
        return {
            'keys': list(output_data.keys()),
            'status': output_data.get('status', 'unknown'),
            'has_error': 'error' in output_data,
            'workflow_type': output_data.get('workflow_type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }

    def _get_data_shape(self, data) -> Optional[tuple]:
        """Get data shape if available"""
        if hasattr(data, 'shape'):
            return data.shape
        elif isinstance(data, list):
            return (len(data),)
        elif isinstance(data, dict):
            return (len(data),)
        return None

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _calculate_data_size_change(self, shape_before: Optional[tuple],
                                  shape_after: Optional[tuple]) -> Dict[str, Any]:
        """Calculate change in data size between steps"""
        if not shape_before or not shape_after:
            return {'change': 'unknown'}

        size_before = shape_before[0] if shape_before else 0
        size_after = shape_after[0] if shape_after else 0

        change = size_after - size_before
        change_pct = (change / size_before * 100) if size_before > 0 else 0

        return {
            'absolute_change': change,
            'percentage_change': change_pct,
            'size_before': size_before,
            'size_after': size_after
        }

    def _save_execution_log(self, execution: WorkflowExecution):
        """Save execution log to JSON file"""
        try:
            log_file = self.log_directory / f"execution_{execution.workflow_id}.json"
            with open(log_file, 'w') as f:
                json.dump(execution.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save execution log: {e}")


class AccuracyBenchmark:
    """
    Accuracy benchmarking system for data analysis workflows.
    Tracks performance against known datasets and expected outputs.
    """

    def __init__(self, benchmark_directory: str = "benchmarks"):
        self.benchmark_directory = Path(benchmark_directory)
        self.benchmark_directory.mkdir(exist_ok=True)

        # Create subdirectories for different benchmark types
        (self.benchmark_directory / "datasets").mkdir(exist_ok=True)
        (self.benchmark_directory / "expected_outputs").mkdir(exist_ok=True)
        (self.benchmark_directory / "results").mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def create_benchmark_suite(self, name: str, datasets: List[Dict[str, Any]],
                              expected_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new benchmark suite with known datasets and expected outputs"""
        suite = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'datasets': datasets,
            'expected_outputs': expected_outputs,
            'metadata': {
                'dataset_count': len(datasets),
                'total_size': sum(d.get('size', 0) for d in datasets)
            }
        }

        # Save suite definition
        suite_file = self.benchmark_directory / f"suite_{name}.json"
        with open(suite_file, 'w') as f:
            json.dump(suite, f, indent=2)

        self.logger.info(f"Created benchmark suite '{name}' with {len(datasets)} datasets")

        return suite

    def run_accuracy_benchmark(self, workflow_class, suite_name: str,
                             workflow_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run accuracy benchmark against a test suite"""

        # Load benchmark suite
        suite_file = self.benchmark_directory / f"suite_{suite_name}.json"
        if not suite_file.exists():
            raise ValueError(f"Benchmark suite '{suite_name}' not found")

        with open(suite_file, 'r') as f:
            suite = json.load(f)

        # Initialize workflow
        workflow_config = workflow_config or {}
        workflow = workflow_class(**workflow_config)

        benchmark_results = {
            'suite_name': suite_name,
            'workflow_class': workflow_class.__name__,
            'started_at': datetime.now().isoformat(),
            'results': [],
            'summary': {
                'total_tests': len(suite['datasets']),
                'passed': 0,
                'failed': 0,
                'average_accuracy': 0.0,
                'total_duration': 0.0
            }
        }

        # Run benchmark on each dataset
        for i, (dataset, expected_output) in enumerate(zip(suite['datasets'], suite['expected_outputs'])):
            self.logger.info(f"Running benchmark {i + 1}/{len(suite['datasets'])}: {dataset.get('name', f'dataset_{i}')}")

            test_result = self._run_single_benchmark(workflow, dataset, expected_output)
            benchmark_results['results'].append(test_result)

            # Update summary
            if test_result['passed']:
                benchmark_results['summary']['passed'] += 1
            else:
                benchmark_results['summary']['failed'] += 1

            benchmark_results['summary']['total_duration'] += test_result['duration_seconds']

        # Calculate final metrics
        total_tests = benchmark_results['summary']['total_tests']
        passed_tests = benchmark_results['summary']['passed']

        benchmark_results['summary']['accuracy_percentage'] = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        benchmark_results['completed_at'] = datetime.now().isoformat()

        # Save benchmark results
        results_file = self.benchmark_directory / "results" / f"benchmark_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)

        self.logger.info(
            f"Benchmark completed: {passed_tests}/{total_tests} tests passed "
            f"({benchmark_results['summary']['accuracy_percentage']:.1f}% accuracy)"
        )

        return benchmark_results

    def _run_single_benchmark(self, workflow, dataset: Dict[str, Any],
                            expected_output: Dict[str, Any]) -> Dict[str, Any]:
        """Run benchmark on a single dataset"""
        start_time = time.time()

        try:
            # Prepare input data
            input_data = dataset.get('input_data', {})

            # Execute workflow
            result = workflow.execute(input_data) if hasattr(workflow, 'execute') else {}

            # Compare with expected output
            accuracy_score = self._calculate_accuracy(result, expected_output)

            test_result = {
                'dataset_name': dataset.get('name', 'unknown'),
                'passed': accuracy_score >= 0.8,  # 80% threshold for passing
                'accuracy_score': accuracy_score,
                'duration_seconds': time.time() - start_time,
                'actual_output': self._sanitize_output_for_logging(result),
                'expected_output': expected_output,
                'error': None
            }

        except Exception as e:
            test_result = {
                'dataset_name': dataset.get('name', 'unknown'),
                'passed': False,
                'accuracy_score': 0.0,
                'duration_seconds': time.time() - start_time,
                'actual_output': None,
                'expected_output': expected_output,
                'error': str(e)
            }

        return test_result

    def _calculate_accuracy(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> float:
        """Calculate accuracy score between actual and expected outputs"""
        if not actual or not expected:
            return 0.0

        # Simple accuracy calculation based on key matching and value similarity
        total_keys = len(expected)
        matching_keys = 0

        for key, expected_value in expected.items():
            if key in actual:
                actual_value = actual[key]

                # Different comparison strategies based on data type
                if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                    # Numerical comparison with tolerance
                    tolerance = abs(expected_value) * 0.1  # 10% tolerance
                    if abs(actual_value - expected_value) <= tolerance:
                        matching_keys += 1
                elif isinstance(expected_value, str) and isinstance(actual_value, str):
                    # String comparison (case - insensitive)
                    if expected_value.lower() == actual_value.lower():
                        matching_keys += 1
                elif isinstance(expected_value, (list, dict)):
                    # Complex data structure comparison (simplified)
                    if str(expected_value) == str(actual_value):
                        matching_keys += 1
                else:
                    # Fallback: string representation comparison
                    if str(expected_value) == str(actual_value):
                        matching_keys += 1

        return matching_keys / total_keys if total_keys > 0 else 0.0

    def _sanitize_output_for_logging(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output data for logging (remove large objects, etc.)"""
        sanitized = {}

        for key, value in output.items():
            if key in ['plot_base64', 'image_data']:
                sanitized[key] = f"<base64_data:{len(str(value))}chars>"
            elif hasattr(value, 'shape'):  # DataFrames, arrays
                sanitized[key] = f"<data_object:shape={value.shape}>"
            elif isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:500] + "..."
            else:
                sanitized[key] = value

        return sanitized

    def get_benchmark_history(self, suite_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get benchmark history for a suite over the last N days"""
        results_dir = self.benchmark_directory / "results"
        cutoff_date = datetime.now() - timedelta(days=days)

        history = []

        for result_file in results_dir.glob(f"benchmark_{suite_name}_*.json"):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)

                # Check if result is within date range
                completed_at = datetime.fromisoformat(result.get('completed_at', '1900 - 01 - 01T00:00:00'))
                if completed_at >= cutoff_date:
                    history.append(result)

            except Exception as e:
                self.logger.error(f"Failed to load benchmark result {result_file}: {e}")

        # Sort by completion time
        history.sort(key=lambda x: x.get('completed_at', ''))

        return history


# Singleton instances for global use
workflow_logger = WorkflowLogger()
accuracy_benchmark = AccuracyBenchmark()


# Convenience functions
def log_workflow_execution(workflow_id: str, workflow_type: str, input_data: Dict[str, Any]) -> WorkflowExecution:
    """Start logging a workflow execution"""
    return workflow_logger.start_workflow(workflow_id, workflow_type, input_data)


def log_step_execution(step_name: str, input_data: Dict[str, Any]) -> WorkflowStepLog:
    """Start logging a step execution"""
    return workflow_logger.start_step(step_name, input_data)


def create_test_benchmark_suite():
    """Create a test benchmark suite for development / testing"""
    datasets = [
        {
            'name': 'test_wikipedia_gdp',
            'input_data': {
                'url': 'https://en.wikipedia.org / wiki / List_of_countries_by_GDP_(nominal)',
                'task_description': 'Extract GDP data and find the top 5 countries'
            },
            'size': 1000
        }
    ]

    expected_outputs = [
        {
            'top_countries': ['United States', 'China', 'Germany', 'Japan', 'India'],
            'data_extracted': True,
            'chart_created': True
        }
    ]

    return accuracy_benchmark.create_benchmark_suite('test_suite', datasets, expected_outputs)
