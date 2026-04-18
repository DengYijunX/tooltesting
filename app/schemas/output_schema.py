from typing import TypedDict, List, Dict, Any


class ProblemStatement(TypedDict, total=False):
    title: str
    description: str
    input_format: str
    output_format: str
    constraints: str
    sample_input: str
    sample_output: str
    sample_explanation: str


class ReferenceSolution(TypedDict, total=False):
    language: str
    explanation: str
    code: str


class TestCase(TypedDict, total=False):
    case_id: int
    case_type: str
    input: str
    expected_output: str


class SandboxResult(TypedDict, total=False):
    passed: bool
    total_cases: int
    passed_cases: int
    details: List[Dict[str, Any]]


class FinalReview(TypedDict, total=False):
    valid: bool
    issues: List[str]
    suggestions: List[str]


class KnowledgeCheck(TypedDict, total=False):
    knowledge_sufficiency: bool
    knowledge_sufficiency_issues: List[str]
    knowledge_stats: Dict[str, Any]


class ConsistencyCheck(TypedDict, total=False):
    consistency_passed: bool
    consistency_issues: List[str]
