"""
RAG Evaluation Benchmark Dataset Module
"""

BENCHMARK_DATASET_VERSION = "1.0.0"

# Curated benchmark dataset containing 12 representative QA pairs
DEFAULT_BENCHMARK_QA = [
    {
        "question": "What is the total project budget?",
        "reference_answer": "The total project budget is $50,000 USD."
    },
    {
        "question": "Who was appointed to lead the engineering team?",
        "reference_answer": "Ahmed was appointed to lead the engineering team."
    },
    {
        "question": "What is the completion deadline for the project?",
        "reference_answer": "The project completion deadline is the 15th of next month."
    },
    {
        "question": "Who is responsible for quality assurance and testing?",
        "reference_answer": "Fatima is responsible for QA and testing."
    },
    {
        "question": "How will project funds be disbursed?",
        "reference_answer": "All project funds will be disbursed directly through the project manager."
    },
    {
        "question": "What primary programming language will be used for backend development?",
        "reference_answer": "Python will be used for backend development."
    },
    {
        "question": "What is the agreed frequency of team status syncs?",
        "reference_answer": "Team status syncs will take place weekly on Mondays."
    },
    {
        "question": "Where will project documentation be hosted?",
        "reference_answer": "Project documentation will be hosted on the central team Notion workspace."
    },
    {
        "question": "What deployment platform will be used for staging?",
        "reference_answer": "Docker containers deployed on AWS EC2 will be used for staging."
    },
    {
        "question": "What security standard must all API endpoints comply with?",
        "reference_answer": "All API endpoints must comply with OAuth 2.0 and TLS 1.3 encryption."
    },
    {
        "question": "What is the expected target code coverage for unit tests?",
        "reference_answer": "The target code coverage for unit tests is 85%."
    },
    {
        "question": "Who should team members contact if they have blocker questions?",
        "reference_answer": "Team members should contact Ahmed or the project manager directly."
    }
]

# Benchmark transcript context corresponding to DEFAULT_BENCHMARK_QA
BENCHMARK_TRANSCRIPT_SAMPLE = """
Welcome everyone to the project kickoff meeting.
Today we are confirming key parameters for the upcoming AI Video Assistant initiative.

First, the total project budget has been approved at fifty thousand dollars ($50,000 USD).
All project funds will be disbursed directly through the project manager upon milestone approvals.

For team structure, Ahmed has been appointed as the technical lead for engineering.
Fatima will take charge of quality assurance, testing, and compliance.
If any team member encounters critical technical blockers, please reach out to Ahmed or the project manager immediately.

Regarding technical stack and standards, Python is our primary backend language.
All API endpoints must enforce OAuth 2.0 authentication and TLS 1.3 encryption.
Our target unit test coverage is established at 85%.

For project infrastructure, documentation will be hosted on our Notion workspace.
Staging deployments will run using Docker containers on AWS EC2.
Weekly status syncs are scheduled every Monday at 10:00 AM.

Finally, the hard deadline for completing Phase 1 development is the 15th of next month.
"""


def get_benchmark_dataset(version: str = BENCHMARK_DATASET_VERSION):
    """Returns the benchmark dataset dictionary and metadata."""
    return {
        "version": version,
        "qa_pairs": DEFAULT_BENCHMARK_QA,
        "transcript": BENCHMARK_TRANSCRIPT_SAMPLE
    }
