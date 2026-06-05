"""Learning and knowledge reinforcement module for theGist pipeline.

This module provides quiz and trivia functionality built on top of the
ChromaDB insight store. Insights extracted from video transcripts are
used to generate multiple choice questions, allowing users to actively
test and reinforce their knowledge of the content.

Typical usage:
    >>> from src.learning import generate_quiz, evaluate_answer
    >>> quiz = generate_quiz("Example_Video_Title")
    >>> for question in quiz:
    ...     print(question["question"])
"""

import logging
import random
from typing import Optional

import ollama

from config import (
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
    OLLAMA_MODEL,
    QUIZ_CHOICES_COUNT,
    QUIZ_QUESTION_COUNT,
)
from src.storage import get_all_insights

# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _generate_question(insight: str) -> Optional[dict]:
    """Generates a multiple choice question from a single insight string.

    Sends the insight to the local Ollama model with a structured prompt
    requesting a question, a correct answer, and distractor options. The
    response is parsed into a structured question dictionary.

    Args:
        insight: A single insight string to base the question on.

    Returns:
        A dictionary containing the question data with the following keys:
            - question: The question string.
            - correct_answer: The correct answer string.
            - choices: A shuffled list of all answer strings including
              the correct answer and distractors.
        Returns None if the model response cannot be parsed into a valid
        question format.

    Example:
        >>> q = _generate_question("Spearmen counter cavalry effectively.")
        >>> print(q["question"])
        'Which unit type is most effective against cavalry?'
    """
    prompt = f"""You are a quiz generator. Given the following insight, create a multiple choice question.

Insight: {insight}

Respond in exactly this format and no other format:
QUESTION: <your question here>
CORRECT: <correct answer here>
WRONG1: <plausible but incorrect answer>
WRONG2: <plausible but incorrect answer>
WRONG3: <plausible but incorrect answer>

Rules:
- The question should test understanding of the insight
- Wrong answers should be plausible but clearly incorrect
- Keep all answers concise, under 15 words each
- Do not include any explanation or extra text"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response["message"]["content"].strip()
        return _parse_question(content, insight)

    except Exception as e:
        logger.warning(f"Question generation failed for insight: {e}")
        return None


def _parse_question(response: str, source_insight: str) -> Optional[dict]:
    """Parses the raw LLM response into a structured question dictionary.

    Extracts the question, correct answer, and wrong answers from the
    model response using the expected line prefix format. Shuffles the
    answer choices to randomise correct answer position.

    Args:
        response: The raw response string from the Ollama model,
            expected to follow the QUESTION/CORRECT/WRONG1-3 format.
        source_insight: The original insight string the question was
            generated from, stored for reference and debugging.

    Returns:
        A structured question dictionary if parsing succeeds, or None
        if any required field is missing from the response.

    Example:
        >>> parsed = _parse_question(raw_response, "Spearmen counter cavalry.")
        >>> print(parsed["correct_answer"])
        'Spearmen'
    """
    fields = {}
    for line in response.splitlines():
        line = line.strip()
        for key in ("QUESTION", "CORRECT", "WRONG1", "WRONG2", "WRONG3"):
            if line.startswith(f"{key}:"):
                fields[key] = line[len(f"{key}:"):].strip()

    required = ("QUESTION", "CORRECT", "WRONG1", "WRONG2", "WRONG3")
    if not all(k in fields for k in required):
        logger.warning(
            f"Incomplete question format received. "
            f"Found keys: {list(fields.keys())}"
        )
        return None

    choices = [
        fields["CORRECT"],
        fields["WRONG1"],
        fields["WRONG2"],
        fields["WRONG3"],
    ]
    random.shuffle(choices)

    return {
        "question": fields["QUESTION"],
        "correct_answer": fields["CORRECT"],
        "choices": choices,
        "source_insight": source_insight,
    }


def _select_insights(
    insights: list[str],
    count: int,
) -> list[str]:
    """Selects a random sample of insights to use for quiz generation.

    Randomly samples from the available insights up to the requested
    count. If fewer insights are available than requested, all insights
    are returned.

    Args:
        insights: The full list of available insight strings to
            sample from.
        count: The desired number of insights to select.

    Returns:
        A randomly sampled list of insight strings with length equal
        to min(count, len(insights)).

    Example:
        >>> selected = _select_insights(all_insights, 10)
        >>> print(len(selected))
        10
    """
    if len(insights) <= count:
        logger.info(
            f"Requested {count} insights but only {len(insights)} available. "
            f"Using all available insights."
        )
        return insights

    return random.sample(insights, count)


# ---------------------------------------------------------------------------
# Public Interface
# ---------------------------------------------------------------------------

def generate_quiz(
    source_name: str,
    question_count: int = QUIZ_QUESTION_COUNT,
) -> list[dict]:
    """Generates a multiple choice quiz from stored insights.

    Retrieves insights for the specified source from ChromaDB, randomly
    samples the configured number of insights, and generates a multiple
    choice question for each using the local Ollama model. Questions that
    fail to generate are skipped and logged as warnings.

    Args:
        source_name: The stem name of the source transcript to generate
            quiz questions from. Must match a source previously stored
            via storage.store_insights().
        question_count: The number of questions to include in the quiz.
            Defaults to the value configured in config.py.

    Returns:
        A list of question dictionaries where each dictionary contains:
            - question: The question string.
            - correct_answer: The correct answer string.
            - choices: A shuffled list of all answer strings.
            - source_insight: The insight the question was derived from.
        The list may contain fewer questions than requested if insufficient
        insights are available or if some questions fail to generate.

    Raises:
        ValueError: If no insights are found for the given source name.

    Example:
        >>> quiz = generate_quiz("Example_Video_Title", question_count=5)
        >>> print(f"Generated {len(quiz)} questions")
        Generated 5 questions
    """
    logger.info(
        f"Generating quiz for: {source_name} "
        f"({question_count} questions requested)"
    )

    insights = get_all_insights(source_name)

    if not insights:
        raise ValueError(
            f"No insights found for source: '{source_name}'. "
            "Ensure the video has been ingested, extracted, and stored."
        )

    selected = _select_insights(insights, question_count)
    quiz = []

    for i, insight in enumerate(selected, start=1):
        logger.info(f"Generating question {i}/{len(selected)}...")
        question = _generate_question(insight)
        if question:
            quiz.append(question)
        else:
            logger.warning(f"Skipped question {i} due to generation failure.")

    logger.info(f"Quiz generation complete. {len(quiz)} questions generated.")
    return quiz


def evaluate_answer(question: dict, user_answer: str) -> dict:
    """Evaluates a user's answer to a quiz question.

    Compares the user's answer against the correct answer for a question
    and returns a result dictionary containing feedback information.

    Args:
        question: A question dictionary as returned by generate_quiz(),
            containing at minimum the keys 'question', 'correct_answer',
            and 'source_insight'.
        user_answer: The answer string selected by the user.

    Returns:
        A result dictionary containing the following keys:
            - correct: Boolean indicating whether the answer was correct.
            - user_answer: The answer string provided by the user.
            - correct_answer: The correct answer string.
            - source_insight: The original insight the question came from.

    Example:
        >>> result = evaluate_answer(question, "Spearmen")
        >>> print(result["correct"])
        True
    """
    is_correct = user_answer.strip().lower() == question["correct_answer"].strip().lower()

    return {
        "correct": is_correct,
        "user_answer": user_answer,
        "correct_answer": question["correct_answer"],
        "source_insight": question["source_insight"],
    }


def run_quiz_session(source_name: str) -> dict:
    """Runs a complete interactive quiz session in the terminal.

    Generates a quiz for the given source, presents each question to
    the user via terminal input, evaluates responses, and returns a
    summary of the session results.

    Args:
        source_name: The stem name of the source transcript to quiz on.

    Returns:
        A session summary dictionary containing the following keys:
            - total: Total number of questions in the session.
            - correct: Number of correctly answered questions.
            - score_percent: Percentage score rounded to one decimal place.
            - results: List of individual result dictionaries from
              evaluate_answer() for each question in the session.

    Example:
        >>> summary = run_quiz_session("Example_Video_Title")
        >>> print(f"Score: {summary['score_percent']}%")
        Score: 80.0%
    """
    quiz = generate_quiz(source_name)

    if not quiz:
        logger.error("No questions could be generated. Ending session.")
        return {"total": 0, "correct": 0, "score_percent": 0.0, "results": []}

    print(f"\n{'='*60}")
    print(f"  theGist Quiz — {source_name.replace('_', ' ')}")
    print(f"  {len(quiz)} questions")
    print(f"{'='*60}\n")

    session_results = []

    for i, question in enumerate(quiz, start=1):
        print(f"Question {i}/{len(quiz)}")
        print(f"{question['question']}\n")

        for j, choice in enumerate(question["choices"], start=1):
            print(f"  {j}. {choice}")

        print()
        while True:
            raw = input(f"Your answer (1-{QUIZ_CHOICES_COUNT}): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= QUIZ_CHOICES_COUNT:
                selected = question["choices"][int(raw) - 1]
                break
            print(f"Please enter a number between 1 and {QUIZ_CHOICES_COUNT}.")

        result = evaluate_answer(question, selected)
        session_results.append(result)

        if result["correct"]:
            print("\n  Correct!\n")
        else:
            print(f"\n  Incorrect. The correct answer was: {result['correct_answer']}")
            print(f"  Insight: {result['source_insight']}\n")

        print(f"{'-'*60}\n")

    correct_count = sum(1 for r in session_results if r["correct"])
    score_pct = round((correct_count / len(quiz)) * 100, 1)

    print(f"\n{'='*60}")
    print(f"  Quiz Complete!")
    print(f"  Score: {correct_count}/{len(quiz)} ({score_pct}%)")
    print(f"{'='*60}\n")

    logger.info(f"Quiz session complete. Score: {correct_count}/{len(quiz)} ({score_pct}%)")

    return {
        "total": len(quiz),
        "correct": correct_count,
        "score_percent": score_pct,
        "results": session_results,
    }