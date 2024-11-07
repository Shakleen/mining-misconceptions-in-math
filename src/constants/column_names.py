from enum import Enum


class TrainCSVColumns(Enum):
    """Column names for the train dataset."""

    QUESTION_ID = "QuestionId"
    SUBJECT_ID = "SubjectId"
    CONSTRUCT_ID = "ConstructId"

    SUBJECT_NAME = "SubjectName"
    CONSTRUCT_NAME = "ConstructName"
    QUESTION_TEXT = "QuestionText"

    ANSWER_A_TEXT = "AnswerAText"
    ANSWER_B_TEXT = "AnswerBText"
    ANSWER_C_TEXT = "AnswerCText"
    ANSWER_D_TEXT = "AnswerDText"
    CORRECT_ANSWER = "CorrectAnswer"

    MISCONCEPTION_A_ID = "MisconceptionAId"
    MISCONCEPTION_B_ID = "MisconceptionBId"
    MISCONCEPTION_C_ID = "MisconceptionCId"
    MISCONCEPTION_D_ID = "MisconceptionDId"

    ANSWER_FORMAT = "Answer{option}Text"

    MISCONCEPTION_FORMAT = "Misconception{option}Id"


class MisconceptionsCSVColumns(Enum):
    """Column names for the misconceptions dataset."""

    MISCONCEPTION_ID = "MisconceptionId"
    MISCONCEPTION_NAME = "MisconceptionName"
