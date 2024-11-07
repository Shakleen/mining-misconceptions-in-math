class TrainCSVColumns:
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


class MisconceptionsCSVColumns:
    """Column names for the misconceptions dataset."""

    MISCONCEPTION_ID = "MisconceptionId"
    MISCONCEPTION_NAME = "MisconceptionName"


class ContrastiveCSVColumns:
    """Column names for the contrastive dataset."""

    QUESTION_ID = "QuestionId"
    SUBJECT_ID = "SubjectId"
    CONSTRUCT_ID = "ConstructId"
    QUESTION_DETAILS = "QuestionDetails"
    MISCONCEPTION_LIST = "MisconceptionList"
    LABEL = "Label"
    MISCONCEPTION_ID = "MisconceptionId"
    DELIMITER = "###"


class ContrastiveTorchDatasetColumns:
    """Column names for the random contrastive torch dataset."""

    QUESTION_IDS = "question_ids"
    QUESTION_MASK = "question_mask"
    MISCONCEPTION_IDS = "misconception_ids"
    MISCONCEPTION_MASK = "misconception_mask"
    LABEL = "label"
    META_DATA_QUESTION_ID = "meta_data_question_id"
    META_DATA_SUBJECT_ID = "meta_data_subject_id"
    META_DATA_CONSTRUCT_ID = "meta_data_construct_id"
    META_DATA_MISCONCEPTION_ID = "meta_data_misconception_id"
    META_DATA_SORTED_MISCONCEPTION_ID_LIST = "meta_data_sorted_misconception_id_list"
