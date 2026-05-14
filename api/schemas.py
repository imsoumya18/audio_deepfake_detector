from pydantic import BaseModel


class Scores(BaseModel):
    bonafide: float
    spoof: float


class PredictionResponse(BaseModel):
    label: str        # "bonafide" or "spoof"
    confidence: float # probability of the predicted class
    scores: Scores    # full probability distribution
