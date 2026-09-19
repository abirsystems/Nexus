import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text


class LanguageDetector:
    def __init__(self, max_results=3, score_threshold=0.5):
        # LanguageDetectorOptions only takes base_options (no max_results/score_threshold
        # fields like TextClassifierOptions has), so those are applied ourselves in detect().
        self.max_results = max_results
        self.score_threshold = score_threshold
        base_options = python.BaseOptions(model_asset_path='models/detector.tflite')
        options = text.LanguageDetectorOptions(base_options=base_options)
        self.detector = text.LanguageDetector.create_from_options(options)

    def detect(self, text_input: str):
        result = self.detector.detect(text_input)
        # Each prediction has .language_code and .probability (not .category_name/.score,
        # which only apply to the generic TextClassifier task).
        predictions = [(pred.language_code, pred.probability) for pred in result.detections]
        predictions = [p for p in predictions if p[1] >= self.score_threshold]
        predictions.sort(key=lambda p: p[1], reverse=True)
        return predictions[: self.max_results]