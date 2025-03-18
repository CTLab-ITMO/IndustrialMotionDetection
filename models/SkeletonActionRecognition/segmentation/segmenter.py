import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# TODO:: сделать препроцессинг
# TODO:: сюда нужно добавить семантическую сегментацию перед самой сегментацией, иначе это хрень какая-то
# TODO:: добавить логгирование


class Segmenter:
    def __init__(self, model_path, config_path, device):
        self.model_path = model_path
        self.model_cfg = config_path
        self._load_model(device)
        self.scores = None
        self.logits = None

    def _load_model(self, device):
        """ Загружаем модель """
        sam2_model = build_sam2(self.model_cfg, self.model_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def configure_params(self, image, points=None, labels=None):
        """ Выбираем лучшую маску """
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        self.scores = scores[sorted_ind]
        self.logits = logits[sorted_ind]

    def predict(self, image, points=None, labels=None):
        """ Запуск сегментации """
        self.predictor.set_image(image)
        mask_input = self.logits[np.argmax(self.scores), :, :]
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        return image, masks, scores
