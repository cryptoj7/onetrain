# pylint: disable=no-member

import os
import cv2
import torch
from app.util import TrainArgs, Obj
from app.config import get_config


config = None


def dynamicrange(image, epsilon=1e-10):
    tensor = torch.from_numpy(image)
    I_min = torch.min(tensor)
    I_max = torch.max(tensor)
    I_min = torch.clamp(I_min, min=epsilon)
    I_max = torch.clamp(I_max, min=epsilon)
    dynamic_range_db = 20 * torch.log10(I_max / I_min)
    if dynamic_range_db < config.min_dynamic_range:
        raise ValueError(f'face-range: {dynamic_range_db}')


def size(img):
    h, w, _c = img.shape
    if h * w < config.min_size:
        raise ValueError(f'image-size: {w*h}')
    if h < config.min_height:
        raise ValueError(f'image-height: {h}')
    if w < config.min_width:
        raise ValueError(f'image-width: {w}')

def blur(image):
    bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = 1 / cv2.Laplacian(bw, cv2.CV_64F).var()
    variance = round(variance, 2)
    if variance > config.max_blur:
        raise ValueError(f'face-blur: {variance}')


def detect(image):
    import app.face
    app.face.load()
    faces, scores = app.face.detect(image, min_confidence=config.min_face_confidence / 4, max_detected=config.max_faces)
    if len(faces) == 0:
        raise ValueError('face-detected: none')
    if max(scores) < config.min_face_confidence:
        raise ValueError(f'face-confidence: {max(scores)}')
    face = faces[0]
    h, w, _c = face.shape
    if h < config.min_face_size:
        raise ValueError(f'face-height: {h}')
    if w < config.min_face_size:
        raise ValueError(f'face-width: {w}')
    return face


def similarity(reference, image):
    if reference is None:
        return
    import app.similarity
    res = app.similarity.distance(reference, image)
    if res > config.max_distance:
        raise ValueError(f'face-similarity: {round(1 - res, 2)}')


def validate(f, image, args: TrainArgs):
    global config # pylint: disable=global-statement
    config = get_config('validate')
    config = Obj(config)
    try:
        # image = cv2.imread(f)
        if image is None:
            raise ValueError('invalid')
        h, w, _c = image.shape
        if h == 0 or w == 0:
            raise ValueError('empty')
        if not args.validate:
            return None
        size(image)
        face = detect(image)
        blur(face)
        dynamicrange(face)
        similarity(args.reference, image)
    except Exception as e:
        # from app.logger import console
        # console.print_exception(max_frames=20)
        return { os.path.basename(f): str(e) }
    return None