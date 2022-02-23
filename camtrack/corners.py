#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

from typing import Tuple

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)

BLOCK_SIZE = 7
MIN_DISTANCE = 7
QUALITY_LEVEL = 0.02
EXISTING_CORNER_QUALITY_MODIFIER = 0.1


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def detect_corners(image, existing_corners, start_id, detection_params: dict):
    mask = None
    if existing_corners is not None:
        mask = 255 * np.ones_like(image, dtype=np.uint8)
        radius = detection_params["minDistance"]
        for x, y in [np.int32(corner) for corner in existing_corners]:
            cv2.circle(mask, (x, y), radius, 0, -1)

    detected_corners = cv2.goodFeaturesToTrack(image, 0, mask=mask, **detection_params)

    if detected_corners is not None:
        corners = detected_corners.reshape(-1, 2)
        ids = np.arange(start_id, start_id + corners.shape[0]).reshape(-1, 1)
        return corners, ids
    else:
        corners = np.ndarray((0, 2), dtype=np.float32)
        ids = np.ndarray((0, 1), dtype=int)
        return corners, ids


def get_quality(image, corners: np.ndarray, block_size: int, harris_k: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
    eigens = cv2.cornerEigenValsAndVecs(image, block_size, ksize=3)
    eigenvals: np.ndarray = np.transpose(eigens[:, :, :2], axes=(1, 0, 2))

    min_eigenvals = eigenvals.min(initial=None, axis=2)
    valid_positions = min_eigenvals.shape[0] - 1, min_eigenvals.shape[1] - 1
    int_corners = np.clip(corners.round().astype(int), a_min=0, a_max=valid_positions)
    corners_quality = min_eigenvals[int_corners[:, 0], int_corners[:, 1]].reshape(-1, 1)

    trace = eigenvals.sum(axis=2)
    harris_score = eigenvals.prod(axis=2) - harris_k * trace * trace
    harris_corners_score = harris_score[int_corners[:, 0], int_corners[:, 1]].reshape(-1, 1)
    return corners_quality, harris_corners_score


def process_initial_frame(image, start_id, detection_params):
    corners, ids = detect_corners(image, None, start_id, detection_params)
    corners_quality, harris_score = get_quality(image, corners, BLOCK_SIZE)
    return corners, ids, corners_quality, harris_score


def process_next_frame(image_0, image_1, corners, ids, next_id, detection_params, lk_params):
    # track existing corners
    moved_corners, status, err = cv2.calcOpticalFlowPyrLK((255 * image_0).astype("uint8"),
                                                          (255 * image_1).astype("uint8"),
                                                          corners, None, **lk_params)

    status1d = status.reshape(-1)

    good_corners = moved_corners[status1d == 1]
    good_ids = ids[status1d == 1]

    # find new corners
    new_corners, new_ids = detect_corners(image_1, good_corners, next_id, detection_params)

    # combine corners
    combined_corners = np.vstack([good_corners, new_corners])
    combined_ids = np.vstack([good_ids, new_ids])

    # calculate corners quality
    corners_quality, harris_score = get_quality(image_1, combined_corners, BLOCK_SIZE)
    max_quality = corners_quality.max(initial=0.0)

    # filter out low quality corners
    # same approach as in `goodFeaturesToTrack`
    # but because we are masking out existing corners, we have to do it manually
    min_quality_threshold = np.repeat(max_quality * QUALITY_LEVEL, combined_corners.shape[0])
    min_quality_threshold[:good_corners.shape[0]] *= EXISTING_CORNER_QUALITY_MODIFIER

    high_quality_corners_mask = corners_quality[:, 0] > min_quality_threshold
    high_quality_corners = combined_corners[high_quality_corners_mask]
    high_quality_ids = combined_ids[high_quality_corners_mask]

    return (high_quality_corners, high_quality_ids,
            corners_quality[high_quality_corners_mask], harris_score[high_quality_corners_mask])


def collect_results(all_results):
    corners = np.vstack([result[0] * 2 ** level for level, result in enumerate(all_results)])
    ids = np.vstack([result[1] for result in all_results])
    corners_quality = np.vstack([result[2] for result in all_results])
    harris_score = np.vstack([result[3] for result in all_results])
    sizes = np.vstack([np.repeat(BLOCK_SIZE * 2 ** level, result[0].shape[0]).reshape(-1, 1) for level, result in
                       enumerate(all_results)])
    return corners, ids, corners_quality, harris_score, sizes


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    PYRAMID_LEVELS = 3
    detection_params = {
        "qualityLevel": QUALITY_LEVEL,
        "minDistance": MIN_DISTANCE,
        "blockSize": BLOCK_SIZE,
    }
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    image_0 = frame_sequence[0]

    cur_image = image_0
    next_id = 0
    all_results = []
    for level in range(PYRAMID_LEVELS):
        result = process_initial_frame(cur_image, next_id, detection_params)
        all_results.append(result)

        cur_image = cv2.pyrDown(cur_image)
        next_id = max(next_id, result[1].max(initial=-1) + 1)

    corners, ids, corners_quality, harris_score, sizes = collect_results(all_results)
    frame_corners = FrameCorners(
        ids,
        corners,
        sizes,
        corners_quality,
        harris_score,
    )
    builder.set_corners_at_frame(0, frame_corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        prev_image = image_0
        cur_image = image_1
        cur_results = []
        for level, prev_results in enumerate(all_results):
            result = process_next_frame(prev_image, cur_image, prev_results[0], prev_results[1], next_id,
                                        detection_params, lk_params)
            cur_results.append(result)

            prev_image = cv2.pyrDown(prev_image)
            cur_image = cv2.pyrDown(cur_image)
            next_id = max(next_id, result[1].max(initial=-1) + 1)

        corners, ids, corners_quality, harris_score, sizes = collect_results(cur_results)

        frame_corners = FrameCorners(
            ids,
            corners,
            sizes,
            corners_quality,
            harris_score,
        )
        builder.set_corners_at_frame(frame, frame_corners)

        image_0 = image_1
        all_results = cur_results


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
