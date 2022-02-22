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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    CORNERS_TO_TRACK = 1000
    BLOCK_SIZE = 7
    MIN_DISTANCE = 7
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    image_0 = frame_sequence[0]
    corners = cv2.goodFeaturesToTrack(image_0, CORNERS_TO_TRACK, 0.01,
                                      MIN_DISTANCE, blockSize=BLOCK_SIZE).reshape(-1, 2)
    ids = np.arange(corners.shape[0]).reshape(-1, 1)
    max_id = ids.max()

    frame_corners = FrameCorners(
        ids,
        corners,
        np.repeat(BLOCK_SIZE, corners.shape[0]),
    )
    builder.set_corners_at_frame(0, frame_corners)

    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        # track existing corners
        moved_corners, status, err = cv2.calcOpticalFlowPyrLK((255*image_0).astype("uint8"),
                                                              (255*image_1).astype("uint8"),
                                                              corners, None, **lk_params)

        status1d = status.reshape(-1)

        good_corners = moved_corners[status1d == 1]
        good_ids = ids[status1d == 1]

        # find new corners
        additional_corners = CORNERS_TO_TRACK - good_corners.shape[0]

        new_corners = np.ndarray((0, 2), dtype=np.float32)
        new_ids = np.ndarray((0, 1), dtype=int)
        if additional_corners > 0:
            mask = 255 * np.ones_like(image_1, dtype=np.uint8)
            for x, y in [np.int32(corner) for corner in good_corners]:
                cv2.circle(mask, (x, y), MIN_DISTANCE, 0, -1)
            new_corners = cv2.goodFeaturesToTrack(image_1, additional_corners, 0.01,
                                                  MIN_DISTANCE, blockSize=BLOCK_SIZE, mask=mask)
            if new_corners is not None:
                new_corners = new_corners.reshape(-1, 2)
                new_ids = np.arange(max_id + 1, max_id + 1 + new_corners.shape[0]).reshape(-1, 1)
                max_id = max(max_id, new_ids.max())

        corners = np.vstack([good_corners, new_corners])
        ids = np.vstack([good_ids, new_ids])

        frame_corners = FrameCorners(
            ids,
            corners,
            np.repeat(BLOCK_SIZE, corners.shape[0]),
        )
        builder.set_corners_at_frame(frame, frame_corners)

        image_0 = image_1


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
