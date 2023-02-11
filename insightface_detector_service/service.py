import numpy as np
import bentoml
import time
from bentoml.io import JSON
from bentoml.io import NumpyNdarray
from bentoml.io import Multipart

from typing import List, Dict, Any
from pydantic import BaseModel

from numpy.typing import NDArray


def build_runners():
    return {
        "insightface_detector": bentoml.torchscript.get("insightface_detector:latest").to_runner(),
    }


insightface_detector_input_spec = Multipart(data=NumpyNdarray())

insightface_detector_output_spec = Multipart(
    score_8=NumpyNdarray(),
    score_16=NumpyNdarray(),
    score_32=NumpyNdarray(),
    bbox_8=NumpyNdarray(),
    bbox_16=NumpyNdarray(),
    bbox_32=NumpyNdarray(),
    kps_8=NumpyNdarray(),
    kps_16=NumpyNdarray(),
    kps_32=NumpyNdarray(),
)

insightface_detector_nms_input_spec = Multipart(
    data=NumpyNdarray(), det_thresh=NumpyNdarray(), nms_thresh=NumpyNdarray()
)

insightface_detector_nms_output_spec = Multipart(
    boxes=NumpyNdarray(),
    scores=NumpyNdarray(),
    kpss=NumpyNdarray(),
)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets, nms_thresh):
    thresh = nms_thresh
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def build_apis(service, runners):
    @service.api(input=insightface_detector_input_spec, output=insightface_detector_output_spec)
    async def insightface_detector(data: NDArray[Any]) -> Dict[str, NDArray[Any]]:
        # data = np.asarray(input.data)
        raw_result = await runners["insightface_detector"].async_run(data)

        return {
            "score_8": raw_result[0].cpu().numpy(),
            "score_16": raw_result[1].cpu().numpy(),
            "score_32": raw_result[2].cpu().numpy(),
            "bbox_8": raw_result[3].cpu().numpy(),
            "bbox_16": raw_result[4].cpu().numpy(),
            "bbox_32": raw_result[5].cpu().numpy(),
            "kps_8": raw_result[6].cpu().numpy(),
            "kps_16": raw_result[7].cpu().numpy(),
            "kps_32": raw_result[8].cpu().numpy(),
        }

    @service.api(input=insightface_detector_nms_input_spec, output=insightface_detector_nms_output_spec)
    async def insightface_detector_nms(
        data: NDArray[Any], det_thresh: NDArray[Any], nms_thresh: NDArray[Any]
    ) -> Dict[str, NDArray[Any]]:
        start_time = time.time()
        # data = np.asarray(input.data)
        feat_stride_fpn = [8, 16, 32]
        fmc = 3
        num_anchors = 2
        input_height = data.shape[1]
        input_width = data.shape[2]
        center_cache = {}
        scores_list = []
        bboxes_list = []
        kpss_list = []
        raw_result = await runners["insightface_detector"].async_run(data)

        print(f"RUNNER {time.time() - start_time}")
        start_time = time.time()
        # result = self.con.modelrun(self.model_name, f"data_{job_id}", output_names)
        # net_outs = self.session.run(self.output_names, {self.input_name : blob})  # original function
        net_outs = [x[0, :, :].cpu().numpy() for x in raw_result]
        # print(net_outs[0][:5, :5])
        # for x in net_outs:
        #     print(x.shape)
        # exit()

        for idx, stride in enumerate(feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in center_cache:
                anchor_centers = center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
                if len(center_cache) < 100:
                    center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            kpss = distance2kps(anchor_centers, kps_preds)
            # kpss = kps_preds
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_inds]
            kpss_list.append(pos_kpss)
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = nms(pre_det, nms_thresh)
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        print(f"API {time.time() - start_time}")
        return {
            "boxes": det[:, :4],
            "scores": det[:, 4],
            "kpss": kpss,
        }
