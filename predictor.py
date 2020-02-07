# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque, defaultdict
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Instances


WINDOW_NAME = 'predictor test'


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        print("metadata: ", self.metadata)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        # counter when ball detected for metrics purposes
        self.counts = {
            'frames': 0,
            'total': 0,
            'score_way': 0,
            'near_way': 0,
            'no_near_score_way': 0,
            'normal_way': 0,
            'candidate_way': 0,
            'candidate_way_detailed': defaultdict(int),
        }

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata,
                                instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(
                    predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(
                    frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(
                        dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            prev_prediction = None
            prev_center = None
            prev_size = None
            for frame in frame_gen:
                self.counts['frames'] += 1

                # predict in normal way
                prediction = self.predictor(frame)

                # try to get prominent instace
                instance = self.get_prominent_instance(
                    prediction, prev_center, prev_size)

                if instance is not None:  # found a ball
                    # print("prediction: ", prediction)
                    self.counts['normal_way'] += 1

                    # set only prominent instance
                    prediction['instances'] = instance

                    # update prediction for next iteration
                    prev_center, prev_size, prev_prediction = self.get_next_data(
                        prediction)

                    yield process_predictions(frame, prediction)
                elif prev_prediction is not None:   # there exists previous prediction
                    candidate_prediction = self.setProminentInstanceByProposal(
                        frame, prev_prediction['instances'], prev_center, prev_size
                    )

                    if candidate_prediction is not None:
                        # found prominent instance
                        self.counts['candidate_way'] += 1

                        # update prediction for next iteration
                        prev_center, prev_size, prev_prediction = self.get_next_data(
                            candidate_prediction)

                        yield process_predictions(frame, candidate_prediction)
                    else:
                        # make sure no prominent instance exist by setting empty instance
                        instances_len = len(prediction['instances'])
                        empty_instance = prediction['instances'][instances_len:]
                        prediction['instances'] = empty_instance

                        # to enable generator continuation with no prediction instance result
                        yield process_predictions(frame, prediction)

                else:   # haven't seen a ball yet
                    yield process_predictions(frame, prediction)

            self.counts['total'] = self.counts['normal_way'] + \
                self.counts['candidate_way']
            import json
            print('counts: \n', json.dumps(self.counts, indent=2))
            assert self.counts['total'] == self.counts['score_way'] + self.counts['near_way'] + \
                self.counts['no_near_score_way'], "total detected frame number is not matching"

    def get_box_size(self, pred_boxes, with_start=False):
        np_pred_boxes = pred_boxes.to(self.cpu_device).tensor.numpy()

        pred_box = np_pred_boxes[0]

        # change box points to integers
        x0, y0, x1, y1 = map(lambda real: int(round(real)), pred_box)
        assert y1 > y0, "Box y1 is not greater than y0"
        assert x1 > x0, "Box x1 is not greater than x0"

        h, w = y1 - y0, x1 - x0

        if with_start:
            return h, w, y0, x0
        else:
            return h, w

    def get_next_data(self, prediction):
        instance = prediction['instances']
        assert len(instance) == 1

        return instance.pred_boxes[0].get_centers(), self.get_box_size(instance.pred_boxes), prediction

    def normalize_box_size(self, prev_size, instance):
        prev_h, prev_w = prev_size
        h, w = self.get_box_size(instance.pred_boxes)

        score = instance.pred_scores[0]

        # new box size is affected by previous size based on its current score
        new_h = prev_h * (1 - score) + h * score
        new_w = prev_w * (1 - score) + w * score

        return new_h, new_w

    def get_near_instance(self, pred_instances, prev_center, prev_size):
        instances_len = len(pred_instances)

        min_dist = torch.dist(
            prev_center, pred_instances.pred_boxes[0].get_centers())
        min_index = 0
        for i in range(1, instances_len):
            dist = torch.dist(
                prev_center, pred_instances.pred_boxes[i].get_centers())
            if dist < min_dist:
                min_dist = dist
                min_index = i

        h, w = prev_size
        # if min_dist is out of twice circumscribing circle of previous rectangle don't consider vicinity
        if min_dist > 2*max(h, w):
            # if instance too far, no instance found
            return None
        else:
            # instances only support slicing not indexing, weird
            return pred_instances[min_index:min_index+1]

    def get_score_instance(self, pred_instances, thresh_score):
        instances_len = len(pred_instances)

        max_score = pred_instances.scores[0]
        max_index = 0
        for i in range(1, instances_len):
            if max_score < pred_instances.scores[i]:
                max_score = pred_instances.scores[i]
                max_index = i

        # since no previous prediction can support it with vicinity, expect a high score(threshold score)
        if max_score > thresh_score:
            # instances only support slicing not indexing, weird
            return pred_instances[max_index:max_index+1]
        else:
            # if score is too low from threshold score, no instance found
            return None

    def get_prominent_instance(self, prediction, prev_center, prev_size, thresh_score=0.5):
        pred_instances = prediction['instances']

        if len(pred_instances) == 0:
            return None

        # if no previous center found, take the box with the highest confidence score
        if prev_center is None:
            prominent_instance = self.get_score_instance(
                pred_instances, thresh_score)

            # TODO: metrics purpose counting(to be removed)
            if prominent_instance is not None:
                self.counts['score_way'] += 1
        else:
            prominent_instance = self.get_near_instance(
                pred_instances, prev_center, prev_size)
            if prominent_instance is None:
                # if no near instance is found, try to find an instance with attracting score
                prominent_instance = self.get_score_instance(
                    pred_instances, thresh_score)

                # TODO: metrics purpose counting(to be removed)
                if prominent_instance is not None:
                    self.counts['no_near_score_way'] += 1
            else:
                # TODO: metrics purpose counting(to be removed)
                self.counts['near_way'] += 1

        return prominent_instance

    def setProminentInstanceByProposal(self, img, prev_instances, prev_center, prev_size, thresh_score=0.5, scale_start=10, scale_diff=5, scale_end=40):
        orig_img_size = img.shape[:2]

        for percent in range(scale_start, scale_end, scale_diff):
            scale = percent / 100
            candidate_crop, new_origin = self.getBallProposal(
                img, prev_instances, scale)

            candidate_prediction = self.predictor(candidate_crop)

            # Visualize croped frame and its prediction
            # vis_frame = process_predictions(
            #             candidate_crop, candidate_prediction)
            # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            # cv2.imshow(WINDOW_NAME, vis_frame)
            # if cv2.waitKey(0) == 27:
            #     break  # esc to quit

            # continue if no prediction instance found
            if len(candidate_prediction['instances']) == 0:
                continue

            # transform prediction coordinates to original image
            self.transform_prediction(
                candidate_prediction, new_origin, orig_img_size)

            prominent_instance = self.get_prominent_instance(
                candidate_prediction, prev_center, prev_size, thresh_score)
            if prominent_instance is None:
                # no prominent instance found continue searching
                continue
            else:
                # TODO: metrics purpose counting(to be removed)
                self.counts['candidate_way_detailed'][scale] += 1

                # if satisfying prominent instance found break loop and return
                candidate_prediction['instances'] = prominent_instance
                return candidate_prediction

        return None

    def getBallProposal(self, img, prev_instances, scale=0.1):
        y_widen = scale
        x_widen = scale

        prev_boxes = prev_instances.pred_boxes
        img_h, img_w = prev_instances.image_size

        h, w, y0, x0 = self.get_box_size(prev_boxes, with_start=True)

        h_fract = int(round(y_widen * img_h))
        w_fract = int(round(x_widen * img_w))

        y0 = y0 - h_fract if y0 > h_fract else 0
        x0 = x0 - w_fract if x0 > w_fract else 0
        new_h = h + 2*h_fract   # 2* because to compensate y0 - h_fract effect
        new_w = w + 2*w_fract

        # ! don't need to check if w and h get out of image limit since python indexing is safe
        # check if w and h doesnot get out of image limit
        # new_h = img_h - y0 if new_h > img_h - y0 else new_h
        # new_w = img_w - x0 if new_w > img_w - x0 else new_w

        if len(img.shape) <= 3:
            return img[y0: y0 + new_h, x0: x0 + new_w], (x0, y0)
        else:
            return img[
                ..., y0: y0 + new_h, x0: x0 + new_w, :
            ], (x0, y0)

    def transform_prediction(self, pred, origin, orig_img_size):
        x0, y0 = origin
        boxes = pred['instances'].pred_boxes

        # works for multpile boxes
        boxes.tensor += torch.tensor([x0, y0, x0, y0],
                                     dtype=boxes.tensor.dtype, device=boxes.tensor.device)
        orig_h, orig_w = orig_img_size
        box_limit = torch.tensor([orig_w, orig_h, orig_w, orig_h],
                                 dtype=boxes.tensor.dtype, device=boxes.tensor.device)
        assert (boxes.tensor <= box_limit).all(
        ), "Transforming prediction boxes is making boxes point out of original image size"


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(
                gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(
                    cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
