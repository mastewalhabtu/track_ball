# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

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
            counts = {
                'frames': 0,
                'normal_way': 0,
                'candidate_way': 0,
            }
            prev_prediction = None
            orig_img_size = None
            for frame in frame_gen:
                counts['frames'] += 1
                orig_img_size = frame.shape[:2]
                prediction = self.predictor(frame)
                # print("prediction: ", prediction)
                if len(prediction['instances']) > 0:  # found a ball
                    counts['normal_way'] += 1
                    prev_prediction = prediction    # update prediction for next iteration
                    yield process_predictions(frame, prediction)
                elif prev_prediction is not None:
                    candidate_crop, new_origin = self.getBallProposal(
                        frame, prev_prediction['instances'])

                    candidate_prediction = self.predictor(candidate_crop)
                    # print('candidate prediction: ', candidate_prediction)
                    vis_frame = process_predictions(
                        candidate_crop, candidate_prediction)

                    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    # cv2.imshow(WINDOW_NAME, vis_frame)
                    # if cv2.waitKey(0) == 27:
                    #     break  # esc to quit

                    if len(candidate_prediction['instances']) > 0:
                        counts['candidate_way'] += 1

                        # transform prediction coordinates to original image
                        self.transform_prediction(
                            candidate_prediction, new_origin, orig_img_size)
                        prev_prediction = candidate_prediction  # update prediction for next iteration

                    # to enable generator continuation only
                    yield process_predictions(frame, prediction)
                else:   # haven't seen a ball yet
                    yield process_predictions(frame, prediction)
            print('counts: ', counts)

    def getBallProposal(self, img, prev_instances):
        y_widen = 0.1
        x_widen = 0.1

        prev_boxes = prev_instances.pred_boxes
        img_h, img_w = prev_instances.image_size
        np_prev_boxes = prev_boxes.to(self.cpu_device).tensor.numpy()

        prev_box = np_prev_boxes[0]

        # change box points to integers
        x0, y0, x1, y1 = map(lambda real: int(round(real)), prev_box)
        assert y1 > y0, "Box y1 is not greater than y0"
        assert x1 > x0, "Box x1 is not greater than x0"

        h, w = y1 - y0, x1 - x0

        h_fract = int(round(y_widen * img_h))
        w_fract = int(round(x_widen * img_w))

        y0 = y0 - h_fract if y0 > h_fract else 0
        x0 = x0 - w_fract if x0 > w_fract else 0
        h = h + 2*h_fract   # 2* because to compensate y0 - h_fract effect
        w = w + 2*w_fract

        # ! don't need to check if w and h get out of image limit since python indexing is safe
        # check if w and h doesnot get out of image limit
        # h = img_h - y0 if h > img_h - y0 else h
        # w = img_w - x0 if w > img_w - x0 else w

        if len(img.shape) <= 3:
            return img[y0: y0 + h, x0: x0 + w], (x0, y0)
        else:
            return img[
                ..., y0: y0 + h, x0: x0 + w, :
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
