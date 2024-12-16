import cv2
import depthai as dai
import numpy as np
import blobconverter

class TextHelper:
    def __init__(self):
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type)

    def rectangle(self, frame, p1, p2):
        cv2.rectangle(frame, p1, p2, self.bg_color, 6)
        cv2.rectangle(frame, p1, p2, self.color, 1)

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setPreviewSize(3840, 2160)  # 4K resolution
camRgb.setInterleaved(False)
camRgb.setFps(2)

# Full 4K output
frameOut = pipeline.create(dai.node.XLinkOut)
frameOut.setStreamName("color")
camRgb.video.link(frameOut.input)

# Scaled down 512x512 chunks for inference
scale_manip = pipeline.create(dai.node.ImageManip)
scale_manip.initialConfig.setResize(512, 512)
scale_manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
camRgb.video.link(scale_manip.inputImage)

# Neural network node
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.3)
nn.setBlobPath("model/qr_model_512x288_rvc2_openvino_2022.1_6shave.blob")  # Update with your model path
nn.input.setQueueSize(1)
nn.input.setBlocking(False)
scale_manip.out.link(nn.input)

# Neural network output
nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Helper functions
def expandDetection(det, percent=2):
    percent /= 100
    det.xmin -= percent
    det.ymin -= percent
    det.xmax += percent
    det.ymax += percent
    if det.xmin < 0: det.xmin = 0
    if det.ymin < 0: det.ymin = 0
    if det.xmax > 1: det.xmax = 1
    if det.ymax > 1: det.ymax = 1

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def sliding_window(image, step, window_size):
    """Generates 512x512 windows for inference."""
    for y in range(0, image.shape[0] - window_size + 1, step):
        for x in range(0, image.shape[1] - window_size + 1, step):
            yield x, y, image[y:y + window_size, x:x + window_size]

# Main loop
with dai.Device(pipeline) as device:
    qColor = device.getOutputQueue("color", maxSize=4, blocking=False)
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)
    c = TextHelper()

    while True:
        # Get the full 4K frame
        colorFrame = qColor.tryGet()
        if colorFrame is None:
            continue
        frame = colorFrame.getCvFrame()

        # Sliding window parameters
        step_size = 256  # Overlapping sliding window step
        window_size = 512  # Size of each window to send for inference

        for x, y, window in sliding_window(frame, step_size, window_size):
            # Preprocess the window for the neural network
            try:
                window_resized = cv2.resize(window, (512, 512), interpolation=cv2.INTER_AREA)
                img_frame = dai.ImgFrame()
                img_frame.setData(window_resized.tobytes())
                img_frame.setWidth(512)
                img_frame.setHeight(512)
                img_frame.setType(dai.ImgFrame.Type.BGR888p)

                # Send the window to the neural network
                device.getInputQueue("nn").send(img_frame)

                # Get detections for the window
                inDet = qDet.tryGet()
                if inDet:
                    detections = inDet.detections
                    for det in detections:
                        expandDetection(det)
                        bbox = frameNorm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                        c.rectangle(frame, (bbox[0] + x, bbox[1] + y), (bbox[2] + x, bbox[3] + y))
                        c.putText(frame, f"{int(det.confidence * 100)}%", (bbox[0] + 10 + x, bbox[1] + 20 + y))
            except Exception as e:
                print(f"Error processing window: {e}")

        # Display the full frame with bounding boxes
        cv2.imshow("4K Image", frame)

        if cv2.waitKey(1) == ord('q'):
            break
