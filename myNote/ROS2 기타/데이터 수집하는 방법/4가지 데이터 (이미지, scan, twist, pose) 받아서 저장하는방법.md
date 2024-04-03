## 이미지 데이터 publish 하는 방법
### Code
```python
class RealSensePublisher(Node):
    def __init__(self):
        super().__init__("realsense_publisher")
        self.realsense_pub_rgb = self.create_publisher(Image, "/rgb_frame", 10)

        timer_period = 0.05
        self.br_rgb = CvBridge()

        try:
            self.pipe = rs.pipeline()
            self.cfg = rs.config()
            self.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipe.start(self.cfg)
            self.timer = self.create_timer(timer_period, self.timer_callback)

        except Exception as e:
            print(e)
            self.get_logger().error("INTEL REALSENSE IS NOT CONNECTED")

  

    def timer_callback(self):
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        pub_image = self.br_rgb.cv2_to_imgmsg(color_image)
        pub_image.header.stamp = self.get_clock().now().to_msg()
        pub_image.header.frame_id = "1"

        self.realsense_pub_rgb.publish(pub_image)
        self.get_logger().info("Publishing rgb frame")
```
#### RealSense 카메라 설정
- Intel RealSense 카메라와의 통신을 위해 `rs.pipeline()`과 `rs.config()` 객체를 생성합니다.
- `enable_stream` 메소드를 호출하여 RGB 스트림을 활성화합니다. 여기서는 해상도 640x480, 색상 형식 BGR8, 프레임레이트 30fps로 설정합니다.
#### Timer callback
- `timer_callback` 메소드는 설정된 주기(`timer_period`)마다 자동으로 호출되어 카메라로 부터 이미지 프레임을 캡처하고 이를 `/rgb_frame` 토픽으로 발행
- `wait_for_frames` 메소드를 통해 최신 이미지 프레임을 가져온 후, `get_color_frame` 메소드로 RGB 프레임을 추출합니다.
- `cv2_to_imgmsg` 메소드로 OpenCV 이미지를 ROS **`Image`** 메세지로 변환하고 메시지의 `.header`에 타임스탬프와 프레임 ID를 설정
### Reference
https://github.com/nickredsox/youtube/blob/master/Robotics/youtube_robot/intel_pub.py

## 여러 topic을 message_filters를 이용해서 frequency 동기화 

### Code
```python
import message_filters
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

class DLTest(Node):
	def __init__(self):
		super().__init__("dl_test")
		self.rgb_sync = message_filters.Subscriber(self, Image, '/rgb_frame', qos_profile=qos_profile_sensor_data)

		self.scan_sync = message_filters.Subscriber(self, LaserScan, '/scan', qos_profile=qos_profile_sensor_data)

		self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sync, self.scan_sync], 10, 0.1, allow_headerless=True) # 10개의 message queue 아 0.1초의 시간 차이 허용
		self.sync.registerCallback(self.sync_callback)

	def sync_callback(self, rgb_data, scan_data)
		self.get_logger().info("sync callback!!")
		
```
- `ApproximateTimeSynchronizer`는 이 두 `Subcriber`를 받아, `10`개의 메시지 큐와 최대 `0.1`초의 시간 차이를 허용하면서, 두 메시지가 대략적으로 동시에 도착했을 때 `callback` 함수를 호출하도록 설정
- `allow_headerless=True` 옵션을 사용하면 `header`가 없는 메시지도 동기화할 수 있습니다. 이 옵션을 활성화하면 `message_filters`가 내부적으로 메시지를 수신한 시간을 기준으로 타임 스탬프를 자동으로 할당