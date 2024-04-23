~/.bashrc 에 보면 `alias go='ros2 launch sybot_nav sybot_nav_launch.py` 으로 시작함
### sybot_nav_launch.py
1. startup_mode가 "mapping_mode"이면 `sybot_mapping_launch.py`을 실행
2. 아닐 경우 `sybot_auto_launch.py`을 실행

### sybot_auto_launch.py
- DB에서 startup_list 에 launch_list를 받아서 실행
- DB에 기록된 launch_list
	- motor_driver_launch.py
	- imu_launch.py
	- imu_wlt_launch.py
	- robot_pose_publisher_launch.py
	- sybot_description_launch.py
	- drive_navigator_launch.py
	- bms_launch.py
	- drive_module_launch.py
	- logistics_module_launch.py
	- charger_module_launch.py
	- agv_main_module_launch.py
	- peripheral_module_launch.py
	- io_manager_launch.py
	- communication_launch.py
	- sybot_joystick_launch.py
	- syswin_vision_matcher_launch.py
	- syswin_vision_localizator_launch.py
	- syswin_vision_obstacle_launch.py
	- syswin_vision_docker_launch.py
	- syswin_vision_reflector_launch.py
	- drive_pid_path_launch.py
