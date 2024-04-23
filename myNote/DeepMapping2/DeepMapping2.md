
# Abstract

- Lidar mapping은 자율 주행 및 모바일 로봇에서 매우 중요하지만 아직도 도전과제로 있을만큼 어렵다.
- 이러한 글로벌 point cloud 등록 문제를 해결하기 위해 DeepMapping[1]은 복잡한 맵 추정을 간단한 deep network의 self-supervised 학습으로 변환한다.
- DeepMapping[1]은 여전히 수천개의 대규모 데이터셋에서는 만족스러운 결과를 내지 못한다.
	- loop closures와 정확한 프레임 간 포인트 대응의 부족
	- global localization network의 느린 수렴(convergence)
- 위의 문제를 해결하기 위해 2가지 새로운 기술을 추가한 DeepMapping2를 제안
	1) loop closin을 통한 map topology에 기반한 training batch 구성
	2) pairwise 등록을 활용한 self-supervised local-to-global point consistency loss

# Introduction

- 매핑은 자율 이동 에이전트에게 필수적인 능력입니다. 에이전트의 지역 센서 관측을 즉, 환경의 global 공간 표현인 맵으로 구성합니다.
- 사전에 구축된 맵은 로봇 공학, 자율 주행 및 증강 현실에서 에이전트가 자신의 위치를 파악하는데 유용합니다. 다양한 동시 위치 추정(localization) 및 Mapping인 SLAM 방법은 2d 및 3d센서에서 새로운 환경의 맵을 생성할 수 있습니다. 특히 LiDAR 기반 맵핑은 LiDAR의 직접적이고 정확한 3D point cloud 감지 기능 떄문에 자율 주행 및 모바일 로봇에서 대규모 맵을 구축하는데 자주 사용된다.
- visual slam과 유사하게, lidar slam방법은 일반적으로 front-end 및 back-end 모듈을 포함합니다. front-end 모듈은 lidar/inertial/wheel odometry를 통해 센서 이동을 추적하고, iterative closest point(ICP) 또는 3D feature detection 및 대응 매칭 알고리즘을 사용하여 순차적인 frame 간의 제약(연결) 조건을 제공 한다.
- back-end 이러한 제약 조건을 사용하여 odometry drift를 최소하기 위해 pose/pose-landmark graph 최적화를 수행한다. 이는 visual slam 및 structure-from-motion (SfM)의 bundle 조정과 유사하다.
- 그러나 정확한 GNSS/IMU 없이 odometry로서, 대규모 lidar 매핑 결과를 불만족스러울 수 있다. 이는 lidar odometry의 오차와 대응 매칭 및 루프 클로징에 대한 어려움 때문이다.
- 이러한 문제를 해결하기 위해 딥러닝 방법을 사용하여 lidar 매핑의 하위 작업을 deep network로 대체하는데 중점을 두고 일반적인 머신러닝 패러다임인 학습후 테스트를 따른다. 하지만 이러한 방법은 훈련 데이터 세트 도메인이 테스트 데이터와 다른 경우 일반화 문제에 직면할 수 있다.
- DeepMapping[1]은 새로운 point cloud mappin을 위한 최적화에 대한 훈련이라는 새로운 패러다임을 제안한다.
- 전역 등록을 point cloud 기반의 PoseNet[24] (L-net)에 캡슐화 하고 이를 BCE Loss를 사용하여 맵 품질을 평가하는 다른 binary occupancy network (M-net)으로 변환한다. 이는 연속적인 맵 최적화를 이진 분류인 self-supervised 학습으로 변환한다. 테스트가 필요하지 않기 때문에 훈련이 완료되면 매핑이 한번 수행되기 때문에 일반화 문제에 직면하지 않는다.
- 하지만 작은 데이터셋에서 매우 좋은 성능에도 불구하고 Deepmapping은 다음과 같은 이유로 큰 데이터셋에서 실패했다.
	1) No-explicit-loop-closure : DeepMapping은 각 미니 배치의 프레임을 시간적 이웃으로 사용하여 L-Net을 점진적으로 최적하고, 전역 맵 일관성을 제어하기 위해 M-Net에만 의존한다. 이는 frame 수가 많을 때 drift하게 되는 incremental registraion과 유사하다. SLAM은 루프 클로징에 의해 이 문제를 해결하며, DeepMapping에 어떻게 통할할지는 아직 명확하지 않다.
	2) No-local-registration : 이전 연구들[25-28]은 local registration이 부분적으로 정확하다는 것을 보여 주었지만, DeepMapping은 이 local-registration을 ICP기반의 pose 초기화에서만 사용하고 최적화 과정에는 사용하지 않는다. 이는 모든 lidar registration 방법이 직면하는 일반적인 문제로, lidar point cloud에서의 point 대응 부족으로 인한 것이다. 희소한 센서 해상도와 장거리 감지로 인해 동일한 3D point가 다른 scan에서 다시 등장하는 경우가 매우 드뭄
	3) Slow-convergence-in-global-registration : L-Net은 하나의 point cloud의 frame으로 global pose를 예측하며, 이는 오직 M-Net과 BCE Loss에 의해서 지도(supervised)된다. 이와 다르게 global registration은 올바른 pose를 출력하기에 충분한 추론 힌트가 부족하므로, 데이터셋이 클때 수렴이 느려짐
- 대규모 lidar 데이터셋에서 맵을 효과적으로 최적화할 수 있는 DeepMapping2를 제안한다. 이는 2가지 새로운 기술로 DeepMapping을 확장한다.
	1) 동일한 배치에 topology/공간적 이웃과 함께 frame을 그룹화함으로써 루프 클로징의 맵 토폴로지를 기반으로 데이터 프레임을 조직화하여 문제(1)을 해결합니다. 이는 free-space 불일치성을 사용하여 M-Net과 BCE loss를 통해 self-supervision을 생성하기 위한 딥매핑에 루프 클로징을 추가하는 가장 좋은 방법으로 판단된다. 왜냐하면 이러한 불일치성은 등록되지 않는 인접 프레임 사이에서 발생하기 때문입니다.
	2) 사전에 계산된 pairwise registration을 활용한 새로운 self-supervised local-to-global point consistency loss입니다. 각 프레임에서 각 포인트에 대해 우리는 새로운 consistence를 계산할 수 있다. 이는 이웃 프레임의 글로벌 포즈를 사용하여 계산된 해당 포인트의 글로벌 좌표의 다른 버전과 pairwise registration에서 두 프레임 간의 상대 포즈 사이의 L2 거리로 정의 된다. 이렇게 하면 다른 프레임 간의 포인트 대응에 의존하지 않고 문제(2)를 해결할 수 있다. 즉, 두 이웃 프레임이 pairwise registration에 대한 대응점으로서 충분한 공통점을 가지고 있지 않더라도, 훈련 중에 local registratio의 결과를 여전히 통합할 수 있다. 이는 새로운 consistency loss로 인해 이제 L-Net과 BCE Loss뿐만 아니라 더 강한 gradient로 지도되기 때문에 문제(3)을 해결한다.

# Method
## Overview
- 여러개의 point cloud들을 한개의 글로벌 프레임으로 등록하는 문제로 인식
	- 수식적으로 정의하면, input cloud는 $\mathcal{S}=S_{i=1}^K$ 으로 정의되고 여기서 $K$는 전체 point cloud의 수 각 point cloud $S_i$는 $\mathcal{N}_i×3$  행렬로 표현되고 $N_i$는 $S_i$에 포함된 point의 수이다. 목표는 센서 포즈 $\mathcal{T} = \{T_i\}_{i=1}^K$ 을 각 $S_i$ 에 대해 추정하는 것이다.
	- DeepMapping[1]에서 영감을 받아 $f$로 표현되는 DNN을 사용하여 센서 포즈 $\mathcal{T}$를 추정한다. DNN을  사용하면 registration problem은 optimal network parameter를 찾는 문제로 변환되고 parameter는 다음 objective function에 의해 최소화 된다.
	- $$(\theta^{*},\phi^{*}  )=\underset{\theta, \phi}{argmin}⁡ \mathcal{L}_{\phi}(f_{\theta}(\mathcal{S}),\mathcal{S}) \;\;\;\;\;\;\;\;\;\;\;\;\;\; (1)$$
	
	- 여기서 $f_θ:S_i→T_i$ 은 L-Net으로 각 point cloud에 대해 글로벌 포즈를 추정한다.
	- $\phi$ 는 글로벌 좌표에서 대응되는 occupancy probabilit로 매핑하는 맵 네트워크(M-Net) $m_{\phi}$ 의 parameter이다.
	- $\mathcal{L}_{\phi}$은 global registration quality를 측정하기 위한 self-supervised binary cross entropy loss 이다.
	- $$\mathcal{L}_{\phi}=\frac{1}{K}\sum\limits^{K}_{i=1}{B[m_{\phi}(G_{i},\;1)]+B[m_{\phi}(s(G_{i})),\;0]} \;\;\;\;\;\;\;\; (2)$$
	- 여기서 global point cloud $G_i$ 는 L-Net 매개변수 $\theta$의 함수이고, $s(G_{i})$는 $G_{i}$ 의 자유공간에서 샘플링된 포인트의 집합이다. 식 (2)에서 $B[p, y]$는 예측된 occupancy probability $p$와 self-supervised binary label $y$간의 BCE이다.
	- $$B[p,y]=-ylog(p)-(1-y)  log⁡(1-p)   \quad\quad\quad(3)$$
	- 게다가 Chamfer distance는 DeepMapping[1]에서 network를 더빠르게 수렴하게 도와주는 또다른 Loss이다. 이 Loss는 global point cloud X와 Y 사이의 distanc를 다음의 식으로 측정한다.
	- $$ d(X, Y)=\frac{1}{|X|} \sum\limits_{x\in X}{ \underset{y\in Y}{min} \left||  x-y\right||_{2} } + \frac{1}{|Y|}\sum\limits{\underset{x\in X}{min} ||x-y||_{2}} $$
	- DeepMapping과의 차이점은 DeepMapping은 각 raw point cloud를 ICP와 같은 기존의 등록 방법을 통해 sub-optimal global pose으로 변환하는 warm start 메커니즘을 도입한다. 이 optinal step은 소규모 데이터셋에서 DeepMapping의 수렴을 가속화 할 것이다. 하지만 대규모 데이터셋에서는 합리적인 초기화 방법이 필요하기 때문에 스크래치로부터 시작하여 수렴하는것은 더 어려울 수 있다.
	- DeepMapping의 한계점은 이러한 작은 데이터셋에서의 성공에도 불구하고 DeepMapping은 대규모 데이터셋으로 규모를 키우는 것은 앞서 언급한 도전과제들로 인해 불가능하다: (1) no-explicit-loop-closure (2)no-local-registration (3)slow-convergence-in-global-registration
## Pipeline

### Batch organization
- DM2의 pipeline은 Fig2에서 보이는 노랑과 파랑 블록으로 보이는 2개의 주요 단계로 구성된다.
- 첫번째 단계는 장소 인식을 기반으로한 map topology를 사용하여 training batch를 구성하는 것이다.
- 입력은 처음에는 초기 상태와 참값 사이의 중간 상태로 등록된 point cloud 시퀸스이다.
- 각 앵커 프레임 A는 off-the-shelf place recognition 알고리즘에서 얻은 맵 토폴로지를 사용하여 이웃 프레임 N과 함께 하나의 배치로 구성된다.

### Pairwise registration

- 두번째 단계에서 최적화 하기전 각 배치내에서 이웃에서 앵커로의 변환을 찾음으로써 앵커 프레임과 이웃 프레임간의 pairwise registratio이 계산된다.
- 이는 아무 off-the-shelf pairwise registration 알고리즘으로 계산이 가능하다.

### Training as optimization

- 두번째 단계의 파이프라인은 학습기반 최적화이다. Sec.3.1에서 소개된 중요한 구성요소들 외에도 consistency loss는 DeepMapping2의 또다른 중요한 구성요소이다.
- Loss Fuc은 다음의 아이디어로 디자인되었다. 앵커 프레임 A의 각 포인트에 대해 consistency를 계산할 수 있는데 이는 서로 다른 버전(A' 와 A'')의 글로벌 좌표간의 L2 distance로 정의된다.
- 이러한 버전은 각 이웃 프레임 N의 글로벌 포즈와 앵커와 이웃 간의 상대 변환에 의해 계산된다.
## Batch organization
- DeepMapping에서는 loop closure의 부족대문에 대규모 환경에 이 방법을 적용하는것은 매우 어려움
- loop closur는 SLAM 시스템에 번들조정을 통해 쉽게 통합될 수 있지만 딥러닝 기반 방법에서는 non-differentiability 때문에 어렵다.
### Effect of different batch organization
![[Pasted image 20240215102644.png]]
- 전체 궤적을 여러 미니 배치로 나누는것은 불가피하다. 그러면 이러한 분할의 올바른 방법은 무엇일까요? Fig.3에 나와 있는 것처럼 데이터셋을 나누기 위해 여러 배치 배열을 테스트 했다.
- fig 3.(a)는 무작위 배치 조직은 매우 안좋은 매핑 결과를 갖는다. fig 3.(b)는 시간순으로  배치를 조직했다. DM1처럼 지역적으로는 잘등록 되지만 전역 매핑은 별로 좋지 않다. fig 3.(c)는 공간적으로 인접한 이웃들을 배치로 사용하는 것이 매우 좋은 결과를 가진다는 것을 알려준다. 이유는 M-Net이 배치내의 프레임들을 함께 끌어 모아서 등록하기 때문이다.
- 루프 클로저는 모든 공간적으로 인접한 프레임이 루프 내에서 올바르게 등록되면 루프가 닫혔기 때문에 훈련 과정에서 통합된다.

### Loop-closure-based batch organization

- off-the-shelf place recognition algorithm에서 얻은 map topology를 사용하여 공간적으로 인접한 이웃들을 포함한 배치를 구성한다.
- 맵 토폴로지는 그래프로 연결되어 있고 각 노드는 프레임을 나타내고 각 엣지는 두 노드가 공간적으로 인접함을 나타낸다.
- 주어진 앵커 프레임 A에서 이러한 배치들을 구성하기 위해 맵 토폴로지에서 앵커 프레임과 연결된 top k의 가장 가까운 프레임들을 찾고 그것들을 하나의 배치로 구성한다. 이렇게 하면 M-net이 이러한 공간적으로 인접한 프레임들을 등록함으로써 훈련 과정에서 루프 클로저 정보를 포함할 수 있게 된다. 이것은 또한 local-to-global point consistency loss에서  pairwise 제약 조건을 만드는 것을 가능하게 한다
## Local-to-global point consistency loss
### Slow convergence

- 새로운 배치 조직 방법을 사용하더라도 DeepMapping은 대규모 데이터셋에서 여전히 느리게 수렴한다. 초기화가 네트워크에 warm start를 제공할 수 있지만, 이 정보는 초기화에서 멀어질수록 노화되며 글로벌 포즈 추정에서 충분한 제약 조건을 주지 않기 때문에 수렴이 느려 질 수 있다.
- 따라서 초기화에서의 정보를 훈련 과정에 통합하여 네트워크에 대한 전역 추론 신호를 제공한다.
- 이는 두 인접한 point cloud pairwise 관계를 제약하여 글로벌 프레임에서 그들의 거리를 최소화 함으로써 수행될 수 있다.

### Point correspondence

- 비록 점 수준의 대응이 필요하여 점들간의 거리를 계산하지만 이러한 점대응은 매핑 알고리즘이 자주 적용되는 대규모 및 야외 시나리오에서는 매우 드물다. Fig. 4에서 볼 수 있듯이 실제 데이터셋에서 대부분의 포인트는 가장 인접한 포인트로부터 약 0.25m 떨어져 있다. 이때 최근접 포인트를 찾거나 수작업으로 descriptor또는 학습 기반 특징을 추상적으로 찾음으로써 alternative distance를 계산할 수 있지만, 첫번째 장소에 point to point 대응이 존재하지 않을 수 있기 때문에 부정확한 계산이 된다.
- 두 point cloud가 다른 센서 자세에서 스캔될때 동일한 포인트(즉, 동일한 장소에 대한 포인트)가 두 point cloud에 동시에 존재할 가능성은 매우 희박하다. 이로 인해 위의 방법들로 찾은 대응들은 자연스럽게 부정확해질 수 밖에 없다. 따라서 질문을 제기한다 : point to point 대응에 의존하지 않고 전역으로 등록된 point cloud의 pairwise 관계를 어떻게 제약 할 수 있을까요?

### Distance metric

- point to point 대응에 의존하지 않고 단일 point cloud내의 points를 고려함으로써 이 문제에 접근한다.
- point cloud S가 서로 다른 변환 행렬 T에 의해 변환될 때, point to point 대응은 보존된다. 왜냐하면 같은 local point cloud로 부터 나왔기 때문이다. 변환 후 두 point cloud 간의 거리는 각 대응 point간의 L2 distance를 평균 내어 쉽게 계산된다. 이 metric은 두 변환 T, T'와 단일 point cloud S의 함수로 다음과 같이 정의된다.
- $$ d(T,T',S) = \frac{1}{|S|}\sum\limits_{s \in S}{ ||T_{S} - T'_{S} ||_{2} } \quad\quad\quad (5) $$
- 식 (5)는 변환 후 두 point cloud간의 거리 뿐만 아니라 두 변환 간의 차이(불일치)도 측정한다. 이는 이 문제에서 매우 바람직한데 상대적으로 정확한 pairwise 변환은 각 이웃 프레임에서 앵커 프레임까지 사용가능 하기 때문이다. 우리는 L-Net의 추정을 제약하여 글로벌 센서 포즈에서 pairwise 관계가 보존되도록 한다.
- ![[Pasted image 20240215103333.png | 400 ]]
- fig (5)는 이 metric의 아이디어를 보여준다. 2가지 버전 앵커 프레임이 보여진다. A'는 L-Net에 의해 추정된 글로벌 포즈에 의해 변환된다. A"는 각 이웃 프레임의 글로벌 포즈와 상대 변환에 의해 변환된다.
- fig (5a)에서 point cloud가 잘 등록되지 않았기 때문에 검은색 화살표로 표시된 거리가 크다. 그러나 fig (5b)에서는 모든 대응 포인트가 겹치기 때문에 거리는 0이다.

### Consistency Loss

- •local과 global 등록 간의 불일치(inconsistency)를 측정하기 위해 local-to-global consistency loss를 설계했다.
- $S_i$ 와 $S_j$  사이의 pairwise transformation을 $T_i^j$ 로 나타내고 $S_i$ 의 global pose를 $T_i^G$ 로 나타낸다. 이웃 $N_i$ 는 $S_i$ 의 이웃 프레임의 인덱스로 정의된다. consistency loss를 다음과 같이 정의한다
- $$ \mathcal{L}_{C} = \frac{1}{K |\mathcal{N}_{i}|} \sum\limits^{K}_{i=1}{ \sum\limits_{j \in \mathcal{N}_{i}} d(T^{G}_{j}T^{i}_{j},T^{G}_{i},S_{i}) } \quad\quad\quad (6) $$
- 식 (6)의 loss는 점 대점 대응을 피하면서 Global inference cue를 제공한다. 각 $T_i^j$는 pairwise하게 계산되고 global estimatio에서 pairwise relations보다 더 정확하게 고려된다.
- $\mathcal{L}_C$를 최소화 함으로써 네트워크는 초기화로부터 더 많은 정보를 얻고 더 빠르게 수렴한다.


## Experiment

### dataset
data
|-- KITTI # Type of dataset
|   |-- 0018 # Trajectory number
|   |   |-- pcd # Point cloud data / total : 2762개
|   |   |   |-- 000000.pcd
|   |   |   |-- 000001.pcd
|   |   |   |-- ...
|   |   |   |-- gt_pose.npy # Ground truth pose
|   |   |-- prior # Prior data
|   |   |   |-- group_matrix.npy # Group matrix
|   |   |   |-- init_pose.npy # Initial pose / (2762, 6)
|   |   |   |-- pairwise_pose.npy # Pairwise registration / (2762, 7, 6)