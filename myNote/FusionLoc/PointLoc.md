# 논문
## 1. Introduction
- 전형적인 Lidar 센서 localization 시스템은 보통 feature extraction module, feature matching algorithm, outlier rejection step, matching cost function, spartial searching, temporal optimization, filtering 매커니즘을 포함함
- global pose를 추정하기 위해 맵 기반 접근 방법들이 제안됨
	- Barsan & wang [4] Lidar intentsity로부터 descriptor를 학습하고 사전에 구축된 intensity에 대해 descriptor를 매칭하여 relocalization을 하는 방법을 제안
		- 지도를 구축하는 것이 어렵고 지도가 커질수록 컴퓨팅 복잡도가 크게 증가
		- 다른 시스템에서 정확한 초기 포즈를 제공해야됨

- 대부분의 localization 방법은 global navigation satellite system(GNSS)를 사용함
	- 고층 건물이 많은 대도시나 실내 환경에서는 사용할 수 없음
- 위와 같은 한계에 대비해 Uy & Lee [8]는 포인트 클라우드 검색 기반 localization 방법을 제안
	- 참조 DB 형태의 사전 구축된 지도에 대한 6 DOF pose를 얻음
- Dube at al[9]는 lidar 센서의 point cloud에 개별적인 데이터 기반 descriptor를 저장해 참조 데이터베이스 저장 효율을 향상시키기위한 Segmap을 제안
- 위와 같은 검색기반 접근 방법은 다음과 같은 문제점을 가지고 있음
	- 쿼리 point cloud와 참조 point cloud 사이의 가장 가까운 매칭를 찾는 시간 복잡도가 O(n)이다. -> 일반적인 실시간 프로그램에서 적합하지 않음
	- O(n)의 저장공간을 차지하며 많은 모바일 로봇에 배포하기에 한계가 있다.

- 최근 딥러닝 학습 기반 접근 방법은 end-to-end-localization 시스템을 만드는데 많은 장점을 가진다. (검색 기반 방법들의 단점을 보완함)
	- runtime동안 참조 데이터베이스를 요구하지 않고 학습된 feature들은 일반적이고 robust한 경향을 가짐 (저장공간 필요하지 않음)
	- 신경망을 훈련시키고 추론하여 직접 포즈를 예측하는데 O(1)의 시간 복잡도를 가짐 (빠른 시간내에 수행하기 때문에 실시간에 적합함)
- 현재 딥러닝 기반 방법들은 RGB 이미지를 통해 global pose를 추정하는데 다음과 같은 단점을 가짐
	- 시각 센서는 환경변화의 민감하고 좁은 시야 Fov를 가짐
	- **그래서 Lidar 센서를 사용한 딥러닝 기반 방법 제안**


## 2. Related Work

### A. Visual Sensor Relocalization
- map registration 방법의 단점들을 보완하기 위해 global pose를 직접 추정하는 딥러닝 기반 방법[10]-[16]을 제안
	- 연속적인 RGB이미지를 통해 PoseNet을 학습한 방법[10], [11], [18]
	- Brahmbhatt[13]은 연속적인 두 이미지 사이의 상대적 포즈를 기하학적 공식을 통해 pose를 추정
### B. Learning-Based Localization Systems
- Almalioghu[20]은 robut MMwave radar-based egomotion estimation
- CelilinDeep[21]은 celluar 신호와 로봇의 위치 사이의 비선형 관계를 포착하기 위해 DNN을 학습
- Alshamaa[22] 센서 위치 결정을 위한 분산 커널 알고리즘 제안 이하 등등

### C. DNN-Based Lidar Odometry
- 연속적인 Lidar Scan사이에서 상대적 위치를 계산하여 Lidar odometry를 추정하는 학습 기반 방법들을 제안
- Wang[27]은 상대 포즈를 직접적으로 추론하기 위한 병렬 DNN을 제안
- Li[28]은 odom추정을 위한 2d lidar와 IMU 센서를 활용한 학습 기반 프레임워크 제안
- Horn[29] point cloud 융합 문제를 해결하기 위해 flow embedding 접근법을 통해 Lidar odometry
- 3dFeat-Net[30] 약한 supervision을 사용한 point cloud matching을 위한 3D feature detector와 descriptor을 학습하기 위해 개발됨
- Lu[31] 두 point cloud를 정확하게 정합하기 위한 가상 대응점 방법을 제안
- Wang[32] sub-network구조를 설계하여 ICP 방법의 어려움을 해결한 방법을 제안
- 위 point cloud registration 접근법들은 Lidar odom을 예측하는데 사용될 수 있지만 이러한 방법은 이전 point cloud로부터 상대적 pose를 추정하기 때문에 초기 포즈가 없는 상태에서 relocalization하는 방법에는 적합하지 않음
![[Pasted image 20240314134912.png | fig 2]]
- fig 2에서 odometry와 relocalization 방법의 차이를 보여줌
	- Lidar odometry는 운전중에 연속적인 point cloud사이에서 상대적인 pose를 추정
	- Lidar relocalization은 어떤 point cloud가 주어지면 해당 point cloud의 global pose를 추정
### D. Deep Learning on Point Clouds
- point cloud에서 DNN 기반 feature 추출 방법[33]-[36]이 좋은 성능을 가지는데 그중 VoxelNet[37]이 객체 탐지를 위해 voxel에서 feature embedding을 학습하기 위해 개발됨
- PointNet++[33], [34], [36]은 픽셀의 위치라는 순서적인 특징인 이미지와 다르게 무순서한(unordered) point cloud에서 feature를 학습하는 방법을 제안함
	- 3d object detection part segmentation, semantic segmentation과 같은 방법에서 효과적인 성능을 보임

## 3. Problem Statement
- Lidar Relocalization을 수행하기 위해 Lidar 센서로 부터 얻은 point cloud data를 사용하여 global pose를 추정하는 DNN 기반 framework를 설계함
	- mobile agent가 이전에 방문한 영역 내에서 6 DOF pose를 추론, 사용 예시는 mobile agent가 이미 쿼리 장소에 한번 방문한 적이 있고 이 쿼리 장소에서 자신의 위치를 찾아야 할 때 사용함

- 각 timestamp t에서 agent는 Lidar 센서로 부터 point cloud frame $P_{t}=\{x_{i}|i=1,...,N\}$ 을 받고, 여기서 $x_{i}$는 현실 세계좌표 $(x,\;y,\;z)$인 vector이다.
	- $P_{t} \;shape\,:(N, \;3)$
- agent의 relocalization의 결과 값은 6-DoF poas $[t,\,r]^{T}$ 으로 파라미터화된다.
	- 여기서 $t \in R^{3}$ 인 3d translation vector
	- $r \in R^{4}$인 4d rotation vector (quaternion)
- Deep 3D pose regressor는 $\mathcal{F}(P_{t})=(t,\, r)^{T}$ 을 학습하는 DNN 기반 신경망이다.

## 4. Deep Point Cloud Relocalization
이번 섹션에서는 우리가 제안하는 Lidar 센서로부터 global pose를 추론하는 deep 3d pose regressor 방법인 PointLoc을 설명한다.
- PointLoc의 전체적인 구조는 다음 Fig 3.에서 나타냄
- ![[Pasted image 20240314144424.png | Fig 3. PointLoc의 전체적인 구조]]
- 전체적인 구조는 point cloud pre-processing, point cloud encoder, self-attention module, Group All Layers module, pose regressor으로 구성되어 있다.
### A. Point Cloud Pre-Processing
- raw point cloud를 neural network에 맞는 input으로 바꾸기 위한 전처리 모듈
- 각 Lidar sensor scan의 point cloud frame은 서로 다른 수의 point를 포함
	- 하지만 neural network는 $(N,\,3)$의 똑같은 point cloud 차원을 가져야함
	- 이 문제를 해결하기 위해 random point cloud sampling 전략을 사용
	- 이 연구에서는 N을 20,480개로 설정
		- 이 실험에서 Radar RobotCar Dataset [6], [7]의 point cloud는 약 21,000개를 포함하고 있고 가능한 정보 손실을 막기 위해 최대 한도로 설정

### B. Point Cloud Encoder
- 이 모듈의 목표는 point cloud로 부터 feature를 추출
	- point cloud encoder로 부터 추출된 feature representation은 정확성과 안정적인 relocalization을 달성하기 위한 중요한 역할
- 직관적으로 인간은 주변 key point와 feature를 통해 자신의 위치를 식별 가능하고 기존의 기하적인 방법들로도 point cloud data를 활용하여 정밀한 localization을 수행할 수 있음
	- 여기서 영감받아, DNN이 localization과 관련된 original point cloud data에서 key point의 부분 집합을 학습하면, 이런 key point을 활용하여 위치를 더 잘 식별할 수 있을 것이다.
- 기존 연구 [34], [40]은 critical-subset 이론, 즉 어떤 point cloud $P$에 대해서도 PointNet과 같은 구조가 두드러진 point 부분 집합 $C$가 $P$에 포함됨을 식별할 수 있음을 증명했기 때문에 relocalization에서 유용하게 사용될 수 있음

- 구체적으로, PointNet은 MLP, feature transformation module, max pooling layer를 사용하여 point cloud 분류와 segmentation을 위한 permutation invariant function을 근사화하는데 활용한다
	- 사실 이것은 universal continuous set function approximator로서 다음과 같이 표현된다.
	- $f(x_{1},...,x_{N})=\phi(MAX(h(x_{i}) | x_{i} \in P)) \;\; (1)$
		- 여기서 $\phi$와 $h$는 2개의 연속 함수이며(일반적으로 MLP로 구현) $MAX$는 max pooling layer을 나타냄
	- PointNet++는 PointNet에서 확장되어 metric space에서 point sets의 계층적 특징을 재귀적으로 포착함
	- 식 (1)에서 PointNet 구조의 결과는 $u=MAX\{h(x_{i}) | x_{i} \in P\}$ 로 결정됨
		- 따라서, $u_{j}=h_{j}(x_{i})$인  $x_{i} \in P$가 존재함
		- 여기서 $u_{j}$는 $j$번째 차원이고 $h_{j}(x_{i})$는 $h(x_{i})$의 $j$번째 차원이다. 이러한 점들은 $C \subseteq P$ 인 중요 부분집합 $C$로 집계될 수 있고 $C$는 $u$를 결정하고 그다음 $\phi(u)$를 계산한다. 
			- 결과적으로 중요 부분집합 이론은 $\phi(MAX\{ h(x_{i}) | x_{i} \in P \})$ 구조의 신경망에 적용될 수 있음
		- PointNet++을 사용

- point cloud encoder는 PointNet++[33],[36]의 set abstraction(SA) Layer를 기반으로 설계
	- 4개의 연속된 SA Layer로 구성
	- 각 SA Layer는 sampling layer, grouping layer, PointNet layer[34]로 구성
	- SA layer는 feature matrix $F \in R^{N \times C}$를 입력으로 받음
		- 여기서 $N$은 point의 수, $C$는 각 point의 feature 차원
	- SA layer의 출력은 마찬가지로 feature matrix $F \in R^{N^{'} \times C^{'}}$ 
		- 여기서 $N^{'}$은 하위 샘플링된 point의 수, $C^{'}$은 각 point의 새로운 feature 차원
- robust feaeture 학습을 위해 SA layer내부에서 multi scale grouping strategy[33]을 활용
	- 구체적으로, 이 layer는 Furthest Point Sampling을 통해 point끼리 가장 먼 point들을 sampling하여 대표성을 가지는 point를 sampling 한다.
		- 예를들어 밀집된 point들은 서로 거리가 가까이 있으므로 1개만 대표적으로 sampling 됨
		- ![[Pasted image 20240314185825.png]]
	- 이렇게 먼거리의 point들을 sampling 하고 나서 grouping을 진행하게 되는데 이때 그룹핑을 다양한 크기로 진행한다.
	- ![[Pasted image 20240314185946.png | 250]]
	- $\mathbf{F}^{'}_{j} = \mathbf{MAX}_{\{i \; | \; ||x_{i} - x_{j} || \le r\}}\{h(\mathbf{F}_{i}, x_{i} - x_{j})\} \; \; (2)$
	- 여기서 $\mathbf{F}_{i}$는 $\mathbf{F}$의 i번째 row이고, $\mathbf{F}^{'}_{j}$는 $\mathbf{F}^{'}$의 j번째 row이며 $h: R^{C} -> R^{C^{'}}$는 MLP이고 $MAX$는 maxpooling layer이다.
### C. Self-Attention Module
- 이 모듈의 목적은 움직이는 object와 같은 outlier를 이전 layer에서 추출된 feature에서 제거하여 relocalization 성능을 향상
	- 이전 연구 [16], [17]은 self-attention 메커니즘으로 noisy feature를 제거함으로써 visual sensor relocalization에서 성능을 향상할 수 있음을 입증
	- 최종 위치를 추정하기 전에 동적 특징들을 제거하는 모듈
- point feature와 **mask**사이에서 element-wise dot product를 통해 original point feature로 부터 움직이는 object의 ouliter feature를 제거하기 위한 **mask**를 학습

- point cloud encoder에서 학습된 결과물 point feature $\mathbf{F} \in R^{R \times C}$에 대해 mask $\mathbf{M} \in R^{1 \times C}$를 학습하는 것
	- 특성 $\mathbf{F}$입력으로 받아 mask $\mathbf{M}$을 직접 생성하는 MLP와 sigmoid함수를 사용한뒤 broadcast하고 masking하여 가중치가 부여된 특성 $\hat{\mathbf{F}}$ 을 얻음
	- ![[Pasted image 20240315102519.png | 300]]
	- 구제척으로 $\mathbf{F}$의 $N \times C$인 feature가 MLP를 통해 mask $\mathbf{M}$의 $1 \times C$를 생성하고 sigmoid와 broadcasting을 통해 $N \times C$으로 만들고 $\hat{\mathbf{F}}=\mathbf{F} \cdot \mathbf{M}$의 가중치 element wise 곱셈을 통해 가중치가 부여된 feature $\hat{\mathbf{F}}$를 얻음

### D. Group All Layers Module
- 이 모듈의 목표는 이전 모든 Layer로 부터 feature 집계하여 embedded feature vector를 생성하는 것
- ![[Pasted image 20240315103638.png | 200]]
- $N_{4} \times C_{4}$의 feature set의 입력을 받고 MLP를 통해 $N_{4} \times C_{5}$ (여기서 $C_{4} < C_{5}$)으로 전파되고 maxpooling layer를 통해 $C_{5}$차원을 같은 feature vector로 downsampling된다. 

### E. Pose Regressor
- 이 모듈의 목적은 최종 pose를 예측하는 것
- 이전 maxpooling layer에서 나온 $C_{5}$차원의 feature vector를 입력으로 받아 최종적으로 translation $t$와 rotation $r$을 추론한다.
- ![[Pasted image 20240315104338.png | 200]]
- pose regressor은 2개의 분기로 나눠진 FC Layer로 구성되고 각 FC Layer는 4개의 층으로 구성됨
	- 마지막 FC Layer를 제외하고 FC Layer층 마다 Leaky Relu activation function을 사용

### F. Loss Function
- 최종 목표는 6-DOF pose $[t, r]^{T}$를 추정, 
- 이전 연구[10]-[12], [42]은 직접 quaternions을 예측하고 $l_{1}$ 과 $l_{2}$ loss를 사용
	- 이는 over-parameterized되어 output quaternion의 normalization가 더 나쁜 정확도가 생김
	- 오일러 각도로 회귀해도 $2\pi$로 커버하기 때문에 적합하지 않음
- 신경망 훈련을 위해 [13]에서 채택한 [11]의 loss function 정의를 사용
	- $K$개의 training sample $\mathcal{G}=\{P_{t} \, | \, t = 1, ..., K\}$와 그에 대응 되는 ground-truth pose $\{ [\hat{t}, \hat{r}]^{T}_{t} \; t=1,...,K\}$ 가 주어 졌을 때 PointLoc의 parameter는 다음의 Loss function으로 학습됨
	- $\mathcal{L}(\mathcal{G})= ||t-\hat{t}||_{1}e^{-\beta}+ \beta+ ||\log{q} -\log{\hat{q}}||_{1}e^{-\gamma} + \gamma \;\;\;\; (4)$  
		- 여기서 $\beta$와 $\gamma$ 는 translation과 rotation을 같이 학습 하기 위한 balanced factor이고 $\beta_{0}$와 $\gamma_{0}$으로 초기화 됨
		- $\log{q}$는 단위 quaternion $q=(u, v)$의 log 형태로 $u$는 스칼라 $v$는 3d vector 이는 다음과 같이 정의됨
		- $$\log{q}=\begin{cases}  \frac{v}{||v||}cos^{-1}u, \;\; if \;\; ||v|| \neq 0 \\ 0, \quad\quad\quad\quad\;\, otherwise \end{cases}$$
## 5. Indoor Lidar Sensor Dataset for Relocalization

## 6. Experiment
- 추론의 정확도를 높이기 위해 다음 논문들의 data augmentation을 참고
	- PoseNet[10], Geometric loss function for camera pose regression with deep learning[11], Geometry-aware learning of maps for camera localization[13], Atloc[17]

# 실제 구현
- Atloc 에 pointLoc에서 사용한 Loss함수가 있음, 중간 부분의 abs_loss만 사용하면 됨

```Python
class AtLocPlusCriterion(nn.Module):  
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):  
        super(AtLocPlusCriterion, self).__init__()  
        self.t_loss_fn = t_loss_fn  
        self.q_loss_fn = q_loss_fn  
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)  
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)  
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)  
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)  
  
    def forward(self, pred, targ):  
        # absolute pose loss  
        s = pred.size()  
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3]) + self.sax + \  
                   torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:], targ.view(-1, *s[2:])[:, 3:]) + self.saq  
  
        # get the VOs  
        pred_vos = calc_vos_simple(pred)  
        targ_vos = calc_vos_simple(targ)  
  
        # VO loss  
        s = pred_vos.size()  
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]) + self.srx + \  
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]) + self.srq  
  
        # total loss  
        loss = abs_loss + vo_loss  
        return loss
```

- \*.pose.txt 파일 취급
	- $4 \times 4 \; matrix$ , homogenous coordinate
	- $$ \begin{bmatrix}   m_{1} & m_{2} & m_{3} & m_{4} \\ m_{5} & m_{6} & m_{7} & m_{8} \\ m_{9} & m_{10} & m_{11} & m_{12} \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
	- 먼저 translation vector 추출 $[m_{4} \quad m_{8} \quad m_{12}]$ 
	- Rotation matrix 추출 후 quanternion vector로 변환
		- $$ \begin{bmatrix} m_{1} & m_{2} & m_{3} \\ m_{5} & m_{6} & m_{7} \\m_{9} & m_{10} & m_{11} \end{bmatrix} \rightarrow q = [q_{1} \quad q_{2} \quad q_{3} \quad q_{4}]$$
		- 활동 범위를 반구(hemisphere)로 제한
			- $np.sign(q_{1}) * [q_{1} \quad q_{2} \quad q_{3} \quad q_{4}]$ 
		- logarithmic form으로 변경
		- $$q'= \frac{[q_{2} \quad q_{3} \quad q_{4}]}{||[q_{2} \quad q_{3} \quad q_{4}]||_{2}} * cos^{-1}q_{1} =[q'_{2} \quad q'_{3} \quad q'_{4}]$$
		- 최종 6-DOF pose는 다음과 같음
			- $target\_pose = [m_{4} \quad m_{8} \quad m_{12} \quad q'_{2} \quad q'_{3} \quad q'_{4}]$ 

# Training
## 2024_0326_1646_54
- Adam lr = 0.001 
- 사용한 transforms
- ![[Pasted image 20240326164916.png]]

- training loss와 valid loss를 보면 잘 줄어 드는 것처럼 보이지만 실제로 rotation_error와 translation_error를 보면 이상하게 진동하고 있음.
![[plot.png]]

- 확인해야 할 사항
	- 들어가는 training data를 normalize 해줘야 하는지?
	- 나오는 결과물에 대해서 rotation pose가 제대로 변환된건지?
		- quaternion 변환시 오류 있는지 확인

## 2024_0327_1652_46
- 모든 transforms 빼고 학습
- ![[Pasted image 20240327170714.png]]

## 2024_0327_1710_35
- PointLocLoss(-3, -3)넣고 시작 
- 모든 transfomrs 뺌
- ![[Pasted image 20240327171223.png]]

## 2024_0327_1814_02
- Group all layer의 mlp 구조에서 batchnorm1d 와 dropout을 뺌
- point loc loss (-3, -3) 넣고 시작
- 모든 transforms 뺌

- 결론
	- 이전에 valid와 train과 벌어지면서 학습이 안되는 현상은 해결이 됨..

## 2024_0327_1902_35
- Group all layer의 mlp 구조에서 batchnorm1d 와 dropout을 뺌
- point loc loss(b=0, g=-3) 넣고 시작
- 모든 transforms 뺌 ![[plot 1.png]]
- 결과 
	- Epoch 796에 rotation error 9도 / trans error 0.21m

## 2024_0328_1131_13
- Group all layer의 mlp 구조에서 batchnorm1d 와 dropout을 뺌
- point loc loss(b=0, g=-3) 넣고 시작
- transform 추가
	- Random jitter
	- Random Rotation
	- Random Translation
![[plot 5.png]]
## 2024_0328_1316_57
- Group all layer의 mlp에서 batchnorm1d 추가
- Point loc loss(b=0, g=-3)
- transforms
	- Random Jitter
- PoseRegressor에서 Dropout 뺌
![[plot 4.png]]
## 2024_0328_1846_10
- Group all layer의 mlp에서 batchnorm1d 추가
- Point loc loss(b=0, g=-3)
- transforms
	- Random Jitter
- PoseRegressor에서 Dropout 뺌
- PoseRegressor의 translation_mlp, rotation_mlp의 LeakyRelu의 negative_slope를 0.02로 조정함
- GroupAll Layers Module의 ReLU를 Leaky ReLu로 수정함
![[plot 3.png]]
## 2024_0329_1326_30
- Group all layer의 mlp에서 batchnorm1d 추가
- Point loc loss(b=0, g=-3)
- transforms
	- Random Jitter
- PoseRegressor에서 Dropout 뺌
- PoseRegressor의 translation_mlp, rotation_mlp의 LeakyRelu의 negative_slope를 0.02로 조정함
- GroupAll Layers Module의 ReLU를 Leaky ReLu로 수정함
- scheduler steplr 사용

## 2024_0329_1854_06
- Group all layer의 mlp에서 batchnorm1d 추가
- Point loc loss(b=0, g=-3)
- transforms
	- Random Jitter
- PoseRegressor에서 Dropout 뺌
- PoseRegressor의 translation_mlp, rotation_mlp의 LeakyRelu의 negative_slope를 0.02에서 0.4로 조정함
- GroupAll Layers Module의 ReLU를 Leaky ReLu로 수정함
- scheduler steplr 사용
## 2024_0401_1236_55
- Group all layer의 mlp에서 batchnorm1d 추가
- Point loc loss(b=0, g=-3)
- transforms
	- Random Jitter
- PoseRegressor에서 Dropout 뺌
- PoseRegressor의 translation_mlp, rotation_mlp의 LeakyRelu의 negative_slope를 0.4에서 0.1로 조정함
- GroupAll Layers Module의 ReLU를 Leaky ReLu로 수정함
- scheduler steplr 사용