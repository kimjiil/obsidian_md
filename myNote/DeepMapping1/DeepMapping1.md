# DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds 논문 요약
---

## Abstract
- DNNs를 보조 함수로 사용하여 여러 포인트 클라우드를 전체적으로 일관된 frame에 맞추기 위한 DeepMapping 프레임워크를 제안
- 전통적으로 수작업을 해왔던 data association, sensor pose 초기화, global refinement와 같은 non-convex mapping process들을 DNNs을 사용하여 모델링한다
- 프레임워크는 2개의 DNNs으로 구성된다
	- input point cloud의 pose를 추정하는 localization network
	- 글로벌 좌표의 occupancy status를 추정하여 scen structure를 모델링하는 Map Network
- 2개의 네트워크를 통해 registration problem을 binary occupancy classification으로 변환하여 gradient기반 최적화를 사용하여 효율적으로 문제를 해결
- DeepMapping은 연속적인 point cloud간의 geometric constraints을 걸어 Lidar SLAM문제를 해결하는데 도움이 된다.
---

## 1. Introduction

- 딥러닝의 기술의 발전에도 불구하고 특히 등록 및 매핑과 같은 computur vision의 기하학적 측면의 개선은 완전히 입증되지 않음
- deep semantic representation이 기하학적 특성을 정확하게 추정하고 모델링하는데 한계가 있음


- deep learning과 geometric vision problem([45], [52], [49], [20], [19], [29], [18])을 통합하기 위해 다양한 연구를 시도중임
- ([26], [9], [20]) 방법들은 주변 환경의 맵을 representation으로 가지고 있는 DNN을 학습하여 camera pose를 추정하려고 한다.
- ([45], [52])의 방법들은 depth와 움직임 간의 내재적인 관계를 활용하는 비지도 학습 접근법을 제안한다.

- 이 논문의 핵심은 DNN이 기하학적 문제에 대해서, 특히 registration과 mapping에서 얼마나 잘 일반화될 것인가
	- Semantic task은 DNN에서 크게 이득을 보고 있는데 해당 문제들은 대부분 경험적으로 정의되어 많은 데이터를 통해 통계적으로 모델링되어 해결된다.
	- 하지만 많은 기하학적 문제들은 이론적으로 정의됨 -> 통계적 모델링을 통한 해결책은 정확도 측면에서 한계가 있음
![[Pasted image 20240215111108.png | 400]]
- 일반적으로 손으로 엔지니어링된 mapping/registration process를 DNN으로 변환하고 학습하여 해결한다. 그러나 훈련된 DNN이 다른 장면으로 일반화되기를 반드시 기대하는 것은 아닙니다.
- 이것을 의미 있게 만들기 위해 ([26], [9], [20])의 지도 학습과 달리, registration quality를 반영하는 적절한 비지도 손실 함수를 정의해야 한다. 이러한 아이디어로 Figure 1에서 좋은 결과를 보여줌
	- unsupervised 방식으로 학습하는 2개의 end-to-end DNN 인 DeepMapping을 제안
	- continuous regression 문제를 DNN과 unsupervised Loss를 사용하여 registration 정확도를 희생하지 않고 이진 분류로 변환함

---

## 2. Related Work
 - pairwise point cloud registration 방법은 크게 local과 global 2가지로 분류된다.
### Pairwise local registration
- local 방법은 두 point cloud사이의 coarse initial alignment를 가정하고 registration을 정밀화하기 위해 transformation을 반복적으로 업데이트함
- 이러한 방법들은 Iterative Closest Point(ICP) 알고리즘 ([6], [10], [34])과 확률 기반 접근 방법 ([23], [31], [12])가 있다.
- local 방법은 수렴 범위가 제한되어 있어 warm start 또는 좋은 초기화가 필요함
### Pairwise global registration
- global 방법([46], [3], [30], [50], [28], [14])은 warm start에 의존하지 않고 임의의 초기 자세를 가진 point clouds에 수행할 수 있다.
- 대부분의 global 방법은 두 point cloud에서 feature descriptors를 추출한다. 이러한 descriptor는 relative pose estimation을 위한 3D-to-3D 대응을 만들기 위해 사용된다.
- 강건한 추정 방법, 예를 들어 RANSAC[17]은 일반적으로 mismatch를 다루기 위해 사용된다.
- feature descriptor들은 FPFH[35], SHOT[43], 3D-SIFT[38], NARF[40], PFH[36], spin images[25]와 같이 hand-crafted이거나 3DMatch[48], PPFNet[13], 3DFeatNet[24]와 같은 학습 기반 descriptor이다.

### Multiple registration
- 몇개의 방법([42], [16], [22], [44], [11])은 multiple point clouds registration을 제안한다. 어떤 한 방법은 모델에 등록된 이전 point clouds에 점진적으로 새로운 point cloud들을 추가한다.
- 점진적 registration 방법의 단점은 registration error도 똑같이 점진적으로 쌓인다.
- 이러한 누적된 에러, drift 모든 센서 pose의 그래프 상의 global cost function을 최소화함으로써 완화될 수 있다([11], [42]).

### DeepLearning Approaches
- 최근 연구들은 mapping과 localization을 통합하는 아이디어를 연구한다. 
- ([45], [52])는 depth와 로봇 움직임 사이의 내재적 연결 관계를 활용하는 비지도 학습 방법이다.
- 이 아이디어는 ([7], [49], [47], [29])에서 visual odometry와 SLAM 문제를 해결하기 위해 deep learning을 사용한다.
	- CodeSLAM[7]은 대응하는 depth 이미지에 대한 VAE를 사용하여 dense geometry를 표현하며 bundle adjustment중에 최적화된다.
- [18]은 메모리 시스템을 사용하여 해당 장면의 representation을 기억하고 부분적으로 관측된 환경에서 pose를 예측하는 generative temporal model이다.
	- 이 방법과 DeepMapping은 모두 관측된 데이터로 붙 센서 자세를 추론할 수 있지만, [18]은 메모리에 로딩하기 위한 지도 학습 단계가 필요하지만 deepmapping은 완전히 비지도 학습이다.
- ([20], [32])방법들은 RNN을 사용하여 미리 지도된 환경에서 이미지 시퀀스를 통해 환결을 모델링한다.
	- MapNet[20]은 RGB-D SLAM 문제를 해결하기 위한 RNN을 개발함, 상대적으로 작은 해상도를 가진 이산화된 공간에서 템플릿 매칭을 사용하여 카메라 센서의 위치를 추정함

- 다른 관련 방법([26], [9])는 DNNs이 학습된 이미지와 같은 환경에서 camera pose를 추정하도록 학습해서 camera localization 문제를 해결함
- DeepMapping의 training과정은 point cloud registration 문제를 해결하는 것과 동일하며 한번 training된 후에는 DNN이 다른 환경에서 일반화되지는 않는다.

---

## 3.Method
### 3.1. Overview
- $\mathcal{S}=\{ S_{i} \}^{K}_{i=1}$ 가 $D$차원 공간에서 Lidar Scanner로 얻어진 $K$개의 input point clouds이고 $i$ 번째 point cloud를 $S_{i}$이라고 하고 $N_{i} \times D$ 행렬 형태로 표현된다. $N_{i}$는 해당 로컬 프레임에 포함된 point수를 나타낸다.
- $K$개의 point cloud가 주어지면 모든 point cloud를 한개의 공통 좌표에 등록하는 것이 목표이고 각각의 point cloud $S_{i}$에 대한 sensor pose $\mathbf{T}={T_{i}}^{K}_{i=1}$ 추정한다. 여기서 $T_{i} \in SE(D)$

- 전통적인 방법([50], [16])은 이런 최적화 문제를 공식화하여 직접적으로 최적의 sensor pose $\mathbf{T}$를 찾음으로써 Loss 함수를 최소화 합니다.
- $$ \mathbf{T}^{*}(\mathbf{S}) = \underset{T}{argmin} \, \mathcal{L}(\mathbf{T}, \mathbf{S}) \quad\quad\quad (1)$$
- 여기서 $\mathcal{L}(\mathbf{T}, \mathbf{S})$ 는 registration quality를 평가하는 objective function이다.
- 직접적으로 $\mathbf{T}$를 최적화 하는 대신 input point cloud $\mathbf{S}$에 대한 sensor pose $\mathbf{T}$를 추정하기 위한 보조 함수 $f_{\theta}(\mathbf{S})$로 모델링된 DNN을 사용한다. 여기서 $\theta$는 최적화 가능한 보조 변수이다.
- 그런 다음 pose $S_{i}$는 global pose $G_{i}$로 매핑하는 변환 행렬에 의해 변환된다.

- registration problem을 다음 새로운 objective function을 최소화함으로써 optimal network parameter를 찾는 문제로 재공식화 한다.
- $$(\theta^{*}, \phi^{*}) = \underset{\theta, \phi}{argsmin} \, \mathcal{L}_{\phi} (f_\theta(\mathbf{S}), \mathbf{S}) \quad \quad \quad (2)$$
- 여기서 $\mathcal{L}_{\phi}$는 비지도로 학습가능한 objective function이다(Sec 3.2와 3.4에서 설명).


![[Pasted image 20240216135102.png]]
- Figure 2에서 DeepMapping의 pipeline을 나타낸다. 여기서 가장 핵심은 2개의 network이다.
	- point cloud $S_{i}$로 부터 sensor pose $T_{i}$를 추정하는 localization network (L-Net)
	- registration quality를 평가하는 occupancy map network (M-Net)
- L-Net 은 공식 (2)에 나타낸 $f_{\theta} \; : \; S_{i} \mapsto \; T_{i}$ 함수이다. 여기서 parameter $\theta$는 모든 point cloud에서 공유된다. 
- 글로벌 좌표인 $G_{i}$는 L-Net을 통해 추론된 sensor pose를 사용하여 얻고 변환된 point cloud로 부터 먼저 occupied와 unoccupied된 location을 샘플링한다.
- 이렇게 샘플된 위치를 M-Net에 입력하고 L-Net의 registration 성능을 평가한다.
- M-Net은 입력 위치가 차지된 확률을 예측하는 이진 분류 네트워크이다.
	- M-Net은 학습가능한 매개변수 $\phi$ 를 가지는 함수이다.
	- 이러한 occupancy probability들은 변환된 point clouds의 global occupancy consistency을 측정하는 비지도 Loss $\mathcal{L}_{\phi}$(식 (2)에 있음)를 계산하는데 사용하므로 registration quailty을 반영한다.

- 식(1) 에서 식(2)으로 변환하는 것은 문제의 차원 및 복잡성을 증가시킬수 있는데, 여기서 간단한 1차원 버전을 사용하고 보조함수로써 DNN을 사용한다.
	- DNN은 gradient-based 방법을 사용하여 고차원 공간에서 최적화하는 것이 원래의 문제를 직접 최적화하는 것보다 더 빠르고 더나은 수렴을 가능하게 한다.

---
### 3.2. DeepMapping Loss
- M-Net $m_{\phi}$를 사용하여 registration quality를 평가하는 비지도 손실 함수 $\mathcal{L}_{\phi}$를 정의한다.
-  M-Net은 연속적인 occupancy map $m_{\phi} : \mathbb{R}^{D} \to [0, 1]$이다.
	- global 좌표를 해당하는 occupancy 확률에 매핑한다.
	- $\phi$는 학습가능한 매개변수이다.
- 만약 global frame에서 좌표가 위치가 차지되있는지 아닌지를 나타내는 binary occupancy label y와 연관되어 있으면 예측된 occupancy 확률 $p$와 라벨 y간의 global 좌표의 loss를 BCE $B$를 통해 계산된다.
- $$B[p, y]=-y \, log(p)-(1-y)log(1-p) \quad\quad\quad (3)$$
- label y를 라벨링하는 방법은 이미 point cloud가 전역적으로 정렬된 상황을 고려한다. 이러한 경우에 라이다 스캐너의 물리적 원리로 인해 적연 프레임에서 스캔된 모든 점이 차지되었다고 표시되어야 하므로 Label 1로 표시된다.

 ![[Pasted_image_20240216145400.png | Figure 3. 샘플링 방법과 self-contradictory occupancy status를 나타낸다. 파란색과 주황색은 각 두 point cloud를 나타내며 X표는 샘플링된 미차지(unoccupied) 점을 나타낸다. (a)와 (b)는 각각 올바르게 정렬되고 잘못 정렬된 정렬된 point cloud를 보여준다. (b)의 빨간 화살표는 self-contradictory status를 가진 점을 표시한다.]] 
- 스캐너 중심과 스캐너로 관측된 모든 점 사이, 즉 시야 선상에 있는 점은 label 0으로 표시되야 된다. Figure 3은 라이다 스캐너를 위해 미차지된 점을 샘플링하는 메커니즘을 보여준다. 점선은 스캐너에서 방출된 레이저 빔을 나타낸다.
- $G_{i}$로 부터 샘플링된 point의 집합을 $s(G_{i})$으로 나타낸다.
	- $G_{i}$는 이러한 레이저 빔 위에 있는 점들 나타내고 X표시로 표현한다.
	- 이러한 점들은 미차지된 위치를 나타내는 label 0으로 나타낸다.

- binary cross entropy와 샘플링 함수를 결합하여 식 (2)에서 사용되는 loss는 모든 point cloud의 모든 위치에 대한 binary cross entropy의 평균으로 정의된다:
- $$\mathcal{L}_{cls} = \frac{1}{K}\sum\limits^{K}_{i=1} B[m_{\phi}(G_{i}), \, 1] + B[m_{\phi}(s(G_{i})), \, 0] \quad\quad\quad (4)$$
- 여기서 $G_{i}$는 L-Net parameter $\theta$의 함수이고 $B[m_{\phi}(G_{i}), 1]$는 모든 point cloud $G_{i}$에 대한 평균 BCE error를 나타낸다.
- $B[m_{\phi}(s(G_{i})), 0]$은 point cloud $G_{i}$에 대응하는 샘플링된 미차지(unoccupied) 위치에 대한 평균 BCE error를 의미한다.

- Figure 3에서 식 (4)에 대한 직관을 나타낸다.
	- figure 3의 (a)처럼 registration이 정확하면 loss function은 더 적은 값에 도달한다.
	- 반면에 figure 3의 (b)처럼 잘못 정렬된 point clouds는 self-contradictory occupany status를 초래하므로 loss function은 큰 값을 가진다.

- 식 (4)의 loss는 point cloud의 내재된 occupancy 상태에 의존하기 때문에 외부에서 라벨링된 ground truth에 의존하지 않는 unsupervised 학습이다.
- 최소화를 위해 Loss function $\mathcal{L}_{cls}$이 $\theta$와 $\phi$  모두에 대해 미분 가능하기 때문에 gradient 기반 최적화를 사용한다.

- 이전 연구들에서 이산형 occupancy map을 사용하는 것과 달리 M-Net은 소수점 좌표를 직접 입력하여 연속적인 occupancy map을 생성하기 때문에 임의의 스케일과 해상도로 환경을 표현할 수 있다는 장점이 있다.

---

### 3.3. Network Achitecture
#### L-Net
- Localization Network인 L-Net의 목표는 global frame에서 sensor pose $T_{i}$를 추론하는 것이다. 이 모듈은 입력 point cloud $S_{i}$의 형식에 따라 달라진다.
- 공간 관계 보존된 이미지 형태의 데이터
	- $S_{i}$가 depth나 disparity 이미지로 부터 생성된 구조화된 point cloud라면 $S_{i}$는 인접한 점 사이의 공간 관계가 보존된 포인트 배열로 구성된다.
	- 이러한 공간 관계가 보존된 $S_{i}$는 CNN을 적용하여 point cloud의 feature vector를 추출하고 local feature를 global feature로 집계하기 위해 global pooling layer를  사용한다.
- 공간적인 순서 없이 정렬되지 않은 Raw point cloud
	- point cloud로 부터 feature를 추출하기 위해 PointNet[33] 구조를 채택한다.
	- [33]에 있는 input과 feature transformation을 제거하고 각 D차원 포인트 좌표를 고차원 feature space에 매핑하는 shared MLP를 사용한다.
	- 모든 point에 대해 global pooling layer가 적용되어 feature를 집계하고 추출된 latent feature vector는 sensor의 pose 자유도를 출력하는 MLP로 처리한다.

#### M-Net
- occupancy map network M-Net은 global space에서 위치 좌표를 입력으로 받아 각 입력 위치에 대해 대응되는 occupancy 확률을 예측하는 이진 분류 network이다.
	- M-Net은 모든 point에 대해 공유되고 sigmoid 함수의 1개 채널 출력을 가지고 D채널의 입력을 갖는 MLP이다.
- 

### 3.4. Extension to Lidar SLAM
- 식 (4)의 loss function은 입력 point cloud $\mathbf{S}$를 시간적으로 정렬된 순서가 아닌 비정렬된 스캔의 집합으로 처리한다. 
- 일부 apps에서는 시간 정보를 사용할 수 있는데 Lidar SLAM은 레이저 스캐너를 사용하여 미지의 환경을 탐색하고 시간 t에서 서로 다른 순서의 point cloud를 캡처한다.

- DeepMapping을 확장하여 이런 시간적 관계를 활용함. 시간적으로 연속적인 point cloud는 서로 비슷한 point cloud를 많이 가지고 있기 때문에 잘 겹쳐질 것이다. 
	- 시간적으로 서로 가까운 point cloud간의 기하적 제약(constraints)를 활용한다.
	- 전역 좌표계에서 두 point cloud X와 Y 사이의 거리를 측정하는 metric으로 chamfer distance를 사용한다.
	- $$
\begin{split}
d(X,Y) &= \frac{1}{|X|}\sum\limits_{x\in X} \underset{y \in Y}{min}||x-y||_{2}
\\     &+ \frac{1}{|Y|}\sum\limits_{y \in Y} \underset{x \in X}{min}||x - y||_{2}
\end{split} \quad \quad \quad (5)
$$
	- chamfer distance는 한 point cloud의 각 point에서 다른 point cloud의 가장 가까운 점까지의 양방향 평균 거리를 측정한다.
	- chamfer distance $d(G_{i}, G_{j})$의 최소화는 두 point cloud $G_{i}$와 $G_{j}$사이가 pairwise alignment된다.
	- chamfer 거리를 DeepMapping에 통합하기 위해 식 (2)의 objective 함수를 다음과 같이 수정한다
	- $$(\theta^{*},\phi^{*}) = \underset{\theta, \phi}{argmin} \, \mathcal{L_{cls}}+\lambda \mathcal{L}_{ch}$$
	- 여기서 $\lambda$는 두 Loss 함수의 균형을 잡기 위한 hyperparameter이고 $\mathcal{L}_{ch}$는 각 point cloud $G_{i}$와 시간적 이웃인 $G_{j}$ 사이의 평균 chamfer distance로 정의된다.
	- $$\mathcal{L}_{ch}=\sum\limits^{K}_{i=1}\sum\limits_{j \in \mathcal{N}(i)} d(G_{i}, G_{j}) \quad\quad\quad (7)$$
---

### 3.5. Warm Start
- 식 (6)을 네트워크 매개변수의 무작위 초기화(즉, Cold start)로 최적화하는 것은 실제 app에서 수렴이 오래걸리기 때문에 좋지 않다.
	- 따라서 모든 point cloud를 incremental ICP와 같은 기존 방법으로 미세 정합한 후에 Warm start 방식으로 DeepMapping을 학습한다.





## Reference

---
[1]: Particle swarm optimization. [GitHub link](https://github.com/iralabdisco/pso) registration. 6

[2]: PyTorch. [Official website](https://pytorch.org/). 5

[3]: https://googlle.com "Dror Aiger, Niloy J Mitra, and Daniel Cohen-Or. 4-points congruent sets for robust pairwise surface registration. In ACMTrans. Graphics, volume 27, page 85, 2008. 2"

[4]: Phil Ammirato, Patrick Poirson, Eunbyung Park, Jana Kosecka, and Alexander C. Berg. A dataset for developing and benchmarking active vision. In Proc. the IEEE Intl. Conf. on Robotics and Auto., 2017. 5, 6, 8

[5]: Andrea Banino, Caswell Barry, Benigno Uria, Charles Blundell, Timothy Lillicrap, Piotr Mirowski, Alexander Pritzel, Martin J Chadwick, Thomas Degris, Joseph Modayil, et al. Vector-based navigation using grid-like representations in artificial agents. Nature, 557(7705):429, 2018. 3

[6]: P. J. Besl and N. D. McKay. A method for registration of 3-D shapes. IEEE Trans. Pattern Anal. Mach. Intel., 14(2):239-256, 1992. 2, 6

[7]: Michael Bloesch, Jan Czarnowski, Ronald Clark, Stefan Leutenegger, and Andrew J. Davison. CodeSLAM- learning a compact, optimizable representation for dense visual SLAM. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., June 2018. 2

[8]: Eric Brachmann, Alexander Krull, Sebastian Nowozin, Jamie Shotton, Frank Michel, Stefan Gumhold, and Carsten Rother. DSAC- differentiable ransac for camera localization. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., July 2017. 2

[9]: Samarth Brahmbhatt, Jinwei Gu, Kihwan Kim, James Hays, and Jan Kautz. Geometry-aware learning of maps for camera localization. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., June 2018. 1, 2

[10]: Yang Chen and G´erard Medioni. Object modelling by registration of multiple range images. Image and Vision Comput. 10(3):145–155, 1992. 2, 6

[11]: Sungjoon Choi, Q. Zhou, and V. Koltun. Robust reconstruction of indoor scenes. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., June 2015. 2, 7, 8

[12]: Martin Danelljan, Giulia Meneghetti, Fahad Shahbaz Khan, and Michael Felsberg. A probabilistic framework for color based point set registration. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., pages 1818-1826, 2016. 2

[13]: Haowen Deng, Tolga Birdal, and Slobodan Ilic. PPFNet: Global context aware local features for robust 3D point matching. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., June 2018. 2

[14]: Gil Elbaz, Tamar Avraham, and Anath Fischer. 3D point cloud registration for localization using a deep neural network auto-encoder. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., pages 2472-2481. IEEE, 2017. 2

[15]: SM Ali Eslami, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S Morcos, Marta Garnelo, Avraham Rudermand, Andrei A Rusu, Ivo Danihelka, Karol Gregor, et al. Neural scene representation and rendering. Science, 360(6394):1204–1210, 2018. 3

[16]: Georgios D Evangelidis, Dionyssos Kounades-Bastian, Radu Horaud, and Emmanouil Z Psarakis. A generative model for the joint registration of multiple point sets. In Euro. Conf. on Comp. Vision, pages 109-122. Springer, 2014. 2, 3

[17]: Martin A Fischler and Robert C Bolles. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM, 24(6):381-395, 1981. 2

[18]: Marco Fraccaro, Danilo Jimenez Rezende, Yori Zwols, Alexander Pritzel, SM Eslami, and Fabio Viola. Generative temporal models with spatial memory for partially observed environments. In Intl. Conf. on Mach. Learning, 2018. 1, 2

[19]: Vitor Guizilini and Fabio Ramos. Learning to reconstruct 3D structures for occupancy mapping from depth and color information. Intl. J. of Robotics Research, 2018. 1, 3

[20]: Joao F Henriques and Andrea Vedaldi. MapNet: An allocentric spatial memory for mapping environments. IEEE Intl. Conf. Comp. Vision and Pattern Recog., 2018. 1, 2

[21]: Berthold KP Horn. Closed-form solution of absolute orientation using unit quaternions. J. Opt. Soc. Am. A, 4(4):629-642, 1987. 5

[22]: Shahram Izadi, David Kim, Otmar Hilliges, David Molyneaux, Richard Newcombe, Pushmeet Kohli, Jamie Shotton, Steve Hodges, Dustin Freeman, Andrew Davison, et al. Kinectfusion: real-time 3D reconstruction and interaction using a moving depth camera. In ACM Symp. User Interface Software and Technology, pages 559-568, 2011. 2

[23]: Bing Jian and Baba C Vemuri. A robust algorithm for point set registration using mixture of gaussians. In IEEE Intl. Conf. Comp. Vision, volume 2, pages 1246-1251, 2005. 2

[24]: Zi Jian Yew and Gim Hee Lee. 3DFeat-Net: Weakly supervised local 3D features for point cloud registration. In Euro. Conf. on Comp. Vision, September 2018. 2

[25]: Andrew E Johnson and Martial Hebert. Using spin images for efficient object recognition in cluttered 3D scenes. IEEE Trans. Pattern Anal. Mach. Intel., (5):433-449, 1999. 2

[26]: Alex Kendall, Matthew Grimes, and Roberto Cipolla. PoseNet: A convolutional network for real-time 6-DOF camera relocalization. In IEEE Intl. Conf. Comp. Vision, December 2015. 1, 2

[27]: Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Intl. Conf. Learning Representations, 2015. 5

[28]: Huan Lei, Guang Jiang, and Long Quan. Fast descriptors and correspondence propagation for robust global point cloud registration. IEEE Trans. Image Proc., 26(8):3614-3623, 2017. 2

[29]: J. Li, H. Zhan, B. M. Chen, I. Reid, and G. H. Lee. Deep learning for 2D scan matching and loop closure. In IEEE Intl. Conf. Intel. Robots and Sys., pages 763-768, Sept 2017. 1, 2

[30]: Nicolas Mellado, Dror Aiger, and Niloy J Mitra. Super 4PCS fast global pointcloud registration via smart indexing. In Comp. Graphics Forum, volume 33, pages 205-215, 2014. 2

[31]: A Myronenko and Xubo Song. Point set registration: Coherent point drift. IEEE Trans. Pattern Anal. Mach. Intel., 32(12):2262-2275, 2010. 2

[32]: Emilio Parisotto, Devendra Singh Chaplot, Jian Zhang, and Ruslan Salakhutdinov. Global pose estimation with an attention-based recurrent network. In IEEE Intl. Conf. Comp. Vision and Pattern Recog. Wksp., pages 237-246, 2018. 2

[33]: Charles R. Qi, Hao Su, Kaichun Mo, and Leonidas J. Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., July 2017. 4

[34]: Szymon Rusinkiewicz and Marc Levoy. Efficient variants of the ICP algorithm. In 3D Digital Imaging and Modeling, pages 145-152, 2001. 2

[35]: Radu Bogdan Rusu, Nico Blodow, and Michael Beetz. Fast point feature histograms (FPFH) for 3D registration. In Proc. the IEEE Intl. Conf. on Robotics and Auto., pages 3212-3217, 2009. 2

[36]: Radu Bogdan Rusu, Nico Blodow, Zoltan Csaba Marton, and Michael Beetz. Aligning point cloud views using persistent feature histograms. In IEEE Intl. Conf. Intel. Robots and Sys., pages 3384-3391, 2008. 2

[37]: Johannes L Schönberger, Marc Pollefeys, Andreas Geiger, and Torsten Sattler. Semantic visual localization. ISPRS J. Photographic and Remote Sensing, 2018. 2

[38]: Paul Scovanner, Saad Ali, and Mubarak Shah. A 3-dimensional SIFT descriptor and its application to action recognition. In ACM Intl. Conf. Multimedia, pages 357-360, 2007. 2

[39]: Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun RGB-D: A RGB-D scene understanding benchmark suite. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., pages 567-576, 2015. 5, 6

[40]: Bastian Steder, Radu Bogdan Rusu, Kurt Konolige, and Wolfram Burgard. NARF: 3D range image features for object recognition. In Wksp. on Defining and Solving Realistic Perception Problems in Personal Robotics at IEEE Intl. Conf. Intel. Robots and Sys., volume 44, 2010. 2

[41]: J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. A benchmark for the evaluation of RGB-D SLAM systems. In IEEE Intl. Conf. Intel. Robots and Sys., Oct. 2012. 5, 6

[42]: Pascal Willy Theiler, Jan Dirk Wegner, and Konrad Schindler. Globally consistent registration of terrestrial laser scans via graph optimization. ISPRS J. Photographic and Remote Sensing, 109:126-138, 2015. 2

[43]: Federico Tombari, Samuele Salti, and Luigi Di Stefano. Unique signatures of histograms for local surface description. In Euro. Conf. on Comp. Vision, pages 356-369, 2010. 2

[44]: Andrea Torsello, Emanuele Rodola, and Andrea Albarelli. Multiview registration via graph diffusion of dual quaternions. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., pages 2441-2448, 2011. 2

[45]: Benjamin Ummenhofer, Huizhong Zhou, Jonas Uhrig, Nikolaus Mayer, Eddy Ilg, Alexey Dosovitskiy,

 and Thomas Brox. DeMoN: Depth and motion network for learning monocular stereo. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., volume 5, page 6, 2017. 1, 2

[46]: J. Yang, H. Li, D. Campbell, and Y. Jia. Go-ICP: A globally optimal solution to 3D ICP point-set registration. IEEE Trans. Pattern Anal. Mach. Intel., 38(11):2241-2254, Nov 2016. 2, 6

[47]: Nan Yang, Rui Wang, Jörg Stückler, and Daniel Cremers. Deep virtual stereo odometry: Leveraging deep depth prediction for monocular direct sparse odometry. In Euro. Conf. on Comp. Vision, pages 835-852, 2018. 2

[48]: Andy Zeng, Shuran Song, Matthias Nießner, Matthew Fisher, Jianxiong Xiao, and Thomas Funkhouser. 3Dmatch: Learning local geometric descriptors from RGB-D reconstructions. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., pages 199-208, 2017. 2

[49]: Huizhong Zhou, Benjamin Ummenhofer, and Thomas Brox. DeepTAM: Deep tracking and mapping. In Euro. Conf. on Comp. Vision, September 2018. 1, 2

[50]: Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Fast global registration. In Euro. Conf. on Comp. Vision, pages 766-782, 2016. 2, 3

[51]: Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3D: A modern library for 3D data processing. arXiv:1801.09847, 2018. 7

[52]: Tinghui Zhou, Matthew Brown, Noah Snavely, and David G Lowe. Unsupervised learning of depth and ego-motion from video. In IEEE Intl. Conf. Comp. Vision and Pattern Recog., volume 2, page 7, 2017. 1, 2



# Work

## 2024.02.21
- 179호기에서 4층 라이다 데이터 수집("E:\Data\syswin\20240221-131246")
	- 총 11,033개 cloud point / 360$^o$ / 1083개 points / 34hz
		- ![[Pasted image 20240221150608.png]]
	- data format
		- local X, local Y, local degree, start degree, end, degree, n_points, distances~ ...
	- ![[Pasted image 20240221145523.png | local 정보로 라이다를 뿌려봄 | 500]]
	- ![[Pasted image 20240221145959.png | local이 흔들려서 같은 벽에대한 Lidar point가 많이 흔들림 | 250]]
	- 현재 가진 데이터수가 너무 많아서 34hz의 1/30 , 1083개 point 중에 1/4만 사용
		- point clouds 11,083개 -> 369개 / 1038개 point중에서 270개  shape :[369, 270, 2]
		- training tact time : 1 epoch 당 3초 

## 2024.02.22
- 4층에서 가진 cloud points로 7500 epoch 학습 결과물 0 epoch -> 7500 epoch![[global_map_pose_e_40.png | 400]]![[global_map_pose_e_7500.png | 400]]
- 학습된 deepmapping 모델에서 localization network만 가지고 새로운 local pose가 들어왔을때 global pose를 estimation 
	- 빨간색 점은 학습된 point clouds / 검은색 점은 새로운 point clouds
- ![[Pasted image 20240223113559.png]]![[Pasted image 20240223113638.png]]
- 새로운 local pose에 대한 global pose가 안좋은 성능을 가짐

## 2024.02.23
### Todo 
- Deepmapping으로 생성된 Map과 실제 Map과 비교 (정이사님 질문)
	- 벽두께, 스케일 비교
		- 실제 맵과 비교를 위해서 Lidar point clouds를 .pgm 형식의 Map으로 변환하는 과정이 필요함(코드 서칭중..)
	- Deepmapping으로 생성된 Map내의 노이즈(움직이는 사람, 물체) 제거 방법?
- 현재 학습된 DeepMapping 모델에서 Localization Net으로 새로운(학습에 사용되지 않은) local pose를 global pose로 변환하면 위 결과처럼 안좋음
	- 학습하는 과정에서 Localization Net의 일반화가 안됨
		1. Deep Closest Point으로 정합(registration)된 Map point clouds에 새로운 local point cloud를 point matching하는 방법으로 대체
		2. Localization Map을 학습할때 일반화 되도록 학습

#### ROS Map File 
- .pgm 파일
	- 맵의 이진 이미지 픽셀은 3가지 값으로 구성됨(205 - Unknown, 254 - move possible, 0 - 벽)
	- ![[4F_1204.png | 200]]
	
- .yaml 파일 
	- image 이름, mode, resolution, origin, negate, occupied_thresh, free_thresh
- ROS에서 Map 파일 생성하는 코드 참조
	- https://github.com/HaoQChen/map_server/blob/master/src/map_saver.cpp


## 2024.02.26
### ToDo
- cloud data로 정합된 데이터를 Map으로 변환하는 법
- 기존 Cartographer로 생성된 Map과 성능비교
- Localization Net 성능 향상