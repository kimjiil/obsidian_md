# Paper
## 1. Introduce
- 이 논문의 기여는 loop closure 제약 조건을 계산하는데 필요한 계산 요구 사항을 줄여 매우 빠르고 더 넓은 지역을 매핑 가능하게 함
## 2. Related Work
- scan-to-scan matching은 레이저 기반 SLAM 방법에서 자주 사용되지만 [1]~[4]에서 설명한 것처럼 scan-to-scan maching은 오류를 빠르게 누적시킨다.
- Scan-to-map matching은 이런 scan-to-scan matching의 오류누적을 제한한다
- 위치 오류 누적을 처리 하기 위한 두 가지 일반적인 접근 방법은 particle filter와 graph-based SLAM[2], [8] 이다.
- Particle filter는 각 입자마다 전체 시스템 상태의 표현을 유지해야하기 때문에 격자 기반 SLAM의 경우 맵이 커질수록 컴퓨팅 자원을 많이 소모하는 문제가 발생함
	- [10]은 필요할때만 업데이트되는 submap을 계산하여 최종 맵이 모든 sub map의 rasterization된 결과가 된다.
- 그래프 기반 접근 방식은 Pose와 feature를 나타내는 노드 모음을 기반으로 동작한다.
	- 그래프의 edge는 관측으로부터 생성된 제약 조건

## 3. System Overview
- 구글의 카토그래퍼는 2D grid map을 생성하는 실시간 실내 매핑 솔루션을 제공
	- grid map의 해상도는 r=5cm
	- 레이저 스캔은 최적의 추정 위치에 submap에 삽입된다.
	- scan matching은 최근 submap에 대해 수행되므로 최근 scan에만 의존하며 world frame에서 pose 추정 오류가 누적된다.
- 적은 컴퓨팅 자원을 소모하기 위해 입자필터를 사용하지 않는다.
- 오류의 누적을 없애기 위해 주기적으로 pose optimization을 한다.
- 서브맵이 완료되면 더 이상 새로운 스캔이 삽입되지 않으므로 루프 클로저를 위해 스캔 매칭에 참여한다.
	- 완료된 모든 서브맵과 스캔은 자동으로 루프 클로저를 위해 고려된다.
- 현재의 pose 추정에 따라 충분히 가깝다면, 스캔 매처는 submap에서 스캔을 찾으려고 시도한다. 현재 추정된 pose 주변의 검색창에서 충분히 좋은 매치가 발견되면, 그것은 최적화 문제에 루프 폐쇄 제약으로 추가된다.
- 루프 폐쇄 스캔 매칭이 새로운 스캔이 추가되는 것보다 빨리 일어나야 한다
	- branch-and-bound 접근 방식과 각 완성된 submap별로 여러 개의 사전 계산된 grid를 사용하여 빠른 계산을 달성한다.

## 4. Local 2D SLAM
- 2D SLAM을 위해 분리된 Local 및 global 접근 방식을 결합한다.
	- 두 접근 방식 모두 Lidar 관측치의 pose $\xi=(\xi x, \xi y, \xi \theta)$를 최적화 하는데 이는 translation 변환과 rotation 변환으로 구성된다.  
- local 접근 방식에서는 각 연속된 스캔이 global의 작은 부분인 submap M에 대해 매치되며 이는 스캔을 하위맵과 정렬하는 비선형 최적화를 사용하여 수행하는 이 과정을 Scan Matching이라 한다.
	- scan matching은 시간에 지남에 따라 오류를 누적시키는데 이는 5장에서 설명된 global 접근 방식을 통해 제거된다.
### 4.A. Scans
- submap 구성은 scan과 submap 좌표계를 반복적으로 정렬하는 과정으로 이를 frame으로 지칭한다. 
- 스캔의 원점을 영점이라 할때, scan point에 대한 정보를 $H=\{h_{k}\}_{k=1,...,K}$ , $h_{k} \in \mathbb{R}^{2}$ 으로 표기한다. scan frame의 pose $\xi$는 submap frame에서 변환 $T_{\xi}$ 로 표현되며, 이 변환은 scam frame의 scan point을 submap frame으로 강체 변환 한다. 이는 다음과 같이 정의한다.
$$
T_{\xi p} = \begin{pmatrix} \cos\xi_{\theta} & -\sin\xi_{\theta} \\ \sin\xi_{\theta} & \cos\xi_{\theta} \end{pmatrix} p + \begin{pmatrix} \xi_{x} \\ \xi_{y} \end{pmatrix}
$$

### 4.B. Submaps
- 몇개의 연속적인 scan 데이터를 가지고 submap을 구성한다. 이러한 서브맵은 확률 그리드 $M : r\mathbb{Z} \times r\mathbb{Z} \rightarrow [p_{min},\; p_{max}]$ 의 형태를 취하며, 주어진 해상도 $r$, 예를들어 5cm에서 이산 그리드 포인트에 매핑합니다.
	- 이 값들은 그리드 포인트가 막혔을 가능성의 확률로 생각할 수 있다. 각 그리드 포인트에 대해 해당 픽셀은 해당 그리드 포인트에 가장 가까운 모든 점으로 구성된다.
	- ![[Pasted image 20240422104142.png]] 
- Scan이 그리드 확률에 삽입될 때, 적중을 위한 그리드 포인트 집합과 miss를 위한 분리된 집합이 계산된다. 적중(hit)이 감지될 때마다 가장 가까운 그리드 포인트를 적중 집합에 추가한다. miss의 경우 scan 원점과 scan point 사이의 광선이 교차하는 각 픽셀과 연관된 그리드 포인트를 miss 집합에 추가한다. 이는 이미 적중 집합에 포함되지 않은 그리드 포인트에만 적용된다. 이전에 관찰되지 않은 모든 그리드 포인트는 해당 집합에 속하는 경우 $p_{hit}$ 또는 $p_{miss}$ 확률이 할당된다. 이미 관찰된 그리드 포인트 x의 경우, 적중과 실패의 확률을 다음과 같이 업데이트한다.

$$odds(p)=\frac{p}{1-p}, \quad\quad\quad (2)  $$$$M_{new}(x)=clamp(odds^{-1}(odds(M_{old}(x)) \cdot odds(p_{hit}))) \quad \quad \quad (3) $$
### 4.C. Ceres scan matching
- Scan을 submap에 삽입하기 전에, Ceres 기반의 Scan matcher를 사용하여 현재 local submap에 대해 scan pose $\xi$ 를 최적화합니다. 
	- 이 Scan matcher는 submap에서 scan point의 확률을 최대화하는 scan pose를 찾는다. 이 문제는 비선형 최소 제곱 문제로 설정된다.
	- $$\underset{\xi}{argmin} \sum\limits^{K}_{k=1}(1-M_{smooth}(T_{\xi}h_{k}))^{2}$$
	- 여기서 $T$는 scan frame에서 submap frame으로 $h_{k}$를 변환하는 transform을 나타내며, scan pose에 따라 결정된다. 함수 $M_{smooth}: \mathbb{R}^{2} \rightarrow \mathbb{R}$ 은 local submap의 확률 값을 부드럽게(smooth)하는 함수이다. 이떄, bicubic interpolation을 사용한다.
	- 결과 $[0, 1]$구간 밖의 값이 발생할 수 있지만 이러한 값들은 시스템에 큰 영향을 주지 않는다.
- smooth function의 수학적 최적화는 일반적으로 그리드의 해상도보다 더 좋은 정밀도를 제공한다. 이는 local optimization이기 때문에 좋은 초기 추정치를 요구한다.
- 각속도를 측정할 수 있는 IMU는 scan match간의 포즈의 회전 성분을 추정하는데 사용될 수 있다. 더 높은 scan match 빈도나 픽셀 정밀 스캔 매칭 접근법은 계산상 더 많은 자원을 요구하지만 IMU가 없는 경우에도 사용될 수 있다.
## 5. Closing Loops
- Scan은 최근 몇개의 scan만을 포함하는 submap에 대해서만 매칭되므로, 위에 설명한 접근법은 점진적으로 오류가 누적된다. 단지 몇십 개의 연속된 Scan에 대해서는 누적오류가 작습니다.
- 더 큰 공간은 많은 작은 서브맵을 많어 처리한다. 우리의 접근법은 모든 scan과 submap의 포즈를 최적화하며, 이는 Sparse Pose Adjustment [2]를 따릅니다. 
- Scan이 삽입되는 상대적 포즈는 loop closing optimization을 위해 메모리에 저장된다. 이러한 상대적 포즈외에도, 서브맵이 더 이상 변경되지 않으면 스캔과 서브맵으로 구성된 모든 쌍이 loop closing을 위해 고려된다. scan macher는 백그라운드에서 실행되며 좋은 match가 발견되면 해당 상대적 포즈가 최적화 문제에 추가된다.
### 5.A. Optimization problem
- loop closure 최적화 역시 스캔 매칭과 같이 비선형 최소 제곱 문제로 설정되며, 추가 데이터를 고려하여 잔차를 쉽게 추가할 수 있다. 몇 초마다, 우리는 Ceres[14]를 사용하여 다음과 같은 해를 계산한다.
$$
\underset{\Xi^{m}, \Xi^{x}}{argmin} \frac{1}{2} \sum\limits_{ij} \rho(E^{2}(\xi^{m}_{i}, \xi^{s}_{j}; \Sigma_{ij}, \xi_{ij})) \qquad \qquad (SPA) 
$$
- 여기서 submap poses $\Xi^{m}=\{ \xi^{m}_{i}\}_{i=1,...,m}$와 global  scan poses $\Xi^{s}=\{\xi^{s}_{j} \}_{j=1,...,n}$가 주어진 제약 조건을 고려하여 최적화 한다. 이러한 제약 조건은 상대적 포즈 $\xi_{ij}$와 관련된 공분산 행렬 $\Sigma_{ij}$ 의 형태를 취합니다
- submap $i$와 scan $j$쌍에 대한 pose $\xi_{ij}$는 Scan이 submap 좌표계에서 어디에 매칭되었는지를 설명한다. 공분산 행렬 $\Sigma_{ij}$ 는 예를들어 [15]의 접근 방식을 따르거나 (CS)의 Ceres[14]의 공분산 추정 기능을 사용하여 로컬로 평가될 수 있다.이러한 제약에 대한 잔차 $E$는 다음과 같이 계산된다.
$$
E^{2}(\xi^{m}_{i}, \xi^{s}_{j};\Sigma_{ij}, \xi_{ij})=e(\xi^{m}_{i},\xi^{s}_{j};\xi_{ij})^{T}\Sigma^{-1}_{ij}e(\xi^{m}_{i},\xi^{s}_{j};\xi_{ij}), \quad \quad(4) 
$$
$$
e(\xi^{m}_{i},\xi^{s}_{j};\xi_{ij})=\xi_{ij}- \begin{pmatrix} R^{-1}_{\xi^{m}_{i}}(t_{\xi^{m}_{i}}-t_{\xi^{s}_{j}}) \\ \xi^{m}_{i;\theta} - \xi^{s}_{j;\theta} \end{pmatrix} \quad\quad (5)
$$
- loss 함수 $\rho$ , 예를 들어 $Huber \; loss$는 scan matching이 최적화 문제에 잘못된 제약을 추가할 때 발생할 수 있는 이상치의 영향을 줄이는데 사용된다. 예를들어, 이는 사무실 칸막이와 같은 국소적으로 대칭적인 환경에서 발생할 수 있다. 이상치에 대한 대체 접근법에는 [16]이 포함됩니다. 

### 5.B. Branch-and-bound scan matching
- 우리는 최적의 픽셀 정확도 매칭에 관심이 있다. 이는 수식으로 다음과 같이 표현된다. 
$$
\xi^{*}=\underset{\xi \in \mathcal{W}}{argmax} \sum\limits^{K}_{k=1}M_{nearest}(T_{\xi}h_{k}), \qquad (BBS)
$$
- 여기서 $\mathcal{W}$는 search window이고 $M_{nearest}$는 그 인자를 가장 가까운 그리드 포인로 반올림함으로써 $M$이 $\mathbb{R}^{2}$ 전체로 확장된 것이다.  즉, 그리드 포인트의 값을 해당 픽셀에 확장하는 것을 의미한다. 매칭의 품질은 추가적으로 (CS)를 사용하여 더욱 향상 될 수 있다.
- 효율성은 스텝 사이즈를 신중하게 선택함으로써 향상된다. 우리는 scan point가 최대 범위 $d_{max}$에서 한 픽셀의 너비인 r보다 더 이동 하지 않도록 각도 step size $\delta_{\theta}$ 를 선택한다. 코사인 법칙을 사용하여 다음을 유도한다.
$$
d_{max}=\underset{k=1,...,K}{max}||h_{k}|| \qquad (6)
$$
$$
\delta_\theta=arccos\left(1-\frac{r^{2}}{2d^{2}_{max}}\right) \qquad (7)
$$
- 주어진 선형 및 각도 검색 창 크기를 커버하는 정수 단계 수를 계산한다. 예를 들어 $W_{x}=W_{y}=7m$

 
