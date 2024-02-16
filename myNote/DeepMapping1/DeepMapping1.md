
# DeepMapping: Unsupervised Map Estimation From Multiple Point Clouds
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
![[Pasted image 20240215111108.png | 400]]

- deep learning과 geometric vision problem([45], [52], [49], [20], [19], [29], [18])을 통합하기 위해 다양한 연구를 시도중임
- ([26], [9], [20]) 방법들은 주변 환경의 맵을 representation으로 가지고 있는 DNN을 학습하여 camera pose를 추정하려고 한다.
- ([45], [52])의 방법들은 depth와 움직임 간의 내재적인 관계를 활용하는 비지도 학습 접근법을 제안한다.

- 이 논문의 핵심은 DNN이 기하학적 문제에 대해서, 특히 registration과 mapping에서얼마나 잘 일반화될 것인가
	- Semantic task은 DNN에서 크게 이득을 보고 있는데 해당 문제들은 대부분 경험적으로 정의되어 많은 데이터를 통해 통계적으로 모델링되어 해결된다.
	- 하지만 많은 기하학적 문제들은 이론적으로 정의됨 -> 통계적 모델링을 통한 해결책은 정확도 측면에서 한계가 있음

- 



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