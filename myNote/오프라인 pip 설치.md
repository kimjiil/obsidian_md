
인터넷 다운로드 받을 수 있는 python 환경 A
오프라인 설치할 python 환경 B
- A, B는 같은 OS환경, python version도 동일해야됨
- A환경에는 라이브러리가 설치되어 있어야됨

1. A환경에서 .whl을 저장할 폴더를 생성하고 pip 명령어를 통해 B환경에 설치할 library 파일 다운(예시 scikit-learn)
		$ pip download -d ./down_dir/ scikit-learn
2. A환경에서 생성된 down_dir 폴더 이하의 파일을 B환경으로 업로드
3. 다음의 명령어로 down_dir 폴더에 있는 .whl을 설치
		$ pip install --no-index --find-links=./down_dir/ scikit-learn

