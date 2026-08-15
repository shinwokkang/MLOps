# 1. 어떤 기본 상자(base image)에서 시작할지 지정
#    python:3.11-slim = 파이썬 3.11이 미리 깔린 가벼운 리눅스(Debian) 이미지
FROM python:3.11-slim

# 2. 컨테이너 내부에서 명령어들이 실행될 기준 폴더
WORKDIR /app

# 3. LightGBM이 리눅스에서 필요로 하는 OpenMP 런타임 설치
#    (macOS에서 brew install libomp 했던 것과 같은 역할, 리눅스에서는 libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt만 먼저 복사해서 패키지 설치
#    (코드보다 먼저 복사하면, 코드만 바뀌었을 때 패키지 재설치를 건너뛸 수 있어 빌드가 빨라짐)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 우리 소스 코드와 설정 파일 복사 (데이터는 넣지 않음 - 실행 시 볼륨으로 연결)
COPY src/ src/
COPY conf/ conf/

# 6. 컨테이너가 시작되면 실행할 기본 명령: 전처리 -> 학습 순서로 실행
CMD ["sh", "-c", "python src/preprocess.py && python src/train.py"]
