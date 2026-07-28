# 🏊:Video-Based Drowning Detection

## 📋프로젝트 개요

**프로젝트 명:**
수영장 환경에서 익수(Drowning)행동을 실시간(RTSP)으로 감지하는 비전 안전시스템 

**프로젝트 기간:**
2025년 ~ 2026년

**프로젝트 목표: 실시간 스트림에서 익수자 탐지**
> 핵심 스택: YOLO + SAM2 (탐지·추적) → VideoMAE v2 K710 + LoRA finetuning (행동 인식) / VLM (보조 분석)

---

## 🔬Pipeline

```
[Stage 0] 원본 영상 / RTSP 스트림
            │  YOLO 사람 탐지 + SAM2 추적
[Stage 1] 객체별 4초 crop 클립 생성
            │
[Stage 2] VideoMAE v2 K710 pretrain + LoRA finetuning → 행동 분류
            │
[Demo]    realtime_detection_lora.py — 탐지·추적·분류 통합 데모
```

---
## Detection - Tracking - Classification 전체 파이프라인 구동 데모 영상
https://github.com/user-attachments/assets/adac6fdd-c303-48cb-aeaf-0d1077ad5fe8

## 🛠️사용 기술 스택
|분류|기술|설명|
|------|---|---|
|Detection|YOLO|단일 이미지(frame)에서 객체의 위치를 탐지하는 모델|
|Segmentation|SAM2 & cutie|한 프레임 내에서 물체 분할|
|Action classification backbone|VideoMAEv2 (OpenGVLab/vit_g_hybrid_pt_1200e_k710_ft)|프레임 묶음 바탕의 행동 특징 추출 백본|
|VLM|Qwen3-VL-4B-Thinking-FP8|비디오 이해 및 추론이 가능한 Vision-Language 모델|
|추론 엔진|vllm|고속 LLM 추론 서빙|
|딥러닝 프레임워크|Pytorch, Transformer|모델 로딩 및 처리|

## Directory Structure

```
Video-Based-Drowning-Detection/
│
├── 탐지 · 추적 (YOLO + SAM2 streaming)
│   ├── test_sam2_track.py            # 추적 파이프라인
│   ├── test_sam2_track_roi.py        # 추적 + ROI 영역 제한
│   ├── test_sam2_rtsp_roi_crop.py    # 추적 + ROI + crop 클립 자동 저장 (학습데이터 생성)
│   ├── test_sam2_crop_object.py      # 익수자 수동 선택 → 클립화 도구
│   └── roi_make.py                   # ROI 폴리곤 라벨링 도구
│
├── 실시간 데모
│   ├── realtime_detection_lora.py    # 최종 LoRA 모델 통합 데모
│   └── realtime_detection.py         # head finetune 버전 (비교용)
│
├── action_recognition/               # 행동 인식 (VideoMAE v2 + LoRA)
│   ├── train_lora.py                 # LoRA 학습
│   ├── test_f1_lora.py               # per-class P/R/F1 평가
│   ├── eval_test_data.py             # 2-class 평가
│   ├── eval_inference_time.py        # 추론 속도 측정
│   ├── models/                       # head.py, lora_backbone.py
│   └── data/                         # video_dataset.py (mp4 → 16-frame)
│
├── vlm/                              # VLM 추론 (별도 환경)
│   ├── zeroshot_eval.py              # Qwen VLM 제로샷 익수 판단 평가
│   └── analyze_drowning.py           # vLLM 오프라인 익수행동 분석
│
├── requirements.txt
└── .gitignore
```

## Setup

### Environment

- Python 3.10 

```bash
conda create -n drowning python=3.10 -y
conda activate drowning
pip install -r requirements.txt
```


### 가중치 / 체크포인트

직접 학습한 가중치 2개 (Object detection, action classification model) 저장소에 포함


### 데이터셋 구성

**전체 데이터**
|항목|수량|설명|
|------|---|---|
|야외 수영장 영상|73|야외 수영장 CCTV|
|실내 수영장 영상|9|실내 수영장 CCTV|
|행동 영상 (4 sec)|749|익수, 걷기, 서있기, 수영, 떠있기 행동 영상|

**학습 데이터**
|항목|수량|설명|
|------|---|---|
|익수 영상 (4 sec)|77|익수행동이 포함된 crop 영상|
|일반 영상 (4 sec)|398|걷기, 서있기, 수영, 떠있기, 물놀이가 포함된 crop 영상|

**검증 데이터**
|항목|수량|설명|
|------|---|---|
|익수 영상 (4 sec)|11|익수행동이 포함된 crop 영상|
|일반 영상 (4 sec)|123|걷기, 서있기, 수영, 떠있기, 물놀이가 포함된 crop 영상|


**테스트 데이터**
|항목|수량|설명|
|------|---|---|
|익수 영상 (4 sec)|18|익수행동이 포함된 crop 영상|
|일반 영상 (4 sec)|122|걷기, 서있기, 수영, 떠있기, 물놀이가 포함된 crop 영상|


### 📊 실험 결과 및 성능 분석

**실험 설정**
- 모델: VideoMAEv2 K710 backbone, Qwen3-vl-4B-thinking-FP8
- 실험 영상: 익수 & 일반 행동 crop 영상 (4 sec)
- 비디오 처리: 16 frame 균일 샘플링

**성능 비교**
|방법|F1-score|특징|
|------|---|---|
|VideoMAE v2 cls_head train|0.8800|backbone 고정 & head 학습|
|VideoMAE v2 LoRA ft|0.9091|LoRA 기법으로 저차원 행렬 학습|

**분석**
1. 익수행동의 데이터 수가 적어, 다양한 환경 데이터를 활용하지 못하고 backbone 학습, vlm 파인튜닝과 같은 작업을 수행하기에 한계가 존재.
   -> 실내 수영장 데이터 직접 확보. 다만, 다양한 환경 및 실제 익수행동의 영상 확보에는 어려움 존재
2. K710 데이터셋으로 학습된 foundation model의 backbone을 활용하여 행동특징 분석 능력 확보.
   -> LoRA 기법으로 저차원행렬 학습하여, 분류능력 향상
3. 탐지(object detection) & 추적 (tracking) 모델만 사용했을 때, 객체 놓침 및 ID 변동 빈번하게 발생 
   -> 수영장 환경에서 탐지 모델 학습(Yolo) 및 segmentation (Sam2 & cutie)를 이용하여 탐지 결과와 seg결과를 비교함으로써 객체 놓침 현상 해결

## References

- SAM 2 — *Segment Anything in Images and Videos* (Meta AI)
- VideoMAE v2 — *Scaling Video Masked Autoencoders with Dual Masking* (CVPR 2023)
- LoRA — *Low-Rank Adaptation of Large Language Models*
- Cutie — *Putting the Object Back into Video Object Segmentation* (CVPR 2024)
