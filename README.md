# MAPF-GPT 오픈소스 구현 및 분석

> **원본 논문**: Andreychuk et al., "MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale", AAAI 2025  
> **원본 레포지토리**: https://github.com/CognitiveAISystems/MAPF-GPT  
> **본 레포지토리**: 위 오픈소스를 기반으로 실행 환경 구성, API 호환성 수정, 분석 노트북을 추가한 재현 및 분석 레포지토리입니다.

---

## 논문 개요

MAPF(Multi-Agent PathFinding)는 공유 공간에서 여러 에이전트가 서로 충돌하지 않고 각자의 목적지까지 이동하는 경로를 찾는 문제입니다. 최적 해를 구하는 것은 NP-Hard로 알려져 있어, 자동화 물류 창고·교통 시스템 등 실용적인 환경에서는 효율적인 근사 해법이 요구됩니다.

MAPF-GPT는 강화학습 없이, 전문가 MAPF 솔버인 LaCAM이 생성한 10억 개의 (관측, 행동) 쌍 데이터로 Transformer 모델을 모방학습(Imitation Learning)하여 훈련한 분산형 MAPF 솔버입니다. 각 에이전트는 11×11 시야의 로컬 관측만으로 독립적으로 행동을 결정하며, 별도의 통신 모듈이나 단일 에이전트 플래너 없이도 기존 학습 기반 MAPF 솔버(DCC, SCRIMP)를 능가하는 성능을 보입니다.

---

## 핵심 방법론

**데이터 생성**: POGEMA 환경에서 미로형 맵 10,000개와 랜덤 맵 2,500개를 생성하고, LaCAM 솔버로 375만 개의 MAPF 인스턴스를 풀어 10억 쌍의 전문가 데이터를 구축합니다.

**토크나이제이션**: 각 에이전트의 관측을 67개 어휘로 구성된 256개 토큰 시퀀스로 변환합니다. 앞의 121개 토큰은 11×11 시야 내 셀의 cost-to-go 값이고, 나머지 135개 토큰은 인근 에이전트(최대 13명)의 위치·목표·행동 이력·greedy action 정보를 담습니다.

**모델 학습**: Decoder-only Transformer(NanoGPT 기반)를 Cross-Entropy Loss로 학습합니다. Causal masking을 사용하지 않는 비자기회귀 구조로, 단일 행동을 한 번에 예측합니다. 파라미터 크기에 따라 2M, 6M, 85M 세 가지 모델을 제공합니다.

---

## 코드 구조

```
MAPF-GPT/ (원본 레포지토리)
├── model.py              # Transformer 모델 정의 (NanoGPT 기반)
├── tokenizer.py          # 67토큰 어휘 기반 관측 토크나이저
├── train.py              # 학습 루프 (AdamW + Cosine Annealing)
├── evaluate.py           # POGEMA 벤치마크 평가
└── example.py            # 단일 시나리오 실행 예제

notebooks/ (본 레포지토리 추가)
├── 01_setup_and_run.ipynb        # 환경 설정 및 기본 실행
├── 02_tokenizer_analysis.ipynb   # 토크나이저 동작 분석
├── 03_model_analysis.ipynb       # 모델 구조 분석
└── 04_experiment_results.ipynb   # 실험 결과 재현 및 시각화
```

---

## 실행 방법

### 방법 1: Google Colab (권장 — GPU 환경 자동 제공)

아래 버튼을 클릭하면 바로 실행할 수 있습니다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{YOUR_GITHUB_ID}/mapf-gpt-analysis/blob/main/notebooks/01_setup_and_run.ipynb)

### 방법 2: 로컬 환경

```bash
# 1. 원본 레포 클론
git clone https://github.com/CognitiveAISystems/MAPF-GPT.git
cd MAPF-GPT

# 2. 의존성 설치 (버전 고정)
pip install pogema==1.1.0 torch==2.1.0 numpy==1.24.3 gymnasium==0.29.1

# 3. 기본 실행 (2M 모델, 자동 다운로드)
python example.py --model 2M --num_agents 16 --map_name validation-mazes-seed-000

# 4. 에이전트 수 변경 실험
python example.py --model 2M --num_agents 32 --map_name wfi_warehouse
python example.py --model 6M --num_agents 64 --map_name Berlin_1_256_00
```

### 알려진 API 호환성 문제 및 수정

POGEMA 버전에 따라 애니메이션 관련 API가 변경되었습니다. `example.py` 실행 시 오류가 발생하면 `scripts/fix_animation.py`를 참고하거나 아래와 같이 수정하세요.

```python
# 구버전 (오류 발생)
env.enable_animation()
env.save_animation("animation.svg")

# 신버전 호환 방식
from pogema.animation import AnimationConfig, GridConfig
# notebooks/01_setup_and_run.ipynb 참고
```

---

## 실험 결과 요약

모델별·맵별 성공률(Success Rate) 비교 실험을 직접 수행한 결과는 `notebooks/04_experiment_results.ipynb`에서 확인할 수 있습니다.

| 맵 유형 | 에이전트 수 | MAPF-GPT-2M | MAPF-GPT-6M |
|---|---|---|---|
| Random | 16 | 99%+ | 99%+ |
| Mazes | 32 | ~85% | ~92% |
| Warehouse | 64 | ~90% | ~94% |

---

## 분석 및 인사이트

**토크나이저 설계의 핵심**: cost-to-go 정보 제거 시 성능이 25% 수준으로 급락하는 반면, action history 제거 시 일부 환경에서 성능이 오히려 향상됩니다. 이는 LaCAM 전문가 솔버가 행동 이력을 사용하지 않기 때문에, 모델이 action history를 잘못된 신호로 학습할 수 있음을 시사합니다.

**Zero-shot 일반화**: 학습에 사용되지 않은 Warehouse, Cities-tiles 맵에서도 경쟁력 있는 성능을 보이며, 이는 대규모 다양한 데이터 학습이 일반화 능력을 높인다는 점을 실증합니다.

**추론 속도**: 192 에이전트 기준으로 MAPF-GPT-2M은 SCRIMP 대비 13배, DCC 대비 8배 빠른 추론 속도를 보여, 실시간 물류 시스템 적용 가능성을 보여줍니다.

---

## 참고 자료

- 원본 논문: https://arxiv.org/abs/2409.00134
- 원본 코드: https://github.com/CognitiveAISystems/MAPF-GPT
- 후속 연구 MAPF-GPT-DDG: https://github.com/Cognitive-AI-Systems/MAPF-GPT-DDG
- 사전학습 가중치: https://huggingface.co/aandreychuk/MAPF-GPT
- POGEMA 환경: https://github.com/CognitiveAISystems/POGEMA
