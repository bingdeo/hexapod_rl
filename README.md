# EUREKA를 활용한 헥사포드 로봇의 험지 보행 강화학습
Validation of EUREKA’s Effectiveness in Hexapod Rough-Terrain Locomotion

## Simulation

### Overview

2023년 MIT에서 발표한 대규모 언어모델 기반 보상 함수 설계 자동화 프레임워크인 EUREKA를 활용하여 헥사포드 로봇에 적용 가능성을 검토한다. 이를 통해 EUREKA 프레임워크의 적용 범위를 더 높은 복잡성을 지닌 로봇 시스템으로 확장하여 향후 지능형 다족 로봇의 자율 보행 연구에 기여하고자 한다. 

---

### Fidelity

1. 시뮬레이션 환경은 Isaac Lab에 구현한다.
2. 학습은 오픈소스 DIY 모델을 단순화한 3D 모델링을 활용한다.
3. 성능 향상에 대한 척도는 전진 속도(Forward Velocity)로 한정한다.
4. 강화학습 하이퍼파라미터는 본 EUREKA 연구와 동일하게 설정한다.

---

### Solution Approach

#### 3D Modeling
선정한 오픈소스 DIY 모델을 시뮬레이션 환경에 맞게끔 곡선은 배제하고 나사같은 미세 부품들을 제거하는 방식으로 단순화 작업을 거쳤다. 총 무게에서 브림과 사용자 정의 부분을 제외하여 실제 무게를 계산.
<br>

<div style="text-align: center;">
  <img src='Picture/2D Model.png' width='40%' style='display:inline-block; margin-right:5px;'/>
  <img src='Picture/3D Model(origin).png' width='40%' style='display:inline-block;'/>
  
  <br/><br/>
  
  <img src='Picture/3D Model(Sim).png' width='50%' style='display:block; margin: 0 auto;'/>
</div>