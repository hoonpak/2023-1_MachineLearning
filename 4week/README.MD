# 과제 4-1 재배환경 별 작물 종류 예측

이상 기후, 온난화등의 문제로 기상 변화 문제로 농업의 생태계는 다양한 변화를 맞이하고 있습니다. 기후를 통제하여 안정적인 생산력을 유지하는 스마트팜과 더불어 변화하는 환경에 맞추어 적절한 작물을 찾는 시도도 필요하며, 본 과제를 통해 이러한 요구를 해결하고자 합니다. 데이터는 다음의 정보를 포함합니다.

* N - 토양 내 질소 함량 비율
* P - 토양 내 인산 함량 비율
* K - 토양 내 칼륨 함량 비율
* temperature - 섭씨 온도
* humidity - 상대 습도(%)
* ph - 토양의 ph 값
* rainfall - 강우량 (mm)

주어진 환경정보를 통해 환경에 적합한 작물을 예측해봅시다.

sklearn에서 제공하는 StandardScaler 함수를 통해 데이터 정규화를 진행할 수 있습니다.

    from sklearn.preprocessing import StandardScaler

    std=StandardScaler()

    train_feature_std=std.fit_transform(train_feature)
    test_feature_std=std.transform(test_feature)
---
# 과제 4-2 자동차 가격 예측

해당 데이터 셋은 중고 차량에 대한 가격, 변속기, 마일리지, 연료 유형, 도로세, 갤런당 마일리지(mpg), 제조회사 및 엔진 크기 등을 확인할 수 있습니다.
위에 언급한 차량의 정보들(제조회사, 엔진크기, 변속기 등)을 통하여 해당 차량의 가격을 예측하는 것이 해당 실습의 목표입니다.

대부분의 항목들이 데이터 실수화가 되어있지만, 일부 항목은 데이터 실수화가 되어있지 않습니다.

sklearn에서 제공하는 LabelEncoder 함수를 통해 데이터 실수화를 진행할 수 있습니다.

sklearn에서 제공하는 StandardScaler 함수를 통해 데이터 정규화를 진행할 수 있습니다.

    from sklearn.preprocessing import StandardScaler

    std=StandardScaler()

    train_feature_std=std.fit_transform(train_feature)
    test_feature_std=std.transform(test_feature)