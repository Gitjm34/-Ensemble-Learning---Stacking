# -Ensemble-Learning---Stacking

### **1. 앙상블 학습이란?**

**앙상블 학습이란 여러 개의 분류기를 생성하고 각 예측들을 결합함으로써 보다 정확한 예측을 도출하는 기법**입니다.

정형 데이터의 예측 분석 영역에서 앙상블이 매우 높은 예측 성능으로 인해 많은 분석가와 데이터 사이언티스트에게 애용되고 있습니다.

앙상블의 기본 알고리즘으로 일반적으로 사용되는 것은 결정 트리이고 앙상블 방식으로는 크게 보팅, 배깅, 부스팅으로 나뉩니다.

근래에는 앙상블 방식이 부스팅 방식으로 발전하고 잇습니다.

### **2. 보팅**

**보팅은 서로 다른 알고리즘으로 예측하고 예측한 결과를 가지고 투표하듯 보팅을 통해 최종 예측 결과를 선정하는 방식**입니다.

보팅은 하드 보팅과 소프트 보팅으로 나뉩니다.

![](https://blog.kakaocdn.net/dn/UbKNx/btraBSLhfCn/CQjgjqSHGKkxR9Ph9A5aRK/img.png)

출처 : https://jaaamj.tistory.com/33?category=906294

**① 하드 보팅**

하드 보팅은 **다수결 원칙**과 비슷합니다. 여러 개의 예측 결과 중 다수의 분류기가 결정한 예측값을 최종 보팅 결괏값으로 선정합니다.

**② 소프트 보팅**

소프트 보팅은 각 예측 결과의 레이블 값 결정 **확률들의 평균값이 높은 레이블 값을 최종 보팅 결괏값**으로 선정합니다.

일반적으로 소프트 보팅이 보팅 방법으로 적용됩니다.

로지스틱 회귀와 KNN을 기반으로 사이킷런 내장 데이터 세트인 위스콘신 유방암 데이터 보팅 분류기를 만들어 보겠습니다.

```python
import pandas as pd

# 모델 불러오기from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# 데이터셋과 데이터나누는 모듈, 정확도 평가 모듈 불러오기from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer_data=load_breast_cancer()

lr_clf=LogisticRegression()
knn_clf=KNeighborsClassifier(n_neighbors=8)
vo_clf=VotingClassifier(estimators=[('LR',lr_clf),('KNN',knn_clf)],voting='soft')

X_train, X_test, y_train, y_test = train_test_split(cancer_data.data,cancer_data.target,test_size=0.2,random_state=100)

vo_clf.fit(X_train,y_train)
pred=vo_clf.predict(X_test)
print('보팅 분류기 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

classifiers=[lr_clf,knn_clf]
for clf in classifiers:
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    class_name=clf.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name,accuracy_score(y_test,pred)))
```

![](https://blog.kakaocdn.net/dn/r13L2/btrav7bFSQu/KNmhJ55N7NzQxRo2V3WUTk/img.png)

**개별 분류기의 성능보다 보팅 분류기의 성능이 우수한 것을 확인**할 수 있습니다.

하지만 데이터 셋을 나눌 때 random_state를 바꿔보면 오히려 개별 분류기가 보팅 분류기보다 성능이 우수한 경우가 있음을 확인할 수 있습니다.

그래도 앙상블 방법이 전반적으로 다른 단일 머신러닝 알고리즘보다 뛰어난 예측 성능을 가지는 경우가 많습니다.

현실에서도 문제를 풀 때 혼자 보다는 많은 사람들과 같이 하는 것이 더 좋은 성과를 내는 것처럼 말입니다.

### **3. 배깅**

배깅은 같은 알고리즘으로 **여러 개의 분류기를 만들어서 보팅으로 최종 결정하는 알고리즘**입니다.

배깅의 대표적인 알고리즘은 랜덤 포레스트입니다.

### **3-1. 랜덤 포레스트**

랜덤 포레스트는 앙상블 알고리즘 중 비교적 빠른 수행 속도를 가지고 있으며 다양한 영역에서 높은 예측 성능을 보입니다.

랜덤 포레스트는 **여러 개의 결정 트리 분류기가 전체 학습 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해 개별적으로 학습을 수행한 뒤 최종적으로 모든 분류기가 보팅을 통해 예측 결정**을 합니다.

![](https://blog.kakaocdn.net/dn/brvKWi/btrav6KEpto/jLG48GsfPaUotdvoIsXsSk/img.png)

학습 데이터를 여러 개의 데이터 세트로 중첩되게 분리하는 것을 부트스트래핑 (bootstrapping) 이라고 합니다.

(bagging은 bootstrap aggregation의 줄임말입니다.)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(cancer_data.data,cancer_data.target,test_size=0.2,random_state=100)
rf_clf=RandomForestClassifier(random_state=42)
rf_clf.fit(X_train,y_train)
pred=rf_clf.predict(X_test)
print('랜덤 포레스트 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))
```

![](https://blog.kakaocdn.net/dn/bstie6/btrazLsoPRc/CvK7R41ayNZtvU061SaY2K/img.png)

같은 유방암 데이터를 랜덤 포레스트로 분류한 결과 0.9649의 정확도가 나왔습니다.

위의 보팅 분류기보다 좀 더 좋은성능을 보였습니다.

**트리 기반의 앙상블 알고리즘의 단점**은 하이퍼 파라미터가 너무 많고, 그로 인해 튜닝을 위한 시간이 많이 소모된다는 점입니다.

트리 기반 자체의 하이퍼 파라미터가 원래 많은 데다 배깅, 부스팅, 학습, 정규화 등을 위한 하이퍼 파라미터까지 추가되므로 일반적으로 다른 머신러닝 알고리즘에 비해 많을 수밖에 없습니다.

**랜덤 포레스트의 하이퍼 파라미터에 대해 알아보겠습니다.**

① n_estimators : 랜덤 포레스트에서 결정 트리의 개수를 지정합니다. 디폴트 값은 10입니다.

② max_features : 결정 트리에 사용된 max_features 파라미터와 같습니다. 디폴트 값은 'sqrt' 입니다.

③ 결정 트리와 동일하게 과적합 개선을 위해서 max_depth와 min_samples_leaf가 사용이 됩니다.

랜덤 포레스트도 GridSearchCV를 이용해 최적의 하이퍼 파라미터 값을 찾을 수 있습니다.

### **4. 부스팅**

부스팅 알고리즘은 **여러 개의 약한 학습기를 순차적으로 학습 - 예측 하면서 잘못 예측한 데이터에 가중치를 부여해서 오류를 개선해 나가면서 점진적으로 학습하는 방식**입니다.

부스팅은 크게 AdaBoost와 GBM이 있습니다.

XGBoost와 LightGBM은 GBM을 더욱 발전시킨 알고리즘입니다.

### **4-1. AdaBoost**

에이다 부스트가 어떻게 학습을 진행하는지 알아보겠습니다.

![](https://blog.kakaocdn.net/dn/B3yAK/btrax6XTfVb/Rk3vfZyCKTTRgNQFK3s3x0/img.png)

Step 1 : 첫 번째 약한 학습기가 파란색과 빨간색을 분류합니다. 아래의 빨간색과 위의 파란색이 오분류된 것입니다.

Step 2 : 오분류된 데이터에 대해 가중치 값을 부여합니다. 이제 다음 약한 학습기가 더 잘 분류할 수 있게 되었습니다.

Step 3 : 두 번째 약한 학습기가 파란색과 빨간색을 분류합니다. 왼쪽의 파란색이 오분류된 것입니다.

Step 4 : 오분류된 데이터에 대해 가중치 값을 부여합니다.

Step 5 : 세 번째 약한 학습기가 파란색과 빨간색을 분류합니다. 오른쪽의 빨간 데이터가 오분류된 것입니다.

Step 6 : 마지막으로 세 개의 약한 학습기를 모두 결합한 예측 결과입니다.

마지막 예측 결과를 보면 개별 약한 학습기보다 훨씬 정확도가 높아진 것을 알 수 있습니다.

### **4-2. GBM (Gradient Boosting Machine)**

GBM은 에이다부스트와 비슷하나 **가중치 업데이트를 경사 하강법을 이용하는 것**이 큰 차이입니다.

오류 값은 (실제 값 - 예측 값)이고 이 값을 최소화하는 방향성을 가지고 반복적으로 가중치 값을 업데이트하는 것이 경사 하강법입니다.

사이킷런에서는 GBM 기반의 분류를 위해 GradientBoostingClassifier 클래스를 제공합니다.

GBM은 예측 성능은 뛰어나나 수행 시간이 오래 걸린다는 단점이 있고 하이퍼 파라미터 튜닝 노력도 더 필요합니다.

**GBM의 하이퍼 파라미터에 대해 알아보겠습니다.**

① loss : 경사 하강법에서 사용할 비용 함수를 지정합니다.

② learning_rate : 경사 하강법에서 학습률을 의미합니다. (선형 회귀 포스팅에 자세히 설명되어 있습니다)

③ n_estimators : 약한 학습기의 개수입니다.

④ subsample : 약한 학습기가 학습에 사용하는 데이터의 샘플링 비율입니다. 기본값은 1 (전체 데이터)입니다.

GBM을 이용해서 유방암 데이터를 분류해 정확도를 평가해보고 GridSearchCV를 통해 하이퍼 파라미터를 튜닝한 후 정확도를 평가해보겠습니다.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(cancer_data.data,cancer_data.target,test_size=0.2,random_state=100)

params={
    'n_estimators':[100,500],
    'learning_rate':[0.05,0.1]
}

gb_clf=GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train,y_train)
pred=gb_clf.predict(X_test)
print('GBM 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))
grid_cv=GridSearchCV(gb_clf,param_grid=params,cv=2,verbose=1)
grid_cv.fit(X_train,y_train)
gb_grid_pred=grid_cv.best_estimator_.predict(X_test)
print('하이퍼 파라미터 튜닝 후 정확도:{0:.4f}'.format(accuracy_score(y_test,gb_grid_pred)))
```

![](https://blog.kakaocdn.net/dn/b9VErV/btraBd95l3H/HI8Efh9nQWUdhuN80oGJkk/img.png)

GridSearchCV를 통해 하이퍼 파라미터를 튜닝하니 GBM의 정확도가 높아진 것을 볼 수 있습니다.

### **4-3. XGBoost (eXtra Gradient Boost)**

XGBoost는 트리 기반의 앙상블 학습에서 가장 각광받고 있는 알고리즘 중 하나입니다.

**XGBoost는 GBM에 기반하고 있지만 GBM의 단점인 느린 수행 시간 및 과적합 규제 부재 등의 문제를 해결해서 인기가 많습니다.**

이렇게 각광받고 있는 XGBoost의 **장점**들을 살펴보겠습니다.

① 뛰어난 예측 성능 : 일반적으로 분류와 회귀 영역에서 예측 성능을 발휘합니다.

② GBM보다 빠른 수행 시간 : 병렬 수행 및 다양한 기능으로 GBM에 비해 빠른 수행 성능으로 보장합니다.

③ 과적합 규제 : 과적합에 강한 내구성을 가질 수 있습니다.

④ 나무 가지치기 : 더 이상 긍정 이득이 없는 분할을 가지치기해서 분할 수를 더 줄이는 장점을 가지고 있습니다.

⑤ 내장된 교차 검증 : 내부적으로 교차 검증을 수행하여 최적화된 반복 수행 횟수를 가질 수 있습니다.

⑥ 결손값 자체 처리 : 결손값을 자체적으로 처리할 수 있는 기능을 가지고 있습니다.

또한 XGBoost는 기본 GBM에서 부족한 다른 여러 가지 성능 향상 기능을 가지고 있습니다.

그중에 수행 속도를 향상시키기 위한 대표적인 기능으로 **조기 중단**이 있습니다.

부스팅 반복 횟수에 도달하지 않더라도 예측 오류가 더 이상 개선되지 않으면 반복을 끝까지 수행하지 않고 중지해 수행 시간을 개선할 수 있습니다.

예를 들어, n_estimators를 200으로 설정하고 조기 중단 파라미터 값을 50으로 설정하면, 100회에서 학습 오류 값보다 작은 값이 101~150회에 하나도 없으면 부스팅은 종료됩니다.

하지만 조기 중단 값을 너무 낮게 설정하면 학습 오류 값이 낮아질 가능성이 있는데도 학습을 종료하게 되어 성능이 더 떨어질 수 있습니다.

사이킷런 래퍼 XGBoost는 분류를 위한  래퍼 클래스 XGBClassifier, 회귀를 위한 래퍼 클래스 XGBRegressor가 있습니다.

유방암 데이터 세트를 XGBClassifier를 이용하여 예측해 보겠습니다.

```python
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

xgb_wrapper=XGBClassifier(n_estimators=700, learning_rate=0.1, max_depth=3)
xgb_wrapper.fit(X_train,y_train)
pred=xgb_wrapper.predict(X_test)
print('XGBoost 정확도:{0:.4f}'.format(accuracy_score(y_test,pred)))
```

![](https://blog.kakaocdn.net/dn/ccJMXb/btrasT5xqKH/czBlpFFviepGBq86Ly4fe1/img.png)

GBM보다 나은 성능을 보이는 것을 볼 수 있습니다.

### **4-4. LightGBM**

LightGBM은 XGBoost와 함께 부스팅 계열 알고리즘에서 가장 각광을 받고 있습니다.

LightGBM의 XGBoost 대비 **장점**은 다음과 같습니다.

① 더 빠른 학습과 예측 수행 시간

② 더 작은 메모리 사용량

③ 카테고리형 피처의 자동 변환과 최적 분할 (원-핫 인코딩을 사용하지 않고도 최적으로 변환, 분할 수행)

**단점**으로는 적은 데이터 세트에 적용할 경우 과적합 가능성이 크다는 점이 있습니다.

LightGBM은 일반 GBM 계열의 트리 분할 방법과 다르게 리프 중심 트리 분할 방식을 사용합니다.

균형 잡힌 트리가 아닌 최대 손실값을 가지는 리프 노드를 지속적으로 분할하면서 트리의 깊이가 깊어지고 비대칭적인 규칙 트리가 생성됩니다.

이렇게 **최대 손실값을 갖는 리프 노드를 지속적으로 분할하는 것이 균형 트리 분할 방식보다 예측 오류 손실을 최소화 할 수 있다는 것**이 LightGBM의 구현 사상입니다.

![](https://blog.kakaocdn.net/dn/cwWqHs/btrazNcYKYs/PuLB9Rpa1NMA7bQsM85us1/img.jpg)

사이킷런 래퍼 LightGBM 하이퍼 파라미터와 사이킷런 래퍼 XGBoost 하이퍼 파라미터를 정리해 보겠습니다.

![](https://blog.kakaocdn.net/dn/yorby/btraEfzvbrh/hbKjxXWu4SjslM3YzksHe0/img.png)

LightGBM을 이용해서 유방암 데이터 세트를 분류 예측해보겠습니다.

```python
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

cancer_data=load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer_data.data,cancer_data.target,test_size=0.2,random_state=100)
lgbm_wrapper=LGBMClassifier(n_estimators=700)
evals=[(X_test,y_test)]
lgbm_wrapper.fit(X_train,y_train,early_stopping_rounds=100,eval_metric="logloss",
                               eval_set=evals,verbose=True)
pred=lgbm_wrapper.predict(X_test)
print('LightGBM 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))
```

![](https://blog.kakaocdn.net/dn/bWaPq2/btraCSEAX9W/fwr3TQbglplrwGAMEPpank/img.png)

결과를 보면 51회 이후에 100회 동안 학습 오류 값이 줄어들지 않았으므로 조기 중단한 것을 볼 수 있습니다.

학습 데이터 세트와 테스트 데이터 세트의 크기가 작아서 XGBoost와 성능 비교 차이는 큰 의미가 없습니다.

이렇게 앙상블 학습 방법의 보팅, 배깅, 부스팅에 대해 알아보았습니다.

앞으로 분류 문제에는 성능이 뛰어난 앙상블 기법들을 이용해야겠습니다 :)
