from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder

from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA,KernelPCA,TruncatedSVD

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pickle

## 数据导入

## 数据处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=True, random_state=0) #划分训练测试集
less_cat_col = [col_name for col_name in X_train.columns if X_train[col_name].dtype=='object' and X_train[col_name].nunique()<10] #少类别型变量
more_cat_col = [col_name for col_name in X_train.columns if X_train[col_name].dtype=='object' and X_train[col_name].nunique()>=10] #多类别型变量
num_col = [col_name for col_name in X_train.columns if X_train[col_name].dtype in ['int64', 'float64']] #数值型特征
# print(less_cat_col, more_cat_col, num_col)

less_cat_transform = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('encoder', OneHotEncoder(handle_unknown='ignore'))]
                              ) #类别型变量先用众数填充再独热编码
more_cat_transform = Pipeline(steps = [('imputer', SimpleImputer(strategy='most_frequent')),
                                       ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]
                             ) #类别型变量先用众数填充再普通编码
num_transform = SimpleImputer(strategy='mean') #数值型变量采用均值填充
preprocessor = ColumnTransformer(transformers = [('less_cat', less_cat_transform, less_cat_col),
												('more_cat', more_cat_transform, more_cat_col),
												('num', num_transform, num_col)]
								) #不同的预处理步骤打包到一起
	
## 特征工程
combined= FeatureUnion(transformer_list = [('linear_pca',PCA(n_components = 3)),
											('kernel_pca',KernelPCA(n_components = 5)),
											("svd", TruncatedSVD(n_components=2))]
					  ) #集成了PCA、KernelPCA和TruncatedSVD的特征
combined_X = combined.fit_transform(X_train)


## 训练模型做交叉验证
model = GradientBoostingRegressor(random_state=0) # 
pipe = Pipeline(steps=[('preprocessing', preprocessor),
                       ('model', model)])
params = {
        'model__n_estimators':[100, 200, 300],
        'model__learning_rate':[0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7, 9,],
        'model__max_features':[5, 7, 11,  14],
        'model__min_samples_leaf': [1, 2, 3]
		}
gs = GridSearchCV(pipe, param_grid = params)
gs.fit(X_train, y_train)
print(gs.best_params_) # 最佳参数
y_pred = gs.best_estimator_.predict(X_test)
MAE = mean_absolute_error(y_test, y_pred) #平均绝对误差
score = gs.score(X_test, y_test)
print("平均绝对误差和得分", MAE, score)


## 保存模型为pickle文件（该序列化文件不仅保存了模型，而且还保存了对应的特征工程处理）
print(gs.named_steps['preprocessing']._feature_names_in)
print("mean_absolute_error: {}, and model score: {}".format(MAE, score))
with open(r'D:\项目\psg_melt_strategy.pickle', "wb") as model_file: #保存模型
        pickle.dump(gs, model_file)





					
