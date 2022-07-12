# importing modules

import pandas as pd
import time
from datetime import datetime
import pickle as pkl
import numpy as np
import logging

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

#modules MIP-EGO

from boruta_feature_selection import boruta_feature_selection
# from BayesOpt import BO
# from BayesOpt.Surrogate import RandomForest
# from BayesOpt.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace

#modules sklearn

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, auc, roc_curve


# Setting up the log file

suffix = time.strftime("%Y%m%d_%H%M%S")
logfile = './test_runs/EMG_log_'+str(suffix)

logger = logging.getLogger('EMG')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('- %(asctime)s [%(levelname)s] -- ''[- %(process)d - %(name)s] %(message)s')

if logfile is not None:
            fh = logging.FileHandler(logfile)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)


# Setting up the file for performance measure
# set suffix of filename:
description='_hcp'

# Name of data set
name_dataset = 'EMG'
file_name = str(suffix) +'_' + name_dataset + '_performance_'+description+'.txt'
file_name_inc = str(suffix) +'_' + name_dataset + '_performance_incremental'+description+'.txt'
folder_name = 'experiments'
f_performance = open(folder_name + '/' + file_name, 'w+')
f_performance_inc = open(folder_name + '/' + file_name_inc, 'w+')


# Number of random forest iterations and CV
start_time = time.time()

iterations_rf = 100               
cv = 5 # original 10.  Changed to 5 for testing.

logger.info('Parameters are: CV='+str(cv)+', random forest iterations='+str(iterations_rf))


# Loading data
logger.info('Loading data...')


with open('./EMG_EXTRACTED.pkl', 'rb') as f:
	X = pkl.load(f)

with open('./EMG_LABEL_01.pkl', 'rb') as f:
	Y = pkl.load(f)

X_boruta = X  # This is a subsitition so I do not
              # have to change X_boruta to X, below


# Initiation of metric scorer
acc_tr = []
acc_vld = []
f1_tr = []
f1_vld = []
precision_tr = []
precision_vld = []
recall_tr = []
recall_vld = []

f1_tr_1 = []
f1_vld_1 = []
precision_tr_1 = []
precision_vld_1 = []
recall_tr_1 = []
recall_vld_1 = []

roc_auc_score_tr = []
roc_auc_score_vld = []

TN_tr = []
FP_tr = []
FN_tr= []
TP_tr= []
TN_vld= []
FP_vld= []
FN_vld= []
TP_vld= []

AUC_tr = []
AUC_vld = []

FPR_tr = np.array([])
TPR_tr = np.array([])
THRES_tr = np.array([])

FPR_vld= np.array([])
TPR_vld= np.array([])
THRES_vld= np.array([])


params_per_split = []
boruta_features_per_split = []
feature_importance_per_rf = []


logger.info('Starting the cross-validation')

# Stratified split
skf= StratifiedKFold(n_splits=cv, random_state=np.random, shuffle=True)

counter = 0
cv_counter = 0

#two cross-validation procedures: 1 called outer (for validation) and  another called inner (for testing hyperparameter combinations).
for train_index, test_index in skf.split(X_boruta, Y['label']):
				
				cv_counter+=1
				logger.info(f'CV counter: {cv_counter}')
				logger.info('Started Boruta on the split')
				#print("Outer CV, TRAIN:", train_index, "TEST:", test_index)
				X_train, X_test = X_boruta.iloc[train_index], X_boruta.iloc[test_index]
				y_train, y_test = Y['label'].iloc[train_index], Y['label'].iloc[test_index]

				X_train, X_test, features_list = boruta_feature_selection(X_train,X_test,y_train)
				boruta_features_per_split.append(features_list)

				#np.random.seed(666)

				dim = 2
				n_step = 5 #reduced from 200 to 20 for testing
				n_init_sample = 10 * dim

				data = X_train
			
				target = y_train.values

				

				df_columns = ['acc' , 'max_depth' , 'n_estimators' , 'bootstrap' , 'max_features' , 'min_samples_leaf' , 'min_samples_split']#, 'class_weigth']

				df_eval = pd.DataFrame(columns = df_columns)

				#objective function
				def obj_func(x):
					
					#logger.info('Started internal cross-validation')
					global df_eval
					global df_columns
					global cv

					acc_ego_vld = []
					
					skf= StratifiedKFold(n_splits=cv, random_state=np.random, shuffle=True)
					for train_index, test_index in skf.split(data, target):
							#print("Inner CV: TRAIN:", train_index, "TEST:", test_index)
							X_train_BO, X_test_BO = data[train_index], data[test_index]
							y_train_BO, y_test_BO = target[train_index], target[test_index]
                                                        
													
							rf_boruta_inside = RandomForestClassifier(n_estimators=int(x[1]), max_depth=int(x[0]), bootstrap=x[2],
									max_features=x[3], min_samples_leaf=x[4], min_samples_split = x[5])
							
							rf_boruta_inside.fit(X_train_BO,y_train_BO)
							
							predictions_ego_vld = rf_boruta_inside.predict(X_test_BO)

							acc_ego_vld.append(accuracy_score(y_test_BO, predictions_ego_vld))

							val = np.mean(acc_ego_vld)

					df_eval_tmp = pd.DataFrame([[val, x[0], x[1], x[2], x[3], x[4], x[5]]], columns=df_columns)#, x[6]]], columns=df_columns)
					df_eval=df_eval.append(df_eval_tmp)
					return val

				#definition of hyperparameter search space:
				max_depth = OrdinalSpace([2, 100])
				n_estimators = OrdinalSpace([1, 1000])
				min_samples_leaf = OrdinalSpace([1, 10])
				min_samples_split = OrdinalSpace([2, 20])
				bootstrap = NominalSpace(['True', 'False'])
				max_features = NominalSpace(['auto', 'sqrt', 'log2'])
				class_weight = NominalSpace(['balanced', 'balanced_subsample'])

				search_space = max_depth + n_estimators + bootstrap + max_features + min_samples_leaf + min_samples_split
				model = RandomForest(levels=search_space.levels)
				
				logger.info('Started hyperparameter optimization using BO with n_step: '+str(n_step))
				opt = BO(search_space, obj_func, model, max_iter=n_step,
						n_init_sample=n_init_sample,
						n_point=1,
						n_job=1,
						minimize=False,
						verbose=False,
						optimizer='MIES')

				opt.run()

				#print('max acc: ', opt.f_max)
 
				best_params_ = df_eval[df_columns[1:]][df_eval['acc'] == df_eval['acc'].max()][:1].to_dict('records')
				params_per_split.append(best_params_)

				for i in range(0,iterations_rf):

						counter+=1
						logger.info('Counter is '+str(counter))
						#print('Counter is '+str(counter))
                                                
						rf = RandomForestClassifier(**best_params_[0])

						rf.fit(X_train,y_train)

						feature_importance_per_rf.append(rf.feature_importances_)

						predictions_tr = rf.predict(X_train)
						predictions_tr_proba = rf.predict_proba(X_train)


						predictions_vld = rf.predict(X_test)
						predictions_vld_proba = rf.predict_proba(X_test)

						tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(y_train, predictions_tr).ravel()
						tn_vld, fp_vld, fn_vld, tp_vld = confusion_matrix(y_test, predictions_vld).ravel()

						fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, predictions_tr_proba[:,1], pos_label=1, drop_intermediate=False)
						fpr_vld, tpr_vld, thresholds_vld = roc_curve(y_test, predictions_vld_proba[:,1], pos_label=1, drop_intermediate=False)

						acc_tr.append(accuracy_score(y_train, predictions_tr))
						acc_vld.append(accuracy_score(y_test, predictions_vld))
						f1_tr.append(f1_score(y_train, predictions_tr, average='macro'))
						f1_vld.append(f1_score(y_test, predictions_vld, average='macro'))
						precision_tr.append(precision_score(y_train, predictions_tr, average='macro'))
						precision_vld.append(precision_score(y_test, predictions_vld, average='macro'))
						recall_tr.append(recall_score(y_train, predictions_tr, average='macro'))
						recall_vld.append(recall_score(y_test, predictions_vld, average='macro'))

						f1_tr_1.append(f1_score(y_train, predictions_tr, average='weighted'))
						f1_vld_1.append(f1_score(y_test, predictions_vld, average='weighted'))
						precision_tr_1.append(precision_score(y_train, predictions_tr, average='weighted'))
						precision_vld_1.append(precision_score(y_test, predictions_vld, average='weighted'))
						recall_tr_1.append(recall_score(y_train, predictions_tr, average='weighted'))
						recall_vld_1.append(recall_score(y_test, predictions_vld, average='weighted'))

						roc_auc_score_tr.append(roc_auc_score(y_train, predictions_tr_proba[:,1]))
						roc_auc_score_vld.append(roc_auc_score(y_test, predictions_vld_proba[:,1]))

						AUC_tr.append(auc(fpr_tr,tpr_tr))
						AUC_vld.append(auc(fpr_vld,tpr_vld))

						TN_tr.append(tn_tr)
						FP_tr.append(fp_tr)
						FN_tr.append(fn_tr)
						TP_tr.append(tp_tr)
						TN_vld.append(tn_vld)
						FP_vld.append(fp_vld)
						FN_vld.append(fn_vld)
						TP_vld.append(tp_vld)

						FPR_tr =np.append(FPR_tr,fpr_tr)
						TPR_tr =np.append(TPR_tr,tpr_tr)
						THRES_tr =np.append(THRES_tr,thresholds_tr)

						FPR_vld =np.append(FPR_vld,fpr_vld)
						TPR_vld =np.append(TPR_vld,tpr_vld)
						THRES_vld =np.append(THRES_vld,thresholds_vld)

				
						
						logger.info("\n--- Incremental Evaluation metrics, EGO with " + str(n_step) + " steps." + " ---\n")
						logger.info('acc train score '  + ': '+str(acc_tr[-1])+ '\n')
						logger.info('acc val score '  + ': '+ str(acc_vld[-1])+ '\n')
						logger.info('f1 train score (macro) '  + ': '+ str(f1_tr[-1])+ '\n')
						logger.info('f1 val score (macro) '  + ': '+ str(f1_vld[-1])+ '\n')
						logger.info('precision train score (macro) '  + ': '+ str(precision_tr[-1])+ '\n')
						logger.info('precision val score (macro) ' + ': '+ str(precision_vld[-1])+ '\n')
						logger.info('recall train score (macro) '  + ': '+ str(recall_tr[-1])+ '\n')
						logger.info('recall val score (macro) '  + ': '+ str(recall_vld[-1])+ '\n')

						logger.info('f1 train score (weighted) '  + ': '+ str(f1_tr_1[-1])+ '\n')
						logger.info('f1 val score (weighted) '  + ': '+ str(f1_vld_1[-1])+ '\n')
						logger.info('precision train score (weighted) '  + ': '+ str(precision_tr_1[-1])+ '\n')
						logger.info('precision val score (weighted) ' + ': '+ str(precision_vld_1[-1])+ '\n')
						logger.info('recall train score (weighted) '  + ': '+ str(recall_tr_1[-1])+ '\n')
						logger.info('recall val score (weighted) '  + ': '+ str(recall_vld_1[-1])+ '\n')

							
						logger.info('roc auc train score ' + ': '+ str(roc_auc_score_tr[-1])+ '\n')
						logger.info('roc auc val score '  + ': '+ str(roc_auc_score_vld[-1])+ '\n')

						logger.info('True Negative (train): ' + str(TN_tr[-1])+ '\n')
						logger.info('False Positive (train): ' + str(FP_tr[-1])+ '\n')
						logger.info('False Negative (train): ' + str(FN_tr[-1])+ '\n')
						logger.info('True Positive (train): ' + str(TP_tr[-1])+ '\n')

						logger.info('True Negative (test): ' + str(TN_vld[-1])+ '\n')
						logger.info('False Positive (test): ' + str(FP_vld[-1])+ '\n')
						logger.info('False Negative (test): ' + str(FN_vld[-1])+ '\n')
						logger.info('True Positive (test): ' + str(TP_vld[-1])+ '\n')

						logger.info('Sensitivity (train): ' + str(TP_tr[-1] / (TP_tr[-1]+FN_tr[-1]))+ '\n')
						logger.info('Sensitivity (test): ' + str(TP_vld[-1] / (TP_vld[-1]+FN_vld[-1]))+ '\n')

						logger.info('Specificity (train): ' + str(TN_tr[-1] / (TN_tr[-1]+FP_tr[-1]))+ '\n')
						logger.info('Specificity (test): ' + str(TN_vld[-1] / (TN_vld[-1]+FP_vld[-1]))+ '\n')

						logger.info('AUC (train): ' + str(AUC_tr[-1])+ '\n')
						logger.info('AUC (test): ' + str(AUC_vld[-1])+ '\n')

						f_performance_inc.write("\n--- Incremental Evaluation metrics, EGO with " + str(n_step) + " steps." + " ---\n")
						f_performance_inc.write('acc train score '  + ': '+str(acc_tr[-1])+ '\n')
						f_performance_inc.write('acc val score '  + ': '+ str(acc_vld[-1])+ '\n')