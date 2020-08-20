# Import packages
# other packages
from datetime import datetime, timedelta
import json
import os
import pandas as pd
from sqlalchemy import create_engine
# airflow related
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
# local imports
from config import CRED_FILE_PATH, TARGET_SQL, DATA_PATH

# initiate the default dag arg
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020, 9, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'schedule_interval': '@daily',
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

# define the functions needed for following tasks


def get_cred(default_path=CRED_FILE_PATH):
    cred_path = default_path
    with open(cred_path) as json_file:
        creds = json.load(json_file)
    return creds


def get_target_training_holdout(path_training, path_holdout, engine):
    df_raw = pd.read_sql(TARGET_SQL, engine)
    df_training = df_raw[df_raw.statement_num == 3].copy()
    df_holdout = df_raw[(df_raw.statement_num == 2) & (
        ~df_raw.acct_id.isin(df_training.acct_id))].copy()
    print(df_raw.shape, df_training.shape, df_holdout.shape)
    df_training.to_csv(path_training, index=False)
    df_holdout.to_csv(path_holdout, index=False)

    return df_training, df_holdout


def qa_target(df_training, df_holdout):
    # change holdout is not in training, vice versa
    acct_id_traing = set(df_training.acct_id.values)
    acct_id_holdout = set(df_holdout.acct_id.values)
    c = acct_id_traing.intersection(acct_id_holdout)
    assert len(c) == 0
    assert len(acct_id_traing) > 0
    assert len(acct_id_holdout) > 0


def get_bdc_master_dataset_w_output(output_path):
    df_master_bdc = pd.read_csv(os.path.join(DATA_PATH, 'bdc_master.csv'),
                                usecols=['id', 'user_id', 'appl_id', 'created_at']).rename(columns={'id': 'master_id'})
    for filename in ['', 'all_user_level_feature_acct.csv', 'all_user_level_feature_address.csv',
                     'all_user_level_feature_inquiry.csv']:
        csv_path = os.path.join(DATA_PATH, filename)
        _df = pd.read_csv(csv_path)
        df_master_bdc = df_master_bdc.merge(_df, on='master_id', how='left')
        print(df_master_bdc.shape, "table {} merged".format(
            filename.split('.')[0].split('_')[-1]))
    df_master_bdc.to_csv(output_path, index=False)
    print("master bdc dataset created with shape of {}".format(df_master_bdc.shape))
    return df_master_bdc


def get_bdc_master_dataset():
    df_master_bdc = pd.read_csv(os.path.join(DATA_PATH, 'bdc_master.csv'),
                                usecols=['id', 'user_id', 'appl_id', 'created_at']).rename(columns={'id': 'master_id'})
    for filename in ['all_user_level_feature_acct.csv', 'all_user_level_feature_address.csv',
                     'all_user_level_feature_inquiry.csv']:
        csv_path = os.path.join(DATA_PATH, filename)
        _df = pd.read_csv(csv_path)
        df_master_bdc = df_master_bdc.merge(_df, on='master_id', how='left')
        print(df_master_bdc.shape, "table {} merged".format(
            filename.split('.')[0].split('_')[-1]))
    # df_master.to_csv(output_path, index=False)
    print("master bdc dataset created with shape of {}".format(df_master_bdc.shape))
    return df_master_bdc


def get_save_app_data(path_save_file, engine):
    df_app = pd.read_sql(APP_SQL, engine)
    col_keep = ['appl_id', 'master_id', 'policy_dcsn_note', 'valid_score_value', 'invalid_score_value', 'job_position',
                'job_industry', 'appl_submit_tms']
    df_app_w_feature = create_derive_features(df_app, col_keep)
    df_app_w_feature.to_csv(path_save_file, index=False)
    return df_app_w_feature


def _education_rank(education_string):
    if education_string == 'undergraduate':
        return 1
    elif education_string == 'undergraduate_incomplete':
        return 2
    elif education_string == 'highschool':
        return 3
    elif education_string == 'basic':
        return 4
    elif education_string == 'no_studies':
        return 5
    else:
        return 9


def create_derive_features(df, col_kept_list):
    df_derive = df.copy()
    source_col = ['birth_date', 'appl_submit_tms', 'appl_start_tms', 'education', 'primary_bank', 'marital_status',
                  'job_industry', 'job_position', 'incm_month', 'expense_month', 'limit_requested', 'email',
                  'first_name_std', 'last_name1_std',
                  'last_name2_std', 'edit_profile_flag', 'gender', 'contact1_relation', 'contact2_relation',
                  'edit_profile_flag', 'ine_submit_tms',
                  'prsn_info_submit_tms', 'addr_submit_tms', 'contact_submit_tms', 'fin_submit_tms']
    for col in source_col:
        assert col in df.columns.values, "{} column is not in the source dataframe".format(col)
    interim_col = ['email_domain', 'email_suffix', 'full_name']
    remove_col = [col for col in set(df_derive.columns.tolist() + interim_col) if col not in col_kept_list]
    df_derive['birth_date'] = pd.to_datetime(df_derive['birth_date'], format='%Y-%M-%d')
    df_derive['age_in_year'] = ((df_derive.appl_start_tms - df_derive.birth_date) / np.timedelta64(1, 'Y')).apply(
        lambda x: '{:.1f}'.format(x)).astype('float')
    df_derive.loc[df_derive.age_in_year == 'nan', ['age_in_year']] = -1

    df_derive['education_level'] = df_derive.education.apply(lambda x: _education_rank(x))
    df_derive['bbva_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'BBVA' else 0)
    df_derive['banamex_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'BANAMEX' else 0)
    df_derive['santander_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'SANTANDER' else 0)
    df_derive['hsbc_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'HSBC' else 0)
    df_derive['banorte_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'BANORTE' else 0)
    df_derive['azteca_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'AZTECA' else 0)
    df_derive['bancoppel_flag'] = df_derive.primary_bank.apply(lambda x: 1 if x == 'BANCOPPEL' else 0)

    df_derive['single_flag'] = df_derive.marital_status.apply(lambda x: 1 if x == 'single' else 0)
    df_derive['married_flag'] = df_derive.marital_status.apply(lambda x: 1 if x == 'married' else 0)
    df_derive['cohabiting_flag'] = df_derive.marital_status.apply(lambda x: 1 if x == 'cohabiting' else 0)
    df_derive['divorced_flag'] = df_derive.marital_status.apply(lambda x: 1 if x == 'divorced' else 0)
    df_derive['widowed_flag'] = df_derive.marital_status.apply(lambda x: 1 if x == 'widowed' else 0)

    df_derive['high_risk_job_industry_flag'] = df_derive.job_industry.apply(
        lambda x: 1 if x in ('transport', 'cleaning', 'construction', 'security') else 0)
    df_derive['low_risk_job_industry_flag'] = df_derive.job_industry.apply(
        lambda x: 1 if x in ('IT', 'finance', 'lawyer') else 0)

    df_derive['high_risk_job_position_flag'] = df_derive.job_position.apply(
        lambda x: 1 if x in ('unemployed', 'home', 'employee_informal') else 0)
    df_derive['low_risk_job_position_flag'] = df_derive.job_position.apply(
        lambda x: 1 if x in ('student', 'military') else 0)
    df_derive['job_position_owner_flag'] = df_derive.job_position.apply(lambda x: 1 if x == 'owner' else 0)
    df_derive['job_position_employee_gov_flag'] = df_derive.job_position.apply(
        lambda x: 1 if x == 'employee_gov' else 0)
    df_derive['job_position_retired_flag'] = df_derive.job_position.apply(lambda x: 1 if x == 'retired' else 0)
    df_derive['job_position_freelancer_flag'] = df_derive.job_position.apply(lambda x: 1 if x == 'freelancer' else 0)
    df_derive['job_position_employee_private_flag'] = df_derive.job_position.apply(
        lambda x: 1 if x == 'employee_private' else 0)

    df_derive['user_reported_month_income'] = df_derive['incm_month']
    df_derive['user_reported_month_expense'] = df_derive['expense_month']
    df_derive['user_expense_income_ratio'] = (
                                                     df_derive.user_reported_month_expense / df_derive.user_reported_month_income).round(
        4) * 100
    # handle divide by zero
    df_derive.loc[df_derive.user_reported_month_income == 0, 'user_expense_income_ratio'] = -1
    df_derive['user_requested_amount'] = df_derive['limit_requested']
    df_derive['user_requested_amount_income_ratio'] = (
                                                              df_derive.user_requested_amount / df_derive.user_reported_month_income).round(
        4) * 100
    df_derive.loc[df_derive.user_reported_month_income == 0, 'user_requested_amount_income_ratio'] = -1
    df_derive['email_domain'] = df_derive['email'].str.split('@').str[1].str.split('.').str[0]
    df_derive['user_email_gmail_flag'] = df_derive.email_domain.apply(lambda x: 1 if x == 'gmail' else 0)
    df_derive['user_email_hotmail_flag'] = df_derive.email_domain.apply(lambda x: 1 if x == 'hotmail' else 0)
    df_derive['user_email_outlook_flag'] = df_derive.email_domain.apply(lambda x: 1 if x == 'outlook' else 0)
    df_derive['user_email_yahoo_flag'] = df_derive.email_domain.apply(lambda x: 1 if x == 'yahoo' else 0)
    df_derive['user_email_live_flag'] = df_derive.email_domain.apply(lambda x: 1 if x == 'live' else 0)
    df_derive['user_email_icloud_flag'] = df_derive.email_domain.apply(lambda x: 1 if x == 'icloud' else 0)

    df_derive['email_suffix'] = df_derive['email'].str.split('@').str[1].str.split('.').str[1]
    df_derive['user_email_suffix_edu_flag'] = df_derive.email_suffix.apply(lambda x: 1 if x == 'edu' else 0)
    df_derive['user_email_suffix_net_flag'] = df_derive.email_suffix.apply(lambda x: 1 if x == 'net' else 0)
    df_derive['user_email_suffix_mx_flag'] = df_derive.email_suffix.apply(lambda x: 1 if x == 'mx' else 0)
    df_derive['user_email_suffix_gob_flag'] = df_derive.email_suffix.apply(lambda x: 1 if x == 'gob' else 0)
    df_derive['user_email_suffix_unam_flag'] = df_derive.email_suffix.apply(lambda x: 1 if x == 'unam' else 0)

    df_derive['full_name'] = df_derive.apply(
        lambda x: '%s %s %s' % (x['first_name_std'], x['last_name1_std'], x['last_name2_std']), axis=1).str.split(' ')
    df_derive['user_single_last_name'] = df_derive.apply(lambda x: 1 if (isinstance(x['last_name1_std'], float)
                                                                         or x['last_name1_std'] is None
                                                                         or len(x['last_name1_std']) == 1
                                                                         or isinstance(x['last_name2_std'], float)
                                                                         or x['last_name2_std'] is None
                                                                         or len(x['last_name2_std']) == 1) else 0,
                                                         axis=1)
    df_derive['user_total_words_in_full_name'] = df_derive.full_name.apply(lambda x: len(x))
    df_derive['user_female_flag'] = df_derive.gender.apply(lambda x: 1 if x == 'F' else 0)
    df_derive['user_contact_relation_friend_flag'] = df_derive.apply(lambda x: 1 if (x['contact1_relation'] == 'friend'
                                                                                     or x[
                                                                                         'contact2_relation'] == 'friend') else 0,
                                                                     axis=1)
    df_derive['user_contact_relation_partner_flag'] = df_derive.apply(
        lambda x: 1 if (x['contact1_relation'] == 'partner'
                        or x['contact2_relation'] == 'partner') else 0, axis=1)
    df_derive['user_contact_relation_sibling_flag'] = df_derive.apply(
        lambda x: 1 if (x['contact1_relation'] == 'sibling'
                        or x['contact2_relation'] == 'sibling') else 0, axis=1)
    df_derive['user_contact_relation_mother_flag'] = df_derive.apply(lambda x: 1 if (x['contact1_relation'] == 'mother'
                                                                                     or x[
                                                                                         'contact2_relation'] == 'mother') else 0,
                                                                     axis=1)
    df_derive['user_contact_relation_other_family_flag'] = df_derive.apply(
        lambda x: 1 if (x['contact1_relation'] == 'other_family'
                        or x['contact2_relation'] == 'other_family') else 0, axis=1)
    df_derive['user_contact_relation_father_flag'] = df_derive.apply(lambda x: 1 if (x['contact1_relation'] == 'father'
                                                                                     or x[
                                                                                         'contact2_relation'] == 'father') else 0,
                                                                     axis=1)
    df_derive['user_contact_relation_son_flag'] = df_derive.apply(lambda x: 1 if (x['contact1_relation'] == 'son'
                                                                                  or x[
                                                                                      'contact2_relation'] == 'son') else 0,
                                                                  axis=1)

    df_derive['user_edit_profile_flag'] = df_derive.edit_profile_flag.apply(lambda x: 1 if x == 1.0 else 0)
    df_derive['total_time_spent_on_application'] = (
            (df_derive.appl_submit_tms - df_derive.appl_start_tms) / np.timedelta64(1, 's')).apply(
        lambda x: '{:.2f}'.format(x)).astype('float')
    df_derive['time_spent_on_submit_ine'] = (
            (df_derive.ine_submit_tms - df_derive.appl_start_tms) / np.timedelta64(1, 's')).apply(
        lambda x: '{:.2f}'.format(x)).astype('float')
    df_derive['time_spent_on_submit_prsn_info'] = (
            (df_derive.prsn_info_submit_tms - df_derive.ine_submit_tms) / np.timedelta64(1, 's')).apply(
        lambda x: '{:.2f}'.format(x)).astype('float')
    df_derive['time_spent_on_submit_addr_info'] = (
            (df_derive.addr_submit_tms - df_derive.prsn_info_submit_tms) / np.timedelta64(1, 's')).apply(
        lambda x: '{:.2f}'.format(x)).astype('float')
    df_derive['time_spent_on_submit_contact_info'] = (
            (df_derive.contact_submit_tms - df_derive.addr_submit_tms) / np.timedelta64(1, 's')).apply(
        lambda x: '{:.2f}'.format(x)).astype('float')
    df_derive['time_spent_on_submit_fin_info'] = (
            (df_derive.fin_submit_tms - df_derive.contact_submit_tms) / np.timedelta64(1, 's')).apply(
        lambda x: '{:.2f}'.format(x)).astype('float')

    df_derive = df_derive.drop(columns=remove_col)
    print('The shape of user profile derived features is: {}\n'.format(df_derive.shape))

    return df_derive


def _create_weight(df):
    for col in ['dq30_plus_flag', 'dq60_plus_flag', 'stm_bal', 'credit_limit']:
        assert col in df.columns
    df['weight_dq'] = 1
    df.loc[df.dq30_plus_flag == 1, 'weight_dq'] = 2
    df.loc[df.stm_bal / df.credit_limit <= 0.1, 'weight_dq'] = 2
    df.loc[df.dq60_plus_flag == 1, 'weight_dq'] = 5


## define the tasks
def get_data_from_redshift():
    # code that read the driver data
    # and the raw data from redshift
    creds = get_cred()
    user = creds['redshift_user']
    password = creds['redshift_pass']
    engine = create_engine('postgresql://{}:{}@localhost:5439/powerup'.format(user, password))
    df_training, df_holdout = get_target_training_holdout(DATA_PATH_TARGET_TRAINING, DATA_PATH_TARGET_HOLDOUT, engine)
    qa_target(df_training, df_holdout)
    df_master_bdc = get_bdc_master_dataset()
    df_app_w_feature = get_save_app_data(DATA_PATH_APP_FEATURE, engine)
    df_training_w_app_bdc_feature = df_training.merge(df_app_w_feature, on=['appl_id'], how='inner').merge(
        df_master_bdc, on=['master_id', 'appl_id'], how='left')
    df_holdout_w_app_bdc_feature = df_holdout.merge(df_app_w_feature, on=['appl_id'], how='inner').merge(df_master_bdc,
                                                                                                         on=[
                                                                                                             'master_id',
                                                                                                             'appl_id'],
                                                                                                         how='left')
    assert df_training_w_app_bdc_feature.shape[1] == df_holdout_w_app_bdc_feature.shape[1]
    print(df_training_w_app_bdc_feature.shape, df_holdout_w_app_bdc_feature.shape)
    _create_weight(df_training_w_app_bdc_feature)
    df_training_w_app_bdc_feature.fillna(-1, inplace=True)
    df_training_w_app_bdc_feature.to_csv(DATA_PATH_BUILD, index=False)
    df_holdout_w_app_bdc_feature.to_csv(DATA_PATH_HOLDOUT, index=False)
    print('done create train/holdout dataset!')

    return None


def data_process_and_feature_creation():
    # code that process the data 
    # and create derived features
    return None


def feature_reduction():
    # process the raw data and do a feature reduction
    return None


def model_train_and_grid_search():
    # train the model and get the best parameters
    return None


def save_model_file():
    # save the model file
    return None


dag = DAG(
    dag_id='my_draft',
    description='Simple tutorial DAG',
    default_args=default_args)


# config = get_hdfs_config()

from_redshift = PythonOperator(
    task_id='get_data_from_redshift',
    python_callable=get_data_from_redshift,
    dag=dag)
data_pro = PythonOperator(
    task_id='data_process_and_feature_reduction',
    python_callable=data_process_and_feature_reduction,
    dag=dag)
train = PythonOperator(
    task_id='model_train_and_grid_search',
    python_callable=model_train_and_grid_search,
    dag=dag)
save = PythonOperator(
    task_id='save_model_file',
    python_callable=save_model_file,
    dag=dag
)

# setting dependencies
from_redshift >> data_pro
data_pro >> train
train >> save
