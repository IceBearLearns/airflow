from os.path import join, dirname, realpath
import numpy as np

CRED_FILE_PATH = '/Users/chutong/repos/creds.json'

PROJECT_PATH = dirname(realpath(__file__))
#
DATA_PATH = join(PROJECT_PATH, 'data')
DATA_PATH_BDC_FEATURE = join(DATA_PATH, 'bdc_feature')

TARGET_SQL = f"""
select
	slap.acct_id
	, slap.appl_id
	, slap.activated_mth
	, slap.approval_date
    , slap.sub_product_code
	, slap.statement_month_num
    , slap.statement_num
	, slap.credit_limit
	, slap.original_bal as stm_bal
	, slap.purchase_amt
	, slap.dq30_plus_flag
	, slap.dq60_plus_flag
--select count(*)
from ua.slap_mvp slap
where credit_card_indicator =1
	-- and statement_num =3
"""