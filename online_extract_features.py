
# coding: utf-8

# **划分数据集**

# In[1]:


#导入数据，并划分数据集
import pandas as pd
from datetime import date
import numpy as np

off_train=pd.read_csv(r"D:\O2O\data\ccf_offline_stage1_train.csv",keep_default_na=False)
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
off_train['date_received']=off_train['date_received'].astype('str')
off_train['date']=off_train['date'].astype('str')

off_test=pd.read_csv(r"D:\O2O\data\ccf_offline_stage1_test_revised.csv",keep_default_na=False)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
off_test['date_received']=off_test['date_received'].astype('str')

on_train=pd.read_csv(r"D:\O2O\data\ccf_online_stage1_train.csv",keep_default_na=False)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']
on_train['date_received']=on_train['date_received'].astype('str')
on_train['date']=on_train['date'].astype('str')

#线下
dataset3=off_test.copy()
feature3=off_train[((off_train.date>='20160315') & (off_train.date<='20160630'))|((off_train.date=='null') & (off_train.date_received>='20160315') &(off_train.date_received<='20160630'))]
dataset2=off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
feature2=off_train[((off_train.date>='20160201') & (off_train.date<='20160514'))|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]
dataset1=off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]
feature1=off_train[((off_train.date>='20160101')&(off_train.date<='20160413'))|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]

#线上
online_feature3=on_train[((on_train.date>='20160315') & (on_train.date<='20160630'))|((on_train.date=='null') & (on_train.date_received>='20160315') &(on_train.date_received<='20160630'))]
online_feature2=on_train[((on_train.date>='20160201') & (on_train.date<='20160514'))|((on_train.date=='null')&(on_train.date_received>='20160201')&(on_train.date_received<='20160514'))]
online_feature1=on_train[((on_train.date>='20160101')&(on_train.date<='20160413'))|((on_train.date=='null')&(on_train.date_received>='20160101')&(on_train.date_received<='20160413'))]


# In[2]:


"""
off-line表提取的特征
-- 1.merchant related:
--       sales_use_coupon. total_coupon
--       transfer_rate = sales_use_coupon/total_coupon.
--       merchant_avg_distance, merchant_min_distance, merchant_max_distance of those use coupon
--       total_sales.  coupon_rate = sales_use_coupon/total_sales.

--       新添加：
--       消费过该商家的不同用户数量   merchant_user_buy_count

-- 2.coupon related:
--       discount_rate. discount_man. discount_jian. is_man_jian
--       day_of_week,day_of_month. (date_received)

--       新添加：（label窗里的coupon在特征窗有出现，应提取相关特征）
--             label窗里的coupon，在特征窗中被消费过的数目  label_coupon_feature_buy_count
--             label窗里的coupon，在特征窗中被领取过的数目  label_coupon_feature_receive_count
--             label窗里的coupon，在特征窗中的核销率
               label_coupon_feature_rate = label_coupon_feature_buy_count/label_coupon_feature_receive_count

-- 3.user related:
--       distance.
--       user_avg_distance, user_min_distance,user_max_distance.
--       buy_use_coupon. buy_total. coupon_received.
--       buy_use_coupon/coupon_received.
--       avg_diff_date_datereceived. min_diff_date_datereceived. max_diff_date_datereceived.
--       count_merchant.

-- 4.user_merchant:
--       user_merchant_buy_total.  user_merchant_received    user_merchant_buy_use_coupon  user_merchant_any  user_merchant_buy_common
--       user_merchant_coupon_transform_rate = user_merchant_buy_use_coupon/user_merchant_received
--       user_merchant_coupon_buy_rate = user_merchant_buy_use_coupon/user_merchant_buy_total
--       user_merchant_common_buy_rate = user_merchant_buy_common/user_merchant_buy_total
--       user_merchant_rate = user_merchant_buy_total/user_merchant_any

-- 5. other feature:（label 窗提取的特征）
--       this_month_user_receive_all_coupon_count
--       this_month_user_receive_same_coupon_count
--       this_month_user_receive_same_coupon_lastone
--       this_month_user_receive_same_coupon_firstone
--       this_day_user_receive_all_coupon_count
--       this_day_user_receive_same_coupon_count

--       新添加：
--       day_gap_before, day_gap_after  (receive the same coupon)
--       商家有交集的用户数目 label_merchant_user_count
--       商家发出的所有优惠券数目  label_merchant_coupon_count
--       商家发出的所有优惠券种类数目  label_merchant_coupon_type_count
--       用户领取该商家的所有优惠券数目  label_user_merchant_coupon_count
--       用户在此次优惠券之后还领取了多少该优惠券   label_same_coupon_count_later
--       用户在此次优惠券之后还领取了多少优惠券     label_coupon_count_later
--       用户有交集的商家数目     label_user_merchant_count


-- 6. user_coupon:
--       对label窗里的user_coupon，特征窗里用户领取过该coupon几次   label_user_coupon_feature_receive_count
--       对label窗里的user_coupon，特征窗里用户用该coupon消费过几次   label_user_coupon_feature_buy_count
--       对label窗里的user_coupon，特征窗里用户对该coupon的核销率   label_user_coupon_feature_rate = label_user_coupon_feature_buy_count/label_user_coupon_feature_receive_count



-- 7. online表的特征（都是用户相关）
--       用户线上购买总次数  online_buy_total
--       用户线上用coupon购买的总次数 online_buy_use_coupon
--       用户线上用fixed购买的总次数  online_buy_use_fixed
--       用户线上收到的coupon总次数   online_coupon_received
--       用户线上有发生购买的merchant个数  online_buy_merchant_count
--       用户线上有action的merchant个数      online_action_merchant_count
--       online_buy_use_coupon_fixed = online_buy_use_coupon+online_buy_use_fixed
--       online_buy_use_coupon_rate = online_buy_use_coupon/online_buy_total
--       online_buy_use_fixed_rate = online_buy_use_fixed/online_buy_total
--       online_buy_use_coupon_fixed_rate = online_buy_use_coupon_fixed/online_buy_total
--       online_coupon_transform_rate = online_buy_use_coupon/online_coupon_received
"""


# **特征工程**

# In[3]:


#计算比率,当分母为0时，值为-1
def cal_rate(one,two):
    result=[]
    for i in range(len(one)):
        if two[i]==0:
            result.append(-1)
        else:
            result.append(float(one[i])/float(two[i]))
    return result


# In[4]:


"""
7. online feature for user
"""


# In[5]:


#for dataset3-dataset1

online_feature_list=[online_feature3,online_feature2,online_feature1]
count=3
for online_feature in online_feature_list:
#获取唯一用户
    t=online_feature[['user_id']].copy()
    t.drop_duplicates(inplace=True)

    #用户线上购买总次数
    t1=online_feature[online_feature['action']==1][['user_id']]
    t1['online_buy_total']=1
    t1=t1.groupby('user_id')['online_buy_total'].agg('sum').reset_index()

    #用户线上用coupon购买的总次数
    t2=online_feature[(online_feature.action==1)&(online_feature.coupon_id!='null')&(online_feature.coupon_id!='fixed')][['user_id']]
    t2['online_buy_use_coupon']=1
    t2=t2.groupby('user_id')['online_buy_use_coupon'].agg('sum').reset_index()

    #用户线上用fixed购买的总次数
    t3=online_feature[(online_feature.action==1)&(online_feature.coupon_id=='fixed')][['user_id']]
    t3['online_buy_use_fixed']=1
    t3=t3.groupby('user_id')['online_buy_use_fixed'].agg('sum').reset_index()

    #用户收到的coupon总次数
    t4=online_feature[(online_feature.coupon_id !='null')&(online_feature.coupon_id !='fixed')][['user_id']]
    t4['online_coupon_received']=1
    t4=t4.groupby('user_id')['online_coupon_received'].agg('sum').reset_index()

    #用户线上有发生购买的merchant个数  online_buy_merchant_count
    t5=online_feature[online_feature.action==1][['user_id','merchant_id']]
    t5['merchant_id']=1
    t5=t5.groupby('user_id')['merchant_id'].agg('sum').reset_index()
    t5.rename(columns={'merchant_id':'online_buy_merchant_count'},inplace=True)

    #线上有action的merchant个数
    t6=online_feature[['user_id','merchant_id']]
    t6['merchant_id']=1
    t6=t6.groupby('user_id')['merchant_id'].agg('sum').reset_index()
    t6.rename(columns={'merchant_id':'online_action_merchant_count'},inplace=True)

    on_feature=pd.merge(t,t1,on='user_id',how='left')
    on_feature=pd.merge(on_feature,t2,on='user_id',how='left')
    on_feature=pd.merge(on_feature,t3,on='user_id',how='left')
    on_feature=pd.merge(on_feature,t4,on='user_id',how='left')
    on_feature=pd.merge(on_feature,t5,on='user_id',how='left')
    on_feature=pd.merge(on_feature,t6,on='user_id',how='left')

    on_feature=on_feature.replace(np.nan,0)
    on_feature['online_buy_use_coupon_fixed']=on_feature['online_buy_use_coupon']+on_feature['online_buy_use_fixed']
    on_feature['online_buy_use_coupon_rate']=cal_rate(on_feature['online_buy_use_coupon'],on_feature['online_buy_total'])
    on_feature['online_buy_use_fixed_rate']=cal_rate(on_feature['online_buy_use_fixed'],on_feature['online_buy_total'])
    on_feature['online_buy_use_coupon_fixed_rate']=cal_rate(on_feature['online_buy_use_coupon_fixed'],on_feature['online_buy_total'])
    on_feature['online_coupon_transform_rate']=cal_rate(on_feature['online_buy_use_coupon'],on_feature['online_coupon_received'])
    on_feature.to_csv(r'D:\O2O\features\on feature\on_feature{}.csv'.format(count),index=None)
    
    count -=1

print('end')


# In[6]:


"""
5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""


# In[7]:


#for dataset3-dataset1

dataset_list=[dataset3,dataset2,dataset1]
count=3

for dataset in dataset_list:
    #每个用户领取的所有优惠券数量
    t=dataset[['user_id']].copy()
    t['this_month_uesr_receive_all_coupon_count']=1
    t=t.groupby('user_id').agg('sum').reset_index()

    #本月用户收到的相同的优惠券数量
    t1=dataset[['user_id','coupon_id']].copy()
    t1['this_month_user_receive_same_coupon_count']=1
    t1=t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

    #用户收到某优惠券的最大日期和最小日期
    t2=dataset[['user_id','coupon_id','date_received']].copy()
    t2=t2.groupby(['user_id','coupon_id'])['date_received'].agg(lambda x: ':'.join(x)).reset_index()
    t2['received_number']=t2.date_received.apply(lambda s:len(s.split(':')))
    t2=t2[t2.received_number>1]
    t2['max_date_received']=t2.date_received.apply(lambda s:max([int(d) for d in s.split(':')]))
    t2['min_date_received']=t2.date_received.apply(lambda s:min([int(d) for d in s.split(':')]))
    t2=t2[['user_id','coupon_id','max_date_received','min_date_received']]

    #用户领取相同优惠券是否是第一个或最后一个，1表示是，0表示不是，-1表示只收到过一个这个优惠券，无最大最小日期
    t3=dataset[['user_id','coupon_id','merchant_id','date_received']].copy()
    t3=pd.merge(t3,t2,on=['user_id','coupon_id'],how='left')
    t3['this_month_user_receive_same_coupon_lastone']=t3.max_date_received-t3.date_received.astype('int')
    t3['this_month_user_receive_same_coupon_firstone']=t3.date_received.astype('int')-t3.min_date_received
    def is_firstlastone(x):
        if x==0:
            return 1
        elif x>0:
            return 0
        else:
            return -1 #those only received once
    t3.this_month_user_receive_same_coupon_firstone=t3.this_month_user_receive_same_coupon_firstone.apply(is_firstlastone)
    t3.this_month_user_receive_same_coupon_lastone=t3.this_month_user_receive_same_coupon_lastone.apply(is_firstlastone)
    t3=t3[['user_id','coupon_id','merchant_id','date_received','this_month_user_receive_same_coupon_lastone','this_month_user_receive_same_coupon_firstone']]

    #用户在一天收到的优惠券总量
    t4=dataset[['user_id','date_received']].copy()
    t4['this_day_user_receive_all_coupon_count']=1
    t4=t4.groupby(['user_id','date_received']).agg('sum').reset_index()

    #用户在一天收到的相同优惠券的数量
    t5=dataset[['user_id','coupon_id','date_received']].copy()
    t5['this_day_user_receive_same_coupon_count']=1
    t5=t5.groupby(['user_id','coupon_id','date_received']).agg('sum').reset_index()

    #label_merchant_user_count
    t6=dataset[['user_id','merchant_id']].copy()
    t6.drop_duplicates(inplace=True)
    t6['user_id']=1
    t6=t6.groupby('merchant_id')['user_id'].agg('sum').reset_index()
    t6.rename(columns={'user_id':'label_merchant_user_count'},inplace=True)

    #label_user_merchant_count
    t7=dataset[['user_id','merchant_id']].copy()
    t7.drop_duplicates(inplace=True)
    t7['merchant_id']=1
    t7=t7.groupby('user_id')['merchant_id'].agg('sum').reset_index()
    t7.rename(columns={'merchant_id':'label_user_merchant_count'},inplace=True)

    #label_merchant_coupon_count
    t8=dataset[['merchant_id','coupon_id']].copy()
    t8['coupon_id']=1
    t8=t8.groupby('merchant_id')['coupon_id'].agg('sum').reset_index()
    t8.rename(columns={'coupon_id':'label_merchant_coupon_count'},inplace=True)

    #label_merchant_coupon_type_count
    t9=dataset[['merchant_id','coupon_id']].copy()
    t9.drop_duplicates(inplace=True)
    t9['coupon_id']=1
    t9=t9.groupby('merchant_id')['coupon_id'].agg('sum').reset_index()
    t9.rename(columns={'coupon_id':'label_merchant_coupon_type_count'},inplace=True)

    #label_user_merchant_coupon_count
    t10=dataset[['user_id','merchant_id']].copy()
    t10['merchant_id']=1
    t10=t10.groupby('user_id')['merchant_id'].agg('sum').reset_index()
    t10.rename(columns={'merchant_id':'label_user_merchant_coupon_count'},inplace=True)

    #label_same_coupon_count_later
    t11=dataset[['user_id','coupon_id','date_received']].copy()
    t11['date_received']=t11['date_received'].astype('int')
    t11.drop_duplicates(inplace=True)
    t11['label_same_coupon_count_later']=t11.groupby(['user_id','coupon_id']).rank(ascending=False)
    t11['label_same_coupon_count_later']=t11['label_same_coupon_count_later']-1
    t11['date_received']=t11['date_received'].astype('str')

    #label_coupon_count_later
    t12=dataset[['user_id','date_received']].copy()
    t12['date_received']=t12['date_received'].astype('int')
    t12.drop_duplicates(inplace=True)
    t12['label_coupon_count_later']=t12.groupby('user_id').rank(ascending=False)
    t12['label_coupon_count_later']=t12['label_coupon_count_later']-1
    t12['date_received']=t12['date_received'].astype('str')

    #输出处理好的文件
    other_feature=pd.merge(t3,t,on='user_id',how='inner')
    other_feature=pd.merge(other_feature,t1,on=['user_id','coupon_id'])
    other_feature=pd.merge(other_feature,t4,on=['user_id','date_received'])
    other_feature=pd.merge(other_feature,t5,on=['user_id','coupon_id','date_received'])
    other_feature=pd.merge(other_feature,t6,on='merchant_id',how='left')
    other_feature=pd.merge(other_feature,t7,on='user_id',how='left')
    other_feature=pd.merge(other_feature,t8,on='merchant_id',how='left')
    other_feature=pd.merge(other_feature,t9,on='merchant_id',how='left')
    other_feature=pd.merge(other_feature,t10,on='user_id',how='left')
    other_feature=pd.merge(other_feature,t11,on=['user_id','coupon_id','date_received'],how='left')
    other_feature=pd.merge(other_feature,t12,on=['user_id','date_received'],how='left')
    other_feature.to_csv(r'D:\O2O\features\other feature\other_feature{}.csv'.format(count),index=None)
    
    count -=1

print('end')


# In[8]:


############# coupon related feature   #############
"""
2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
"""


# In[9]:


def calc_discount_rate(s):
    s=str(s)
    s=s.split(':')
    if len(s)==1:
        return float(s[0])
    else:
        return 1.0-float(s[1])/float(s[0])

def get_discount_man(s):
    s=str(s)
    s=s.split(':')
    if len(s)==1:
        return 'null'
    else:
        return int(s[0])

def get_discount_jian(s):
    s=str(s)
    s=s.split(':')
    if len(s)==1:
        return 'null'
    else:
        return int(s[1])

def is_man_jian(s):
    s=str(s)
    s=s.split(':')
    if len(s)==1:
        return 0
    else:
        return 1


# In[10]:


#for dataset3

dataset3['day_of_week']=dataset3['date_received'].apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset3['day_of_month']=dataset3['date_received'].apply(lambda x:int(x[6:8]))
dataset3['days_distance']=dataset3['date_received'].apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,6,30)).days)
dataset3['discount_man']=dataset3.discount_rate.apply(get_discount_man)
dataset3['discount_jian']=dataset3.discount_rate.apply(get_discount_jian)
dataset3['is_man_jian']=dataset3.discount_rate.apply(is_man_jian)
dataset3['discount_rate']=dataset3.discount_rate.apply(calc_discount_rate)

#优惠券类型
d=dataset3[['coupon_id']].copy()
d['coupon_count']=1
d=d.groupby('coupon_id')['coupon_count'].agg('sum').reset_index()

# label窗里的coupon，在特征窗中被领取过的数目,  label_coupon_feature_receive_count
coupon_dataset=dataset3[['coupon_id']].copy()
coupon_dataset.drop_duplicates(inplace=True)
coupon_dataset['coupon_id']=coupon_dataset['coupon_id'].astype('int')

coupon_feature=feature3[['coupon_id']].copy()
coupon_feature['label_coupon_feature_receive_count']=1
coupon_feature['coupon_id'].replace('null',-1,inplace=True)
coupon_feature['coupon_id']=coupon_feature['coupon_id'].astype('int')

d1=pd.merge(coupon_dataset,coupon_feature,on='coupon_id',how='left')
d1['label_coupon_feature_receive_count'].replace(np.nan,0,inplace=True)
d1=d1.groupby('coupon_id').agg('sum').reset_index()

# label窗里的coupon，在特征窗中被消费过的数目  label_coupon_feature_buy_count
coupon_dataset=dataset3[['coupon_id']].copy()
coupon_dataset.drop_duplicates(inplace=True)
coupon_dataset['coupon_id']=coupon_dataset['coupon_id'].astype('int')

coupon_feature=feature3[(feature3.coupon_id!='null')&(feature3.date !='null')][['coupon_id']].copy()          ##&(feature3.date !='null')][['coupon_id']].copy()
coupon_feature['label_coupon_feature_buy_count']=1
coupon_feature['coupon_id']=coupon_feature['coupon_id'].astype('int')

d2=pd.merge(coupon_dataset,coupon_feature,on='coupon_id',how='left')
d2['label_coupon_feature_buy_count'].replace(np.nan,0,inplace=True)
d2=d2.groupby('coupon_id').agg('sum').reset_index()

dataset3=pd.merge(dataset3,d,on='coupon_id',how='left')
dataset3=pd.merge(dataset3,d1,on='coupon_id',how='left')
dataset3=pd.merge(dataset3,d2,on='coupon_id',how='left')

# label窗里的coupon，在特征窗中的核销率
dataset3['label_coupon_feature_rate']=cal_rate(dataset3['label_coupon_feature_buy_count'],dataset3['label_coupon_feature_receive_count'])

dataset3.to_csv(r'D:\O2O\features\coupon feature\coupon3_feature.csv',index=None)

#for dataset2

dataset2['day_of_week']=dataset2['date_received'].apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset2['day_of_month']=dataset2['date_received'].apply(lambda x:int(x[6:8]))
dataset2['days_distance']=dataset2['date_received'].apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,5,14)).days)
dataset2['discount_man']=dataset2.discount_rate.apply(get_discount_man)
dataset2['discount_jian']=dataset2.discount_rate.apply(get_discount_jian)
dataset2['is_man_jian']=dataset2.discount_rate.apply(is_man_jian)
dataset2['discount_rate']=dataset2.discount_rate.apply(calc_discount_rate)

#优惠券类型
d=dataset2[['coupon_id']].copy()
d['coupon_count']=1
d=d.groupby('coupon_id')['coupon_count'].agg('sum').reset_index()

# label窗里的coupon，在特征窗中被领取过的数目,  label_coupon_feature_receive_count
coupon_dataset=dataset2[['coupon_id']].copy()
coupon_dataset.drop_duplicates(inplace=True)


coupon_feature=feature2[['coupon_id']].copy()
coupon_feature['label_coupon_feature_receive_count']=1
coupon_feature['coupon_id'].replace('null',-1,inplace=True)


d1=pd.merge(coupon_dataset,coupon_feature,on='coupon_id',how='left')
d1['label_coupon_feature_receive_count'].replace([-1,np.nan],0,inplace=True)
d1=d1.groupby('coupon_id').agg('sum').reset_index()

# label窗里的coupon，在特征窗中被消费过的数目  label_coupon_feature_buy_count
coupon_dataset=dataset2[['coupon_id']].copy()
coupon_dataset.drop_duplicates(inplace=True)


coupon_feature=feature2[(feature2.coupon_id!='null')&(feature2.date !='null')][['coupon_id']].copy()      ##&(feature2.date !='null')][['coupon_id']].copy()
coupon_feature['label_coupon_feature_buy_count']=1


d2=pd.merge(coupon_dataset,coupon_feature,on='coupon_id',how='left')
d2['label_coupon_feature_buy_count'].replace(np.nan,0,inplace=True)
d2=d2.groupby('coupon_id').agg('sum').reset_index()

dataset2=pd.merge(dataset2,d,on='coupon_id',how='left')
dataset2=pd.merge(dataset2,d1,on='coupon_id',how='left')
dataset2=pd.merge(dataset2,d2,on='coupon_id',how='left')

# label窗里的coupon，在特征窗中的核销率
dataset2['label_coupon_feature_rate']=cal_rate(dataset2['label_coupon_feature_buy_count'],dataset2['label_coupon_feature_receive_count'])

dataset2.to_csv(r'D:\O2O\features\coupon feature\coupon2_feature.csv',index=None)

#for dataset1

dataset1['day_of_week']=dataset1['date_received'].apply(lambda x:date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
dataset1['day_of_month']=dataset1['date_received'].apply(lambda x:int(x[6:8]))
dataset1['days_distance']=dataset1['date_received'].apply(lambda x:(date(int(x[0:4]),int(x[4:6]),int(x[6:8]))-date(2016,4,13)).days)
dataset1['discount_man']=dataset1.discount_rate.apply(get_discount_man)
dataset1['discount_jian']=dataset1.discount_rate.apply(get_discount_jian)
dataset1['is_man_jian']=dataset1.discount_rate.apply(is_man_jian)
dataset1['discount_rate']=dataset1.discount_rate.apply(calc_discount_rate)

#优惠券类型
d=dataset1[['coupon_id']].copy()
d['coupon_count']=1
d=d.groupby('coupon_id')['coupon_count'].agg('sum').reset_index()

# label窗里的coupon，在特征窗中被领取过的数目,  label_coupon_feature_receive_count
coupon_dataset=dataset1[['coupon_id']].copy()
coupon_dataset.drop_duplicates(inplace=True)


coupon_feature=feature1[['coupon_id']].copy()
coupon_feature['label_coupon_feature_receive_count']=1
coupon_feature['coupon_id'].replace('null',-1,inplace=True)


d1=pd.merge(coupon_dataset,coupon_feature,on='coupon_id',how='left')
d1['label_coupon_feature_receive_count'].replace([-1,np.nan],0,inplace=True)
d1=d1.groupby('coupon_id').agg('sum').reset_index()

# label窗里的coupon，在特征窗中被消费过的数目  label_coupon_feature_buy_count
coupon_dataset=dataset1[['coupon_id']].copy()
coupon_dataset.drop_duplicates(inplace=True)


coupon_feature=feature1[(feature1.coupon_id!='null')&(feature1.date !='null')][['coupon_id']].copy()    ##&(feature1.date !='null')][['coupon_id']].copy()
coupon_feature['label_coupon_feature_buy_count']=1


d2=pd.merge(coupon_dataset,coupon_feature,on='coupon_id',how='left')
d2['label_coupon_feature_buy_count'].replace(np.nan,0,inplace=True)
d2=d2.groupby('coupon_id').agg('sum').reset_index()

dataset1=pd.merge(dataset1,d,on='coupon_id',how='left')
dataset1=pd.merge(dataset1,d1,on='coupon_id',how='left')
dataset1=pd.merge(dataset1,d2,on='coupon_id',how='left')

# label窗里的coupon，在特征窗中的核销率
dataset1['label_coupon_feature_rate']=cal_rate(dataset1['label_coupon_feature_buy_count'],dataset1['label_coupon_feature_receive_count'])

dataset1.to_csv(r'D:\O2O\features\coupon feature\coupon1_feature.csv',index=None)

print('end')


# In[11]:


############# merchant related feature   #############
"""
1.merchant related: 
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon

"""


# In[12]:


#for dataset3-dataset1
offline_feature_list=[feature3,feature2,feature1]
count=3
for offline_feature in offline_feature_list:

    merchant3=offline_feature[['merchant_id','user_id','coupon_id','distance','date_received','date']].copy()

    #商家id
    t=merchant3[['merchant_id']].copy()
    t.drop_duplicates(inplace=True)

    #total_sales
    t1=merchant3[merchant3.date !='null'][['merchant_id']]   
    t1['total_sales']=1
    t1=t1.groupby('merchant_id')['total_sales'].agg('sum').reset_index()

    #sales_use_coupon
    t2=merchant3[(merchant3.date!='null')&(merchant3.coupon_id!='null')][['merchant_id']]
    t2['sales_use_coupon']=1
    t2=t2.groupby('merchant_id')['sales_use_coupon'].agg('sum').reset_index()

    #total_coupon
    t3=merchant3[merchant3.coupon_id!='null'][['merchant_id']]
    t3['total_coupon']=1
    t3=t3.groupby('merchant_id')['total_coupon'].agg('sum').reset_index()

    t4=merchant3[(merchant3.date!='null')&(merchant3.coupon_id!='null')&(merchant3.distance!='null')][['merchant_id','distance']]
    t4.distance=t4.distance.astype('int')

    #merchant_min_distance
    t5=t4.groupby('merchant_id')['distance'].agg('min').reset_index()
    t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

    #merchant_max_distance
    t6=t4.groupby('merchant_id')['distance'].agg('max').reset_index()
    t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

    #merchant_mean_distance
    t7=t4.groupby('merchant_id')['distance'].agg('mean').reset_index()
    t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

    #merchant_median_distance
    t8=t4.groupby('merchant_id')['distance'].agg('median').reset_index()
    t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

    #消费过该商家的不同用户数量  merchant_user_buy_count
    t9=merchant3[merchant3.date!='null'][['merchant_id','user_id']]
    t9.drop_duplicates(inplace=True)
    t9['user_id']=1
    t9=t9.groupby('merchant_id').agg('sum').reset_index()
    t9.rename(columns={'user_id':'merchant_user_buy_count'},inplace=True)

    t10=merchant3[merchant3.coupon_id!='null'][['merchant_id','coupon_id']]
    t10.drop_duplicates(inplace=True)
    t10['coupon_id']=1
    t10=t10.groupby('merchant_id').agg('sum').reset_index()
    t10.rename(columns={'coupon_id':'merchant_distinct_coupon_count'},inplace=True)

    merchant_feature=pd.merge(t,t1,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t2,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t3,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t5,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t6,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t7,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t8,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t9,on='merchant_id',how='left')
    merchant_feature=pd.merge(merchant_feature,t10,on='merchant_id',how='left')

    merchant_feature['sales_use_coupon']=merchant_feature['sales_use_coupon'].replace(np.nan,0)
    merchant_feature['merchant_coupon_transfer_rate']=merchant_feature['sales_use_coupon'].astype('float')/merchant_feature['total_coupon']
    merchant_feature['coupon_rate']=merchant_feature.sales_use_coupon.astype('float')/merchant_feature.total_sales
    merchant_feature['total_coupon']=merchant_feature['total_coupon'].replace(np.nan,0)

    merchant_feature.to_csv(r'D:\O2O\features\merchant feature\merchant{}_feature.csv'.format(count),index=None)
    
    count -=1
print('end')


# In[13]:


############# user related feature   #############
"""
3.user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap
      

"""


# In[14]:


def get_user_date_datereceived_gap(s):
    s=s.split(':')
    return (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days


# In[15]:


#for dataset3-dataset1
offline_feature_list=[feature3,feature2,feature1]
count=3
for offline_feature in offline_feature_list:
    
    user3=offline_feature[['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']]

    t=user3[['user_id']].copy()
    t.drop_duplicates(inplace=True)

    #count_merchant
    t1=user3[user3.date!='null'][['user_id','merchant_id']]
    t1.drop_duplicates(inplace=True)
    t1['merchant_id']=1
    t1=t1.groupby('user_id')['merchant_id'].agg('sum').reset_index()
    t1.rename(columns={'merchant_id':'count_merchant'},inplace=True)

    t2=user3[(user3.date!='null')&(user3.coupon_id!='null')&(user3.distance!='null')][['user_id','distance']]
    t2.distance=t2.distance.astype('int')

    #user_min_distance
    t3=t2.groupby('user_id')['distance'].agg('min').reset_index()
    t3.rename(columns={'distance':'user_min_distance'},inplace=True)

    #user_max_distance
    t4=t2.groupby('user_id')['distance'].agg('max').reset_index()
    t4.rename(columns={'distance':'user_max_distance'},inplace=True)

    #user_mean_distance
    t5=t2.groupby('user_id')['distance'].agg('mean').reset_index()
    t5.rename(columns={'distance':'user_mean_distance'},inplace=True)

    #user_median_distance
    t6=t2.groupby('user_id')['distance'].agg('median').reset_index()
    t6.rename(columns={'distance':'user_median_distance'},inplace=True)

    #buy_use_coupon
    t7=user3[(user3.date!='null')&(user3.coupon_id!='null')][['user_id']]
    t7['buy_use_coupon']=1
    t7=t7.groupby('user_id')['buy_use_coupon'].agg('sum').reset_index()

    #buy_total
    t8=user3[user3.date!='null'][['user_id']]
    t8['buy_total']=1
    t8=t8.groupby('user_id')['buy_total'].agg('sum').reset_index()

    #coupon_received
    t9=user3[user3.coupon_id!='null'][['user_id']]
    t9['coupon_received']=1
    t9=t9.groupby('user_id')['coupon_received'].agg('sum').reset_index()

    t10=user3[(user3.date_received!='null')&(user3.date!='null')][['user_id','date_received','date']]
    t10['user_date_datereceived_gap']=t10.date +':'+t10.date_received
    t10['user_date_datereceived_gap']=t10['user_date_datereceived_gap'].apply(get_user_date_datereceived_gap)
    t10=t10[['user_id','user_date_datereceived_gap']]

    #avg_user_date_datereceived_gap
    t11=t10.groupby('user_id')['user_date_datereceived_gap'].agg('mean').reset_index()
    t11.rename(columns={'user_date_datereceived_gap':'avg_user_date_datereceived_gap'},inplace=True)

    #min_user_date_datereceived_gap
    t12=t10.groupby('user_id')['user_date_datereceived_gap'].agg('min').reset_index()
    t12.rename(columns={'user_date_datereceived_gap':'min_user_date_datereceived_gap'},inplace=True)

    #max_user_date_datereceived_gap
    t13=t10.groupby('user_id')['user_date_datereceived_gap'].agg('max').reset_index()
    t13.rename(columns={'user_date_datereceived_gap':'max_user_date_datereceived_gap'},inplace=True)

    user_feature=pd.merge(t,t1,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t3,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t4,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t5,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t6,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t7,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t8,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t9,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t11,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t12,on='user_id',how='left')
    user_feature=pd.merge(user_feature,t13,on='user_id',how='left')
    
    user_feature.count_merchant=user_feature.count_merchant.replace(np.nan,0)
    user_feature.buy_use_coupon=user_feature.buy_use_coupon.replace(np.nan,0)

    user_feature['buy_use_coupon_rate']=user_feature.buy_use_coupon.astype('float')/user_feature.buy_total.astype('float')
    user_feature['user_coupon_transfer_rate']=user_feature.buy_use_coupon.astype('float')/user_feature.coupon_received.astype('float')
    user_feature['buy_total']=user_feature['buy_total'].replace(np.nan,0)
    user_feature['coupon_received']=user_feature['coupon_received'].replace(np.nan,0)

    user_feature.to_csv(r'D:\O2O\features\user feature\user{}_feature.csv'.format(count),index=None)
    count -=1
print('end')


# In[16]:


##################  user_merchant related feature #########################

"""
4.user_merchant:
      times_user_buy_merchant_before. 
"""


# In[17]:


#for dataset3
offline_feature_list=[feature3,feature2,feature1]
count=3
for offline_feature in offline_feature_list:

    all_user_merchant=offline_feature[['user_id','merchant_id']].copy()
    all_user_merchant.drop_duplicates(inplace=True)

    #user_merchant_buy_total
    t=offline_feature[['user_id','merchant_id','date']].copy()
    t=t[t.date!='null'][['user_id','merchant_id']]
    t['user_merchant_buy_total']=1
    t=t.groupby(['user_id','merchant_id'])['user_merchant_buy_total'].agg('sum').reset_index()
    t.drop_duplicates(inplace=True)

    #user_merchant_received
    t1=offline_feature[['user_id','merchant_id','coupon_id']].copy()
    t1=t1[t1.coupon_id!='null'][['user_id','merchant_id']]
    t1['user_merchant_received']=1
    t1=t1.groupby(['user_id','merchant_id'])['user_merchant_received'].agg('sum').reset_index()
    t1.drop_duplicates(inplace=True)

    #user_merchant_buy_use_coupon
    t2=offline_feature[['user_id','merchant_id','date','date_received']].copy()
    t2=t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
    t2['user_merchant_buy_use_coupon']=1
    t2=t2.groupby(['user_id','merchant_id'])['user_merchant_buy_use_coupon'].agg('sum').reset_index()
    t2.drop_duplicates(inplace=True)

    #用户-商家数量  user_merchant_any
    t3=offline_feature[['user_id','merchant_id']].copy()
    t3['user_merchant_any']=1
    t3=t3.groupby(['user_id','merchant_id'])['user_merchant_any'].agg('sum').reset_index()
    t3.drop_duplicates(inplace=True)

    #user_merchant_buy_commom
    t4=offline_feature[['user_id','merchant_id','date','coupon_id']].copy()
    t4=t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
    t4['user_merchant_buy_common']=1
    t4=t4.groupby(['user_id','merchant_id'])['user_merchant_buy_common'].agg('sum').reset_index()
    t4.drop_duplicates(inplace=True)

    user_merchant=pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
    user_merchant=pd.merge(user_merchant,t1,on=['user_id','merchant_id'],how='left')
    user_merchant=pd.merge(user_merchant,t2,on=['user_id','merchant_id'],how='left')
    user_merchant=pd.merge(user_merchant,t3,on=['user_id','merchant_id'],how='left')
    user_merchant=pd.merge(user_merchant,t4,on=['user_id','merchant_id'],how='left')

    user_merchant.user_merchant_buy_use_coupon = user_merchant.user_merchant_buy_use_coupon.replace(np.nan,0)
    user_merchant.user_merchant_buy_common = user_merchant.user_merchant_buy_common.replace(np.nan,0)

    user_merchant['user_merchant_coupon_transfer_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_received.astype('float')
    user_merchant['user_merchant_coupon_buy_rate'] = user_merchant.user_merchant_buy_use_coupon.astype('float') / user_merchant.user_merchant_buy_total.astype('float')
    user_merchant['user_merchant_rate'] = user_merchant.user_merchant_buy_total.astype('float') / user_merchant.user_merchant_any.astype('float')
    user_merchant['user_merchant_common_buy_rate'] = user_merchant.user_merchant_buy_common.astype('float') / user_merchant.user_merchant_buy_total.astype('float')

    user_merchant.to_csv(r'D:\O2O\features\user_merchant feature\user_merchant{}.csv'.format(count),index=None)
    count -=1
print('end')


# In[18]:


"""
6. user_coupon
    label_user_coupon_feature_receive_count
    label_user_coupon_feature_buy_count
    label_user_coupon_feature_rate = label_user_coupon_feature_buy_count/label_user_coupon_feature_receive_count
"""


# In[19]:


#for dataset3

# 对label窗里的user_coupon，特征窗里用户领取过该coupon几次   label_user_coupon_feature_receive_count
user_coupon_dataset=dataset3[['user_id','coupon_id']].copy()
user_coupon_dataset.drop_duplicates(inplace=True)

user_coupon_feature=feature3[['user_id','coupon_id']].copy()
user_coupon_feature['label_user_coupon_feature_receive_count']=1

t1=pd.merge(user_coupon_dataset,user_coupon_feature,on=['user_id','coupon_id'],how='left')
t1['label_user_coupon_feature_receive_count'].replace(np.nan,0,inplace=True)
t1=t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 对label窗里的user_coupon，特征窗里用户用该coupon消费过几次   label_user_coupon_feature_buy_count
user_coupon_dataset=dataset3[['user_id','coupon_id']].copy()
user_coupon_dataset.drop_duplicates(inplace=True)

user_coupon_feature=feature3[(feature3.date !='null')&(feature3.coupon_id !='null')][['user_id','coupon_id']].copy()
user_coupon_feature['label_user_coupon_feature_buy_count']=1

t2=pd.merge(user_coupon_dataset,user_coupon_feature,on=['user_id','coupon_id'],how='left')
t2['label_user_coupon_feature_buy_count'].replace(np.nan,0,inplace=True)
t2=t2.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 对label窗里的user_coupon，特征窗里用户对该coupon的核销率   
#label_user_coupon_feature_rate = label_user_coupon_feature_buy_count/label_user_coupon_feature_receive_count
user_coupon=pd.merge(t1,t2,on=['user_id','coupon_id'],how='left')
user_coupon['label_user_coupon_feature_rate']=cal_rate(user_coupon['label_user_coupon_feature_buy_count'],user_coupon['label_user_coupon_feature_receive_count'])

user_coupon.to_csv(r'D:\O2O\features\user_coupon feature\user_coupon3.csv',index=None)


#for dataset2

# 对label窗里的user_coupon，特征窗里用户领取过该coupon几次   label_user_coupon_feature_receive_count
user_coupon_dataset=dataset2[['user_id','coupon_id']].copy()
user_coupon_dataset.drop_duplicates(inplace=True)

user_coupon_feature=feature2[['user_id','coupon_id']].copy()
user_coupon_feature['label_user_coupon_feature_receive_count']=1

t1=pd.merge(user_coupon_dataset,user_coupon_feature,on=['user_id','coupon_id'],how='left')
t1['label_user_coupon_feature_receive_count'].replace(np.nan,0,inplace=True)
t1=t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 对label窗里的user_coupon，特征窗里用户用该coupon消费过几次   label_user_coupon_feature_buy_count
user_coupon_dataset=dataset2[['user_id','coupon_id']].copy()
user_coupon_dataset.drop_duplicates(inplace=True)

user_coupon_feature=feature2[(feature2.date !='null')&(feature2.coupon_id !='null')][['user_id','coupon_id']].copy()
user_coupon_feature['label_user_coupon_feature_buy_count']=1

t2=pd.merge(user_coupon_dataset,user_coupon_feature,on=['user_id','coupon_id'],how='left')
t2['label_user_coupon_feature_buy_count'].replace(np.nan,0,inplace=True)
t2=t2.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 对label窗里的user_coupon，特征窗里用户对该coupon的核销率   
#label_user_coupon_feature_rate = label_user_coupon_feature_buy_count/label_user_coupon_feature_receive_count
user_coupon=pd.merge(t1,t2,on=['user_id','coupon_id'],how='left')
user_coupon['label_user_coupon_feature_rate']=cal_rate(user_coupon['label_user_coupon_feature_buy_count'],user_coupon['label_user_coupon_feature_receive_count'])

user_coupon.to_csv(r'D:\O2O\features\user_coupon feature\user_coupon2.csv',index=None)


#for dataset1

# 对label窗里的user_coupon，特征窗里用户领取过该coupon几次   label_user_coupon_feature_receive_count
user_coupon_dataset=dataset1[['user_id','coupon_id']].copy()
user_coupon_dataset.drop_duplicates(inplace=True)

user_coupon_feature=feature1[['user_id','coupon_id']].copy()
user_coupon_feature['label_user_coupon_feature_receive_count']=1

t1=pd.merge(user_coupon_dataset,user_coupon_feature,on=['user_id','coupon_id'],how='left')
t1['label_user_coupon_feature_receive_count'].replace(np.nan,0,inplace=True)
t1=t1.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 对label窗里的user_coupon，特征窗里用户用该coupon消费过几次   label_user_coupon_feature_buy_count
user_coupon_dataset=dataset1[['user_id','coupon_id']].copy()
user_coupon_dataset.drop_duplicates(inplace=True)

user_coupon_feature=feature1[(feature1.date !='null')&(feature1.coupon_id !='null')][['user_id','coupon_id']].copy()
user_coupon_feature['label_user_coupon_feature_buy_count']=1

t2=pd.merge(user_coupon_dataset,user_coupon_feature,on=['user_id','coupon_id'],how='left')
t2['label_user_coupon_feature_buy_count'].replace(np.nan,0,inplace=True)
t2=t2.groupby(['user_id','coupon_id']).agg('sum').reset_index()

# 对label窗里的user_coupon，特征窗里用户对该coupon的核销率   
#label_user_coupon_feature_rate = label_user_coupon_feature_buy_count/label_user_coupon_feature_receive_count
user_coupon=pd.merge(t1,t2,on=['user_id','coupon_id'],how='left')
user_coupon['label_user_coupon_feature_rate']=cal_rate(user_coupon['label_user_coupon_feature_buy_count'],user_coupon['label_user_coupon_feature_receive_count'])

user_coupon.to_csv(r'D:\O2O\features\user_coupon feature\user_coupon1.csv',index=None)

print('end')


# **生成训练集和验证集**

# In[20]:


"""dataset split:
                      (date_received)                              
           dateset3: 20160701~20160731 (113640),features3 from 20160315~20160630  (off_test)
           dateset2: 20160515~20160615 (258446),features2 from 20160201~20160514  
           dateset1: 20160414~20160514 (138303),features1 from 20160101~20160413   
"""


# In[21]:


def get_label(s):
    s=s.split(':')
    if s[0]=='null':
        return 0        #没有使用 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1        #15天内使用  1
    else:
        return -1       #15天后使用  -1
    


# In[22]:


#dataset3
coupon3=pd.read_csv(r'D:\O2O\features\coupon feature\coupon3_feature.csv')
merchant3=pd.read_csv(r'D:\O2O\features\merchant feature\merchant3_feature.csv')
user3=pd.read_csv(r'D:\O2O\features\user feature\user3_feature.csv')
user_merchant3=pd.read_csv(r'D:\O2O\features\user_merchant feature\user_merchant3.csv')
other_feature3=pd.read_csv(r'D:\O2O\features\other feature\other_feature3.csv')
user_coupon3=pd.read_csv(r'D:\O2O\features\user_coupon feature\user_coupon3.csv')
on_feature3=pd.read_csv(r'D:\O2O\features\on feature\on_feature3.csv')

dataset3=pd.merge(coupon3,merchant3,on='merchant_id',how='left')
dataset3=pd.merge(dataset3,user3,on='user_id',how='left')
dataset3=pd.merge(dataset3,user_merchant3,on=['user_id','merchant_id'],how='left')
dataset3=pd.merge(dataset3,other_feature3,on=['user_id','coupon_id','merchant_id','date_received'],how='left')
dataset3=pd.merge(dataset3,user_coupon3,on=['user_id','coupon_id'],how='left')
dataset3=pd.merge(dataset3,on_feature3,on='user_id',how='left')
dataset3.drop_duplicates(inplace=True)
print('dataset3:',dataset3.shape)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan,0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan,0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan,0)
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset3.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset3 = pd.concat([dataset3,weekday_dummies],axis=1)
dataset3.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
dataset3 = dataset3.replace('null',np.nan)
dataset3.to_csv(r'D:\O2O\dataset3.csv',index=None)
print('dataset3:',dataset3.shape)


# In[23]:


#dataset2
coupon2=pd.read_csv(r'D:\O2O\features\coupon feature\coupon2_feature.csv')
merchant2=pd.read_csv(r'D:\O2O\features\merchant feature\merchant2_feature.csv')
user2=pd.read_csv(r'D:\O2O\features\user feature\user2_feature.csv')
user_merchant2=pd.read_csv(r'D:\O2O\features\user_merchant feature\user_merchant2.csv')
other_feature2=pd.read_csv(r'D:\O2O\features\other feature\other_feature2.csv')
user_coupon2=pd.read_csv(r'D:\O2O\features\user_coupon feature\user_coupon2.csv')
on_feature2=pd.read_csv(r'D:\O2O\features\on feature\on_feature2.csv')

dataset2=pd.merge(coupon2,merchant2,on='merchant_id',how='left')
dataset2=pd.merge(dataset2,user2,on='user_id',how='left')
dataset2=pd.merge(dataset2,user_merchant2,on=['user_id','merchant_id'],how='left')
dataset2=pd.merge(dataset2,other_feature2,on=['user_id','coupon_id','merchant_id','date_received'],how='left')
dataset2=pd.merge(dataset2,user_coupon2,on=['user_id','coupon_id'],how='left')
dataset2=pd.merge(dataset2,on_feature2,on='user_id',how='left')
dataset2.drop_duplicates(inplace=True)
print('dataset2:',dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan,0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan,0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan,0)
dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2,weekday_dummies],axis=1)

dataset2=dataset2.replace(np.nan,'null')
dataset2['label'] = dataset2.date.astype('str') + ':' + dataset2.date_received.astype('str')
dataset2.label = dataset2.label.apply(get_label)
dataset2.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset2 = dataset2.replace('null',np.nan)
dataset2.to_csv(r'D:\O2O\dataset2.csv',index=None)
print('dataset2:',dataset2.shape)


# In[24]:


#dataset1
coupon1=pd.read_csv(r'D:\O2O\features\coupon feature\coupon1_feature.csv')
merchant1=pd.read_csv(r'D:\O2O\features\merchant feature\merchant1_feature.csv')
user1=pd.read_csv(r'D:\O2O\features\user feature\user1_feature.csv')
user_merchant1=pd.read_csv(r'D:\O2O\features\user_merchant feature\user_merchant1.csv')
other_feature1=pd.read_csv(r'D:\O2O\features\other feature\other_feature1.csv')
user_coupon1=pd.read_csv(r'D:\O2O\features\user_coupon feature\user_coupon1.csv')
on_feature1=pd.read_csv(r'D:\O2O\features\on feature\on_feature1.csv')

dataset1=pd.merge(coupon1,merchant1,on='merchant_id',how='left')
dataset1=pd.merge(dataset1,user1,on='user_id',how='left')
dataset1=pd.merge(dataset1,user_merchant1,on=['user_id','merchant_id'],how='left')
dataset1=pd.merge(dataset1,other_feature1,on=['user_id','coupon_id','merchant_id','date_received'],how='left')
dataset1=pd.merge(dataset1,user_coupon1,on=['user_id','coupon_id'],how='left')
dataset1=pd.merge(dataset1,on_feature1,on='user_id',how='left')
dataset1.drop_duplicates(inplace=True)
print('dataset1:',dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan,0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan,0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan,0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1,weekday_dummies],axis=1)

dataset1=dataset1.replace(np.nan,'null')
dataset1['label'] = dataset1.date.astype('str') + ':' + dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset1 = dataset1.replace('null',np.nan)
dataset1.to_csv(r'D:\O2O\dataset1.csv',index=None)
print('dataset1:',dataset1.shape)

