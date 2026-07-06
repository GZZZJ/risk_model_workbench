-- 复借G卡取样逻辑历史 SQL 留档
--
-- 用途：
-- - 保留复借 G 卡 v6_1 样本构造、切分、标签和历史基准分拼接的原始 SQL 证据。
-- - 帮助后续使用者理解当前项目样本口径来源和关键取舍。
--
-- 边界：
-- - 本文件位于 docs/ 下，仅作为 lineage/reference，不是 rmw 工作流可直接执行脚本。
-- - SQL 中包含 ${pdm_risk}、${bizdate} 等运行环境变量、历史表名和人工校验语句。
-- - 如需重新执行或产品化，应先迁移到受控 CLI/配置流程，并经过 SQL 审批。
--
-- 人类可读摘要见：docs/样本与取样口径.md
-- 历史最终基准样本表：pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1

--001. 加工随机观察日样本 hlm名单标签 
drop table if exists pdm_risk.pdm_risk_cus_label_df_gcard_v6_1;
create table if not exists pdm_risk.pdm_risk_cus_label_df_gcard_v6_1  AS 
select 
cus_grp.uid,
cus_grp.day,
-- cus_type_hlm_lv4,
--  when cus_grp.day<='2023-09-04' and hlm1.cus_type_hlm_lv4 like '%不可经营用户_暂时%' then '不可经营用户_暂时'
--     when cus_grp.day<='2023-09-04' and hlm1.cus_type_hlm_lv4 like '%不可经营用户_永久%' then '不可经营用户_永久'
--     when cus_grp.day<='2023-09-04' and hlm1.cus_type_hlm_lv4 like '%可经营用户%' then '可经营用户'
--     when cus_grp.day<='2023-09-04' and hlm1.cus_type_hlm_lv4 like '%重资产额度为0%' then '无重资产额度'
    case when  hlm2.cus_type_hlm_lv4 like '%不可经营用户_暂时%' then '不可经营用户_暂时'
    when  hlm2.cus_type_hlm_lv4 like '%不可经营用户_永久%' then '不可经营用户_永久'
    when  hlm2.cus_type_hlm_lv4 like '%可经营用户%' then '可经营用户'
    when  hlm2.cus_type_hlm_lv4 like '%重资产额度为0%' then '无重资产额度'
    else '其他' end as cus_type_hlm
from
(select 
    uid
    ,blue_customer_flag
    ,date(to_date(ds, 'yyyymmdd')) as day
    ,ds
from pdm_risk_ftr.blue_customer_group_df 
where ds in 
(    '20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130')
and blue_customer_flag in ('E2', 'B2', 'E3')
) cus_grp
-- left join 
--     (select * from pdm_risk.pdm_risk_hlm_cus_type_new_df 
--     where ds in ('20230731','20230831')
--     ) hlm1
--     on cus_grp.uid = hlm1.uid 
--     and date_format(last_day(add_months(date_format(to_date(cus_grp.ds,'yyyymmdd'),'yyyy-MM-dd'),-1)),'yyyyMMdd') = hlm1.ds
left JOIN 
    (select * from pdm_risk.pdm_risk_hlm_cus_type_new_df
    where ds in 
    (
        '20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130')
    ) hlm2
    on cus_grp.uid = hlm2.uid 
    and cus_grp.ds=hlm2.ds
;


-- 002. 取日更样本客群标签
drop table if exists ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp_v6_1;
create table ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp_v6_1 as
select a.*,b.cus_type_hlm
from(
select uid,
    blue_customer_flag,
    ds
from pdm_risk_ftr.blue_customer_group_df
where ds in ('20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130') 
) a
left join (
    select uid,cus_type_hlm,date_format(day,'yyyyMMdd') as ds,date_add(day,1) as mdl_dte
    -- from pdm_risk.pdm_risk_cus_label_df 
    from pdm_risk.pdm_risk_cus_label_df_gcard_v6_1
    where day in ('2025-06-01','2025-06-02','2025-06-05','2025-06-09','2025-06-23','2025-06-27',
    '2025-07-08','2025-07-12','2025-07-25','2025-07-28','2025-07-29','2025-07-30',
    '2025-08-07','2025-08-12','2025-08-13','2025-08-17','2025-08-18','2025-08-24',
    '2025-09-05','2025-09-12','2025-09-16','2025-09-18','2025-09-20','2025-09-23',
    '2025-10-01','2025-10-06','2025-10-10','2025-10-12','2025-10-23','2025-10-27',
    '2025-11-02','2025-11-03','2025-11-10','2025-11-13','2025-11-17','2025-11-18',
    '2025-12-03','2025-12-06','2025-12-11','2025-12-17','2025-12-23','2025-12-25',
    '2026-01-05','2026-01-07','2026-01-14','2026-01-26','2026-01-28','2026-01-30')
) b
on a.uid=b.uid and a.ds=b.ds
where b.cus_type_hlm='可经营用户'
;

-- select *
-- from vdm_risk_jupyter.vdm_risk_jupyter_gcard_all_sample_scr_ben where uid='0070c001-7124-4a20-9085-50c4ce9b8d39' and mdl_dte='2024-01-15' a
-- left join(
--     select uid,mdl_dte,count(1) as num from(
-- select uid,date_format(day,'yyyyMMdd') as ds,date_add(day,1) as mdl_dte
--        ,fq_diff_grp,mob_group,zc_level,gd_lmt_grp
-- from pdm_risk.pdm_risk_cus_label_df
-- where day in ('2023-08-02','2023-08-09','2023-08-10','2023-08-16','2023-08-22','2023-08-27',
-- '2023-09-04','2023-09-11','2023-09-16','2023-09-17','2023-09-23','2023-09-29',
-- '2023-10-02','2023-10-03','2023-10-06','2023-10-08','2023-10-11','2023-10-14',
-- '2023-11-06','2023-11-09','2023-11-13','2023-11-20','2023-11-22','2023-11-26',
-- '2023-12-11','2023-12-13','2023-12-17','2023-12-19','2023-12-23','2023-12-28',
-- '2024-01-02','2024-01-14','2024-01-21','2024-01-22','2024-01-30','2024-01-31',
-- '2024-02-04','2024-02-07','2024-02-08','2024-02-20','2024-02-26','2024-02-27',
-- '2024-03-02','2024-03-06','2024-03-13','2024-03-20','2024-03-25','2024-03-28',
-- '2024-04-09','2024-04-16','2024-04-17','2024-04-22','2024-04-23','2024-04-28',
-- '2024-05-03','2024-05-05','2024-05-08','2024-05-09','2024-05-14','2024-05-24')
-- and cus_type_hlm='可经营用户') group by uid,mdl_dte having num>1
-- ) b
-- on a.uid 

-- 003. 取历史发起、未来发起和发起间隔
drop table if exists ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp1_v6_1;
create table ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp1_v6_1 as
select a.uid,a.ds,
    -- 未来发
    min(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) else null end) as min_ftr_ord_dte_dif,
    max(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) else null end) as max_ftr_ord_dte_dif,
    avg(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) else null end) as avg_ftr_ord_dte_dif,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then prc_amt else 0 end) / sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then 1 else 0 end) as avg_ftr_ord_amt,
    min(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then prc_amt else null end) as min_ftr_ord_amt,
    max(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then prc_amt else null end) as max_ftr_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then prc_amt else 0 end) as sum_ftr_ord_amt,

    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) < 0 then 1 else 0 end) as ftr_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -7 and -1 then 1 else 0 end) as ftr_7d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -15 and -1 then 1 else 0 end) as ftr_15d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -30 and -1 then 1 else 0 end) as ftr_30d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -60 and -1 then 1 else 0 end) as ftr_60d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -90 and -1 then 1 else 0 end) as ftr_90d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -120 and -1 then 1 else 0 end) as ftr_120d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -150 and -1 then 1 else 0 end) as ftr_150d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -180 and -1 then 1 else 0 end) as ftr_180d_ord_cnt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -360 and -1 then 1 else 0 end) as ftr_360d_ord_cnt,

    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -7 and -1 then prc_amt else 0 end) as ftr_7d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -15 and -1 then prc_amt else 0 end) as ftr_15d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -30 and -1 then prc_amt else 0 end) as ftr_30d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -60 and -1 then prc_amt else 0 end) as ftr_60d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -90 and -1 then prc_amt else 0 end) as ftr_90d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -120 and -1 then prc_amt else 0 end) as ftr_120d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -150 and -1 then prc_amt else 0 end) as ftr_150d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -180 and -1 then prc_amt else 0 end) as ftr_180d_ord_amt,
    sum(case when DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), b.loan_dte) between -360 and -1 then prc_amt else 0 end) as ftr_360d_ord_amt

from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp_v6_1 as a
left join 
(
    -- 该表已不存在
    -- select order_no as ord_no,
    --     uid,
    --     date(apply_record_crt_time) as loan_dte,
    --     loan_success_flag,
    --     coalesce(loan_principal_amount,apply_amount) as prc_amt
    -- from ${ld_loan}.fct_txn_heavy_loancore_order_df
    -- where ds = '${bizdate}'
    -- and business_type in ('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN') -- 重资产 
    select order_no as ord_no,
      uid,
      date(apply_record_crt_time) as loan_dte,
      loan_success_flag,
      coalesce(loan_principal_amount,apply_amount) as prc_amt
    from dwt.dwt_heavy_order_df
    where ds = '${bizdate}'
    and business_type in ('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN')
    
) as b
on a.uid = b.uid
group by a.uid,a.ds;



-- 004.曾逾期和历史逾期
drop table if exists ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp2_v6_1;
create table ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp2_v6_1 as
select uid,ds,
    max(cur_ovd_day) as max_cur_ovd_day,
    max(his_ovd_day) as max_his_ovd_day
from
(
    select a.uid,a.ds,
    
        case when rep_tim is null or date(rep_tim) > to_date(to_date(a.ds,'yyyymmdd')) then datediff(date_add(to_date(a.ds,'yyyymmdd'), 1), due_date) 
            else 0 end cur_ovd_day, -- 当前逾期天数
        case when rep_tim is null or date(rep_tim) > to_date(to_date(a.ds,'yyyymmdd')) then datediff(date_add(to_date(a.ds,'yyyymmdd'), 1), due_date) 
            else datediff(date(rep_tim), due_date) end as his_ovd_day -- 历史逾期天数
        
    from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp_v6_1 as a
    left join
    (
        -- select order_no,
        --     uid,
        --     due_date,
        --     settle_up_time as rep_tim,
        --     crt_time
        -- from ${ld_loan}.fct_txn_heavy_loancore_stage_plan_df
        -- where ds = '${bizdate}'
        -- and stage_plan_status in ('U', 'O', 'C', 'S', 'X', 'E')
          select order_no,
            uid,
            due_date,
            settlement_time as rep_tim,
            create_time as crt_time
        from ${cdmx}.cdmx_fct_heavy_stage_plan_df
        where ds = '${bizdate}'
        and original_stage_plan_status in ('U', 'O', 'C', 'S', 'X', 'E')
    ) as b
    on a.uid = b.uid
    left join
    (
        -- select order_no,
        --        date(apply_record_crt_time) as loan_dte
        -- from ${ld_loan}.fct_txn_heavy_loancore_order_df
        -- where ds = '${bizdate}'
        select order_no,
        date(apply_record_crt_time) as loan_dte
        from dwt.dwt_heavy_order_df
        where ds = '${bizdate}'
    ) as c
    on b.order_no = c.order_no
    where c.loan_dte <= to_date(to_date(a.ds,'yyyymmdd'))
) as c
group by uid,ds;


-- 005. 流失时长
drop table if exists ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp3_v6_1;
create table ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp3_v6_1 as
select a.uid,a.ds,

min(case when rep_tim is null or date(rep_tim) > to_date(to_date(a.ds,'yyyymmdd')) then 0 else DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')), date(rep_tim)) end) as liushi_days,

min(
    case when business_type in('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN') and rep_tim is null then 0 
    when business_type in('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN') and date(rep_tim) > to_date(to_date(a.ds,'yyyymmdd')) then 0 
    when business_type in('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN') then DATEDIFF(to_date(to_date(a.ds,'yyyymmdd')),date(rep_tim)) end
) as liushi_days_heavy
    
from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp_v6_1 as a
left join
(
    -- select order_no,
    --     uid,
    --     due_date,
    --     settle_up_time as rep_tim,
    --     crt_time,
    --     business_type
    -- from ${ld_loan}.fct_txn_heavy_loancore_stage_plan_df
    -- where ds = '${bizdate}'
    -- and stage_plan_status in ('U', 'O', 'C', 'S', 'X', 'E')
      select order_no,
      uid,
      due_date,
      settlement_time as rep_tim,
      create_time as crt_time,
      original_biz_type as business_type
  from ${cdmx}.cdmx_fct_heavy_stage_plan_df
  where ds = '${bizdate}'
  and original_stage_plan_status in ('U', 'O', 'C', 'S', 'X', 'E')
) as b
on a.uid = b.uid
left join
(
    -- select order_no,
    --     date(apply_record_crt_time) as loan_dte
    -- from ${ld_loan}.fct_txn_heavy_loancore_order_df
    -- where ds = '${bizdate}'
    select order_no,
      date(apply_record_crt_time) as loan_dte
    from dwt.dwt_heavy_order_df
    where ds = '${bizdate}'
    
) as c
on b.order_no = c.order_no
where c.loan_dte <= to_date(to_date(a.ds,'yyyymmdd'))
group by a.uid,a.ds;

-- select ds,count(1) from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp group by ds order by ds
-- select ds,count(1) from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp1 group by ds order by ds
-- select ds,count(1) from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp2 group by ds order by ds
-- select ds,count(1) from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp3 group by ds order by ds
-- select ds,count(1) from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_nlz_df group by ds order by ds


-- 006.汇总
drop table if exists ${pdm_risk}.pdm_risk_dz_gscore_base_sample_nlz_df_v6_1;
create table ${pdm_risk}.pdm_risk_dz_gscore_base_sample_nlz_df_v6_1 as
select a.uid,
    date_add(to_date(a.ds, 'yyyymmdd'), 1) as mdl_dte,
    a.ds,
    a.blue_customer_flag,

    min_ftr_ord_dte_dif,
    max_ftr_ord_dte_dif,
    avg_ftr_ord_dte_dif,
    avg_ftr_ord_amt,
    min_ftr_ord_amt,
    max_ftr_ord_amt,
    sum_ftr_ord_amt,
    ftr_ord_cnt,
    ftr_7d_ord_cnt,
    ftr_15d_ord_cnt,
    ftr_30d_ord_cnt,
    ftr_60d_ord_cnt,
    ftr_90d_ord_cnt,
    ftr_120d_ord_cnt,
    ftr_150d_ord_cnt,
    ftr_180d_ord_cnt,
    ftr_360d_ord_cnt,

    ftr_7d_ord_amt,
    ftr_15d_ord_amt,
    ftr_30d_ord_amt,
    ftr_60d_ord_amt,
    ftr_90d_ord_amt,
    ftr_120d_ord_amt,
    ftr_150d_ord_amt,
    ftr_180d_ord_amt,
    ftr_360d_ord_amt,
    
    max_cur_ovd_day,
    max_his_ovd_day,
    liushi_days,
    liushi_days_heavy
from ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_head_table_nlz_tmp_v6_1 as a
left join ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp1_v6_1 as b
on a.uid = b.uid and a.ds=b.ds
left join ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp2_v6_1 as c
on a.uid = c.uid and a.ds=c.ds
left join ${pdm_risk}.pdm_risk_dz_gscore_base_sample_df_nlz_tmp3_v6_1 as d
on a.uid = d.uid and a.ds=d.ds;


--007.新增交易风险
drop table if exists pdm_risk.pdm_risk_nlz_newloan_flag_v6_1;
create table pdm_risk.pdm_risk_nlz_newloan_flag_v6_1 as 
select a.*

,case when prc_amt_xz_30d_1m >0 then  prc_amt_xz_30d_1m else 0 end as prc_amt_xz_30d_1m
,case when ovd_amt_xz_30d_1m >0 then  ovd_amt_xz_30d_1m else 0 end as ovd_amt_xz_30d_1m
,case when due_uid_xz_30d_1m >0 then  due_uid_xz_30d_1m else 0 end as due_uid_xz_30d_1m
,case when ovd_uid_xz_30d_1m >0 then  ovd_uid_xz_30d_1m else 0 end as ovd_uid_xz_30d_1m

,case when prc_amt_xz_30d_3m >0 then  prc_amt_xz_30d_3m else 0 end as prc_amt_xz_30d_3m
,case when ovd_amt_xz_30d_3m >0 then  ovd_amt_xz_30d_3m else 0 end as ovd_amt_xz_30d_3m
,case when due_uid_xz_30d_3m >0 then  due_uid_xz_30d_3m else 0 end as due_uid_xz_30d_3m
,case when ovd_uid_xz_30d_3m >0 then  ovd_uid_xz_30d_3m else 0 end as ovd_uid_xz_30d_3m

from pdm_risk.pdm_risk_dz_gscore_base_sample_nlz_df_v6_1 a
left JOIN
(
    select a.uid
        , a.repay_day
    
        , sum(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,2) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,1))>=29 then b.schedule_pay_principal else 0 end ) as prc_amt_xz_30d_1m  --未来一个月的新增交易的1期30+应还金额
        , sum(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,2) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,1))>=29 then b.overdue_30d_unpaid_principal else 0 end ) as ovd_amt_xz_30d_1m --未来一个月的新增交易的1期30+逾期金额
        , max(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,2) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,1))>=29 and b.schedule_pay_principal>0 then 1 else 0 end ) as due_uid_xz_30d_1m --未来一个月新增交易的1期30+应还人数
        , max(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,2) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,1))>=29 and b.overdue_30d_unpaid_principal>0 then 1 else 0 end) as ovd_uid_xz_30d_1m --未来一个月新增交易的1期30+逾期人数

        , sum(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,4) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,3))>=29 then b.schedule_pay_principal else 0 end ) as prc_amt_xz_30d_3m  --未来一个月的新增交易的3期30+应还金额
        , sum(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,4) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,3))>=29 then b.overdue_30d_unpaid_principal else 0 end ) as ovd_amt_xz_30d_3m --未来一个月的新增交易的3期30+逾期金额
        , max(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,4) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,3))>=29 and b.schedule_pay_principal>0 then 1 else 0 end ) as due_uid_xz_30d_3m --未来一个月新增交易的3期30+应还人数
        , max(case when b.loan_date>a.repay_day and b.loan_date<=add_months(a.repay_day,1) and b.due_date<=add_months(a.repay_day,4) and datediff(to_date(to_date(b.ds,'yyyymmdd')),add_months(loan_date,3))>=29 and b.overdue_30d_unpaid_principal>0 then 1 else 0 end) as ovd_uid_xz_30d_3m --未来一个月新增交易的3期30+逾期人数

 from 
    (
        select uid,date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd') as repay_day 
        from pdm_risk.pdm_risk_dz_gscore_base_sample_nlz_df_v6_1
        group by uid,date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd')
    ) a
    left join
    (
        select uid, order_no, stage_plan_no, loan_date, due_date, settle_up_time, current_overdue_days,
        schedule_pay_principal, overdue_1d_unpaid_principal, overdue_4d_unpaid_principal,
        overdue_10d_unpaid_principal, overdue_30d_unpaid_principal, ds
        from dcube.dcube_risk_stage_plan_overdue_info_df
        where ds ='${bizdate}'
        and business_type in ('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN')--CASH-现金分期、BALANCE_TRANSFER-信用卡代还、ENJOY_PAY-还享花、HUGE_LOAN-大额账户
    ) b
    on a.uid = b.uid
    where (to_date(b.settle_up_time) > a.repay_day or b.settle_up_time is null)
    and due_date>a.repay_day
    group by a.uid, a.repay_day 

) t1 on a.uid = t1.uid and date_format(to_date(a.ds,'yyyymmdd'),'yyyy-MM-dd') = t1.repay_day
;


-- --008. uid维度去重
-- drop table if exists pdm_risk.pdm_risk_gcard_base_sample_uid_unique;
-- create table pdm_risk.pdm_risk_gcard_base_sample_uid_unique as
-- with unique_uid as(
--     select uid,ds 
--     from(
--             select uid,ds,row_number() over(partition by uid order by rand(2333) asc) as rn
--             from pdm_risk.pdm_risk_nlz_newloan_flag
--             where max_cur_ovd_day <= 0
--         ) 
--     where rn=1
-- )
-- select a.uid,a.ds,(uid|ds|rn)?+.+,case when b.uid is null then 'untrain' else 'train' end as sample_flag
-- from(
--     select *
--     from pdm_risk.pdm_risk_nlz_newloan_flag
--     where max_cur_ovd_day <= 0
-- ) a
-- left join (
--     select aa.uid,aa.ds,case when bb.uid is null then 'old' else 'new' end as uid_tag 
--     from unique_uid aa
--     left join(
--         select uid,ds 
--         from unique_uid
--         where uid in (select uid from pdm_risk.pdm_risk_nlz_newloan_flag where ds>='20240401' and max_cur_ovd_day <= 0 )
--         and uid not in (select uid from pdm_risk.pdm_risk_nlz_newloan_flag where ds<='20240331' and max_cur_ovd_day <= 0 )  
--         )bb
--     on aa.uid=bb.uid and aa.ds=bb.ds
--     )b 
-- on a.uid=b.uid and a.ds=b.ds
-- ;
-- create table pdm_risk.pdm_risk_gcard_base_sample_all as
-- select uid,date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd') as mdl_dte--,(uid|mdl_dte)?+.+
-- from pdm_risk.pdm_risk_nlz_newloan_flag
-- where max_cur_ovd_day <= 0


drop table if exists pdm_risk.pdm_risk_gcard_base_sample_all_v6_1;
create table pdm_risk.pdm_risk_gcard_base_sample_all_v6_1 as
select uid,date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd') as mdl_dte--,(uid|mdl_dte)?+.+
from pdm_risk.pdm_risk_nlz_newloan_flag_v6_1
where max_cur_ovd_day <= 0
;
select substr(mdl_dte,1,7),count(1) from pdm_risk.pdm_risk_gcard_base_sample_all group by substr(mdl_dte,1,7) order by substr(mdl_dte,1,7)
select substr(mdl_dte,1,7),count(1) from pdm_risk.pdm_risk_gcard_base_sample_all_v6_1 group by substr(mdl_dte,1,7) order by substr(mdl_dte,1,7)

-- 可经营用户----------------------------
drop table if exists pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1;
-- create table pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1 as
-- select a.uid,a.ds
-- -- ,(uid|ds|rn)?+.+
-- ,a.* except(uid, ds),
-- ,case when b.uid is null then 'untrain' else 'train' end as sample_flag
-- from(   select * 
--         from pdm_risk.pdm_risk_nlz_newloan_flag_v6_1
--         where max_cur_ovd_day <= 0) a 
-- left join(
-- select uid,ds
-- from(
--         select uid,ds,row_number() over(partition by ds order by rand(2555) asc) as rn
--         from pdm_risk.pdm_risk_nlz_newloan_flag_v6_1
--         where max_cur_ovd_day <= 0
--     ) 
-- where rn<=100000) b
-- on a.uid=b.uid and a.ds=b.ds
-- ;

create table pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1 as
select 
    a.*,
    case when b.uid is null then 'untrain' else 'train' end as sample_flag
from (
    select * 
    from pdm_risk.pdm_risk_nlz_newloan_flag_v6_1
    where max_cur_ovd_day <= 0
) a 
left join (
    select uid, ds
    from (
        select 
            uid,
            ds,
            row_number() over(partition by ds order by rand(2555) asc) as rn
        from pdm_risk.pdm_risk_nlz_newloan_flag_v6_1
        where max_cur_ovd_day <= 0
    ) t
    where rn <= 100000
) b
on a.uid = b.uid 
and a.ds = b.ds
;

select uid,ds,count(1) as num from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1 group by uid,ds having num>1;
select * from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1 limit 1000

-- 长流失高风险
-- 可用额度低
-- 新uid
drop table if exists pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_01_v6_1;
create table pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_01_v6_1 as
-- select uid,mdl_dte,count(1) as num from(
select * from(
select *,case when sample_flag='train' and time_window='DEV' then 'DEV'
        when sample_flag='train' and time_window='OOT' then 'OOT'
        when all_sample_tag='OOS-untrain' and time_window='DEV' then 'DEV-OOS'
        when all_sample_tag='OOS-untrain' and time_window='OOT' then 'OOT-OOS'
        else null
        end as final_flag
from(

        select a.*,
            case when b.sample_tag is null then 'INS-train'  --train
            when b.sample_tag ='OOS-untrain' then 'OOS-untrain'  -- untrain中新uid
            else 'INS' end as all_sample_tag
            ,case when a.mdl_dte<'2025-12-01' then 'DEV' else 'OOT' end as time_window
            ,case when c.uid is not null then 'due' else 'un_due' end as due_date_flag
            ,c.due_pay_amt_all
        from (select * from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1 where max_cur_ovd_day <= 0) as a
        left join
        (
                select aa.*,case when bb.uid is null then 'OOS-untrain' else 'INS' end as sample_tag
                from (select *
                        from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1
                        where sample_flag='untrain' and max_cur_ovd_day <= 0
                    ) aa
                left join (
                        select uid
                        from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_v6_1
                        where sample_flag='train' and max_cur_ovd_day <= 0
                        group by uid
                        ) bb
                on aa.uid = bb.uid
        ) as b
        on a.uid = b.uid and a.mdl_dte=b.mdl_dte

        left join(

            -- select uid,due_date,sum(loan_original_principal) as due_pay_amt_all
            -- from ld_loan.fct_txn_heavy_loancore_stage_plan_df
            -- where ds ='${bizdate}'
            -- and business_type in ('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN')
            -- and stage_plan_status in ('U', 'O', 'C', 'S', 'X', 'E') 
            -- group by uid,due_date 
             select uid,
                due_date,
                sum(original_loan_principal) as due_pay_amt_all
            from cdmx.cdmx_fct_heavy_stage_plan_df
            where ds = '${bizdate}'
            and original_biz_type in ('BUSINESS_LOAN','BALANCE_TRANSFER','CASH','HB_IMPREST','CREDIT_TRANSACTION','HUGE_LOAN')
            and original_stage_plan_status in ('U', 'O', 'C', 'S', 'X', 'E')
            group by uid, due_date
        ) c
        on a.uid=c.uid and a.mdl_dte=c.due_date)

        -- group by sample_flag,all_sample_tag,time_window
-- where sample_flag ='train' or all_sample_tag='OOS_untrain'
)where final_flag is not null
--)group by uid,mdl_dte having num>1
;



drop table if exists pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_v6_1;
create table pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_v6_1 as
select uid, ds, mdl_dte, blue_customer_flag
    , ftr_7d_ord_cnt, ftr_15d_ord_cnt, ftr_30d_ord_cnt
	, ftr_60d_ord_cnt, ftr_90d_ord_cnt, ftr_120d_ord_cnt, ftr_150d_ord_cnt, ftr_180d_ord_cnt
	, ftr_360d_ord_cnt, ftr_7d_ord_amt, ftr_15d_ord_amt, ftr_30d_ord_amt, ftr_60d_ord_amt
	, ftr_90d_ord_amt, ftr_120d_ord_amt, ftr_150d_ord_amt, ftr_180d_ord_amt, ftr_360d_ord_amt
	, max_cur_ovd_day, max_his_ovd_day, liushi_days, liushi_days_heavy, prc_amt_xz_30d_1m
	, ovd_amt_xz_30d_1m, due_uid_xz_30d_1m, ovd_uid_xz_30d_1m, prc_amt_xz_30d_3m, ovd_amt_xz_30d_3m
	, due_uid_xz_30d_3m, ovd_uid_xz_30d_3m, sample_flag, all_sample_tag, time_window
	, due_date_flag, due_pay_amt_all, final_flag
from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_01_v6_1
where final_flag in ('DEV','OOT')

union all

-- 未参与训练样本也需要抽样，不抽样日更样本量太大，jupyter运行不动
select  uid, ds, mdl_dte, blue_customer_flag
    , ftr_7d_ord_cnt, ftr_15d_ord_cnt, ftr_30d_ord_cnt
	, ftr_60d_ord_cnt, ftr_90d_ord_cnt, ftr_120d_ord_cnt, ftr_150d_ord_cnt, ftr_180d_ord_cnt
	, ftr_360d_ord_cnt, ftr_7d_ord_amt, ftr_15d_ord_amt, ftr_30d_ord_amt, ftr_60d_ord_amt
	, ftr_90d_ord_amt, ftr_120d_ord_amt, ftr_150d_ord_amt, ftr_180d_ord_amt, ftr_360d_ord_amt
	, max_cur_ovd_day, max_his_ovd_day, liushi_days, liushi_days_heavy, prc_amt_xz_30d_1m
	, ovd_amt_xz_30d_1m, due_uid_xz_30d_1m, ovd_uid_xz_30d_1m, prc_amt_xz_30d_3m, ovd_amt_xz_30d_3m
	, due_uid_xz_30d_3m, ovd_uid_xz_30d_3m, sample_flag, all_sample_tag, time_window
	, due_date_flag, due_pay_amt_all, final_flag
from(
select *,row_number() over(partition by ds order by rand(255) asc) as rn
from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_01_v6_1
where final_flag in ('DEV-OOS','OOT-OOS')
) where rn<=100000
;
select uid,mdl_dte,count(1) as num from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva group by uid,mdl_dte having num>1
select final_flag from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_v6_1 group by final_flag
-- 验证uid独立
select * from (select * from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva where final_flag in ('DEV','OOT')) a 
inner join (select * from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva where final_flag in ('OOT-OOS')) b on a.uid=b.uid

-- benchmark
drop table if exists pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1;
create table pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_ben_v6_1 as
select a.uid,a.mdl_dte,a.ds,a.blue_customer_flag
        ,case when a.ftr_30d_ord_cnt>0 then 1 else 0 end as ftr_30d_ord_flag
        ,ftr_30d_ord_amt
        ,prc_amt_xz_30d_3m, ovd_amt_xz_30d_3m
        ,liushi_days,due_date_flag, final_flag
        ,fq_diff_grp
        ,mob_group,zc_level,gd_lmt_grp
       ,b.mdl_adj_prb as gcard_v2
       ,c.score as gcard_v4
       ,d.sco as gcard_v5
       ,f.score as gcard_v6
       ,rand(1) as rand_flag0
       ,rand(1) as rand_flag1
       ,rand(1) as rand_flag2
       ,rand(1) as rand_flag3
       ,rand(1) as rand_flag4
       ,rand(1) as rand_flag5
from pdm_risk.pdm_risk_gcard_base_sample_uid_ds_eva_v6_1 a
left join (
select uid,date_add(date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd'),1) as mdl_dte,ds,cus_flg,mdl_typ1,mdl_adj_prb
from pdm_risk.pdm_risk_dz_risk_fq_model_alingment_summary_table_df   --gcrad v2
where ds in
('20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130') --limit 10
) b
on a.uid=b.uid and a.mdl_dte=b.mdl_dte
-- V4
left join(
select uid,mdl_dte,reloan_user_group_day,score
from --dwa_risk_model.dwa_risk_model_dz_subnew_30_100d_fq_model_202309_v4_feature_and_score_df
dwa_risk_model.off_dp_dz_old_100d_fq_model_202309_v4_feature_and_score_df  --gcard v4
where ds in
('20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130')
) c
on a.uid=c.uid and a.mdl_dte=c.mdl_dte

-- V5 长流失意愿 结清12+以上
left join(
select uid,date_add(date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd'),1) as mdl_dte,grp,sco
from dwa_risk_model.dz_cls_loss_desire_model_features_and_score_df  -- gcard_v5
where ds in
('20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130')
) d
on a.uid=d.uid and a.mdl_dte=d.mdl_dte

left join(
select uid,date_format(day,'yyyyMMdd') as ds,date_add(day,1) as mdl_dte
       ,fq_diff_grp,mob_group,zc_level,gd_lmt_grp
from pdm_risk.pdm_risk_cus_label_df
where day in ('2025-06-01','2025-06-02','2025-06-05','2025-06-09','2025-06-23','2025-06-27',
    '2025-07-08','2025-07-12','2025-07-25','2025-07-28','2025-07-29','2025-07-30',
    '2025-08-07','2025-08-12','2025-08-13','2025-08-17','2025-08-18','2025-08-24',
    '2025-09-05','2025-09-12','2025-09-16','2025-09-18','2025-09-20','2025-09-23',
    '2025-10-01','2025-10-06','2025-10-10','2025-10-12','2025-10-23','2025-10-27',
    '2025-11-02','2025-11-03','2025-11-10','2025-11-13','2025-11-17','2025-11-18',
    '2025-12-03','2025-12-06','2025-12-11','2025-12-17','2025-12-23','2025-12-25',
    '2026-01-05','2026-01-07','2026-01-14','2026-01-26','2026-01-28','2026-01-30')
and cus_type_hlm='可经营用户'
) e
on a.uid=e.uid and a.mdl_dte=e.mdl_dte

left join(
    select uid,date_add(date_format(to_date(ds,'yyyymmdd'),'yyyy-MM-dd'),1) as mdl_dte,dz_gscore_fq30_202408_v6_score as score
from ads_app_modelplt_off.ads_app_modelplt_off_dz_gscore_fq30_202408_v6_di -- gcard_v6
where ds in
('20250601','20250602','20250605','20250609','20250623','20250627',
    '20250708','20250712','20250725','20250728','20250729','20250730',
    '20250807','20250812','20250813','20250817','20250818','20250824',
    '20250905','20250912','20250916','20250918','20250920','20250923',
    '20251001','20251006','20251010','20251012','20251023','20251027',
    '20251102','20251103','20251110','20251113','20251117','20251118',
    '20251203','20251206','20251211','20251217','20251223','20251225',
    '20260105','20260107','20260114','20260126','20260128','20260130')
) f
on a.uid=f.uid and a.mdl_dte=f.mdl_dte

;
