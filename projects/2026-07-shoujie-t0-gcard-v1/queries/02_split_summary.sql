-- 按 final_flag 查看样本量、时间范围和基础标签浓度。
select
  final_flag,
  min(sample_date) as min_mdl_dte,
  max(sample_date) as max_mdl_dte,
  min(sample_month) as min_ds,
  max(sample_month) as max_ds,
  count(1) as cnt,
  avg(cast(target as double)) as target_rate
from 
group by final_flag
order by final_flag;
