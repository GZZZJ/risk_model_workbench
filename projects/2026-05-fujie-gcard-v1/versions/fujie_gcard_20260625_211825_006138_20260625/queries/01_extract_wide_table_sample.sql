-- 复借G卡主模型从0重跑 - 宽表数据抽取
-- request_id: 20260625-1740-model-request
-- run_id: 20260625_211825_006138
-- description: 从 D01/D02 宽表抽取 10% 样本数据用于完整建模流程
-- sample_definition: 可经营、当前未逾期用户、重资产订单；标签为观察日30天内是否发起
--
-- 字段合同:
--   主键: uid, mdl_dte
--   时间字段: mdl_dte
--   分区字段: ds
--   标签字段: ftr_30d_ord_flag
--   切分字段: final_flag
--   训练集: DEV, OOS: DEV-OOS, OOT: OOT / OOT-OOS
SELECT *
FROM pdm_risk.pdm_risk_fujie_gcard_d01_d02_wide_feature_2837_num_v6_1
WHERE rand_flag1 < 0.1
