(function initPreviewBuilder(root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (root) root.RmwPreviewRequestBuilder = api;
  if (typeof document !== "undefined") api.boot();
})(typeof window !== "undefined" ? window : globalThis, function previewBuilderFactory() {
  const STORAGE_KEY = "risk_model_request_builder_preview";
  const STAGES = [
    "sample_check",
    "feature_metadata",
    "feature_prescreen",
    "build_wide_sql",
    "feature_refine",
    "train_baseline",
    "evaluate",
    "compare",
    "report",
  ];

  const WORKFLOW_STAGE_SCOPE = {
    full_modeling: STAGES,
    feature_selection: ["feature_metadata", "feature_prescreen", "build_wide_sql", "feature_refine"],
    train_baseline: ["train_baseline"],
    challenger_evaluation: ["evaluate", "compare"],
  };

  const TASK_MODE_LABELS = {
    full_modeling: "完整建模",
    feature_selection: "特征筛选",
    train_baseline: "训练基线",
    challenger_evaluation: "挑战者评估",
  };

  const BUSINESS_DOMAIN_LABELS = {
    acquisition: "获客",
    preloan: "贷前",
    inloan_risk: "贷中风险",
    inloan_operation: "贷中经营",
  };

  const PROFILE_LABELS = {
    fujie_gcard_main_lgbm: "复借 G 卡主模型",
    inloan_operation: "贷中经营通用",
    inloan_behavior_card: "贷中行为卡",
    preloan_credit_card: "贷前信用卡",
    acquisition_quality: "获客质量",
    acquisition_conversion: "获客转化",
    feature_gain_eval: "特征增益评估",
    credit_product_eval: "资信产品评估",
  };

  const STAGE_LABELS = {
    sample_check: "样本检查",
    feature_metadata: "特征元数据",
    feature_prescreen: "特征初筛",
    build_wide_sql: "宽表 SQL",
    feature_refine: "特征精筛",
    train_baseline: "训练",
    evaluate: "评估",
    compare: "对比",
    report: "报告",
  };

  const STEP_LABELS = {
    field_contract: "字段契约",
    key_uniqueness: "主键去重",
    monthly_label_distribution: "月度标签分布",
    segment_distribution: "分客群分布",
    account_status_distribution: "账期分布",
    channel_distribution: "渠道统计",
    dual_target_split: "双 Y 标拆分",
    credit_product_coverage: "资信覆盖",
    feature_metadata_export: "元数据导出",
    feature_quality_prescreen: "特征质量初筛",
    wide_sql_generation: "宽表 SQL 生成",
    sql_review_gate: "SQL Review Gate",
    feature_availability_filter: "可用性过滤",
    missing_rate_filter: "缺失率过滤",
    constant_value_filter: "恒一值过滤",
    iv_filter: "IV 过滤",
    psi_filter: "PSI 过滤",
    correlation_dedup: "相关性去重",
    random_noise_importance: "随机噪声重要性",
    null_importance_filter: "空标签重要性",
    baseline_importance_filter: "基线模型重要性",
    lightgbm_binary_training: "LightGBM 二分类训练",
    scale_pos_weight: "正负样本权重",
    teacher_student_distillation: "蒸馏训练",
    hier_ranknet_training: "HierRankNet 训练",
    auc_ks: "AUC / KS",
    decile_lift: "十分箱 Lift",
    monthly_stability: "月度稳定性",
    score_psi: "分数 PSI",
    score_psi_bin_detail: "分数 PSI 分箱明细",
    segment_metrics: "分客群评估",
    intent_zc_cross_risk: "意愿 x 资质交叉风险",
    intent_risk_segmented: "意愿矩阵分客群",
    cross_gain_matrix: "交叉增益矩阵",
    roll_rate_analysis: "滚动率分析",
    channel_metrics: "分渠道评估",
    dual_model_synergy: "双模型协同",
    sub_funnel_metrics: "子漏斗评估",
    credit_product_standalone_eval: "资信产品单点评估",
    credit_product_fusion_eval: "资信产品融合评估",
    feature_gain_summary: "特征增益汇总",
    champion_challenger: "Champion / Challenger 对比",
    model_report: "模型报告",
    credit_product_report: "资信产品报告",
  };

  const PROFILE_BUSINESS_DOMAINS = {
    acquisition_quality: "acquisition",
    acquisition_conversion: "acquisition",
    preloan_credit_card: "preloan",
    credit_product_eval: "preloan",
    inloan_behavior_card: "inloan_risk",
    feature_gain_eval: "inloan_operation",
    inloan_operation: "inloan_operation",
    fujie_gcard_main_lgbm: "inloan_operation",
  };

  const PROJECT_CONTRACT = {
    project: "2026-05-fujie-gcard-v1",
    source_table: "ads_app_off_feature.ds29531_backtrack_fj_gcard_model_v6_1_sample",
    feature_location: "configs/feature_tables.txt",
    target_column: "ftr_30d_ord_flag",
    id_columns: ["uid", "mdl_dte"],
    time_column: "mdl_dte",
    period_column: "ds",
    split_column: "final_flag",
    dev_values: ["DEV"],
    oos_values: ["DEV-OOS"],
    oot_values: ["OOT", "OOT-OOS"],
    sample_definition: "可经营、当前未逾期用户、重资产订单；标签为观察日30天内是否发起。",
    champions: ["gcard_v2", "gcard_v4", "gcard_v5", "gcard_v6"],
    risk_profile_dimensions: ["blue_customer_flag", "zc_level"],
  };

  const PROFILE_STAGE_STEPS = {
    fujie_gcard_main_lgbm: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution"],
      feature_metadata: ["feature_metadata_export"],
      feature_prescreen: ["feature_quality_prescreen"],
      build_wide_sql: ["wide_sql_generation", "sql_review_gate"],
      feature_refine: [
        "feature_availability_filter",
        "missing_rate_filter",
        "constant_value_filter",
        "iv_filter",
        "correlation_dedup",
        "random_noise_importance",
        "null_importance_filter",
        "baseline_importance_filter",
      ],
      train_baseline: ["lightgbm_binary_training"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "segment_metrics", "intent_zc_cross_risk"],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    inloan_operation: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution"],
      feature_refine: [
        "feature_availability_filter",
        "missing_rate_filter",
        "constant_value_filter",
        "iv_filter",
        "correlation_dedup",
        "random_noise_importance",
        "null_importance_filter",
        "baseline_importance_filter",
      ],
      train_baseline: ["lightgbm_binary_training"],
      evaluate: [
        "auc_ks",
        "decile_lift",
        "monthly_stability",
        "score_psi",
        "score_psi_bin_detail",
        "segment_metrics",
        "intent_zc_cross_risk",
        "intent_risk_segmented",
        "cross_gain_matrix",
        "feature_gain_summary",
      ],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    inloan_behavior_card: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution", "account_status_distribution"],
      feature_refine: [
        "feature_availability_filter",
        "missing_rate_filter",
        "constant_value_filter",
        "iv_filter",
        "psi_filter",
        "correlation_dedup",
        "baseline_importance_filter",
      ],
      train_baseline: ["lightgbm_binary_training", "scale_pos_weight"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "cross_gain_matrix", "roll_rate_analysis"],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    preloan_credit_card: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution"],
      feature_refine: ["feature_availability_filter", "missing_rate_filter", "constant_value_filter", "iv_filter", "psi_filter", "baseline_importance_filter"],
      train_baseline: ["lightgbm_binary_training"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "cross_gain_matrix"],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    acquisition_quality: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution", "channel_distribution", "dual_target_split"],
      feature_refine: ["feature_availability_filter", "missing_rate_filter", "constant_value_filter", "iv_filter", "psi_filter", "baseline_importance_filter"],
      train_baseline: ["lightgbm_binary_training", "teacher_student_distillation"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "channel_metrics", "dual_model_synergy"],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    acquisition_conversion: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution", "channel_distribution", "dual_target_split"],
      feature_refine: ["feature_availability_filter", "missing_rate_filter", "constant_value_filter", "iv_filter", "psi_filter", "baseline_importance_filter"],
      train_baseline: ["lightgbm_binary_training", "hier_ranknet_training"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "channel_metrics", "sub_funnel_metrics", "dual_model_synergy"],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    feature_gain_eval: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution"],
      feature_refine: ["feature_availability_filter", "missing_rate_filter", "constant_value_filter", "iv_filter", "psi_filter", "baseline_importance_filter"],
      train_baseline: ["lightgbm_binary_training"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "cross_gain_matrix", "feature_gain_summary"],
      compare: ["champion_challenger"],
      report: ["model_report"],
    },
    credit_product_eval: {
      sample_check: ["field_contract", "key_uniqueness", "monthly_label_distribution", "credit_product_coverage"],
      train_baseline: ["lightgbm_binary_training"],
      evaluate: ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "credit_product_standalone_eval", "credit_product_fusion_eval"],
      compare: ["champion_challenger"],
      report: ["credit_product_report"],
    },
  };

  const DEFAULTS = {
    request_id: "",
    title: "复借 G 卡主模型从0重跑",
    owner: "辜子骏",
    workflow: "full_modeling",
    business_domain: "inloan_operation",
    scenario_profile: "fujie_gcard_main_lgbm",
    contract_mode: "inherit",
    data_source_mode: "remote_table",
    sample_location: PROJECT_CONTRACT.source_table,
    feature_location: PROJECT_CONTRACT.feature_location,
    target_column: PROJECT_CONTRACT.target_column,
    id_columns: PROJECT_CONTRACT.id_columns.join(", "),
    time_column: PROJECT_CONTRACT.time_column,
    period_column: PROJECT_CONTRACT.period_column,
    split_column: PROJECT_CONTRACT.split_column,
    dev_values: PROJECT_CONTRACT.dev_values.join(", "),
    oos_values: PROJECT_CONTRACT.oos_values.join(", "),
    oot_values: PROJECT_CONTRACT.oot_values.join(", "),
    sample_definition: PROJECT_CONTRACT.sample_definition,
    objective:
      "基于最新复借G卡宽表数据口径，重新执行样本检查、特征收敛、LightGBM训练、评估、历史分对比和报告生成，形成可审计的新版本产物。",
    experiment_name: "main_lgbm",
    experiment_method: "lightgbm",
    experiment_description: "训练全客群 LightGBM 主模型，分客群只用于评估切片，不单独训练分客群模型。",
    candidate_targets: PROJECT_CONTRACT.target_column,
    sample_variants: "all, e2e3, b2",
    champions: PROJECT_CONTRACT.champions.join(", "),
    risk_profile_dimensions: PROJECT_CONTRACT.risk_profile_dimensions.join(", "),
    comparison_dimensions: ["split", "month", "segment", "decile"],
    metrics: ["auc", "ks", "decile_lift", "ranking_inversion", "psi"],
    report_sections: [
      "sample_overview",
      "feature_screening",
      "modeling_plan",
      "top_features",
      "model_performance",
      "champion_comparison",
      "risk_profile",
      "next_action",
    ],
    report_outputs: ["model_report.md", "model_card.md", "executive_summary.md"],
    extra_notes:
      "训练前必须确认样本检查、特征清单和 SQL 审批状态；如缺少真实训练数据或特征清单，应停止并标记原因，不得继续产出伪完成结果。\n\n真实 DP 拉数前必须先生成 SQL 并获得明确审批。不得把导入产物或占位结果当作本轮重跑证据。",
    missing_rate_threshold: "0.9",
    constant_max_unique_values: "1",
    iv_min: "0.005",
    psi_max: "0.2",
    correlation_method: "spearman",
    correlation_max_abs: "0.8",
    score_psi_warn: "",
    sql_block_high_risk: true,
  };

  let activeStep = "task";
  let requestId = "";
  let toastTimer = null;

  function pad2(value) {
    return String(value).padStart(2, "0");
  }

  function generateRequestId(date = new Date()) {
    return `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}-${pad2(date.getHours())}${pad2(date.getMinutes())}-model-request`;
  }

  function ensureRequestId(value) {
    requestId = String(value || requestId || generateRequestId()).trim();
    return requestId;
  }

  function list(value) {
    if (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
    return String(value || "")
      .split(/[,\n]/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  function yamlScalar(value) {
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return String(value);
    const text = String(value ?? "");
    if (!text) return '""';
    if (/^[A-Za-z0-9_.:/-]+$/.test(text)) return text;
    return JSON.stringify(text);
  }

  function yamlList(items, indent = 0) {
    const pad = " ".repeat(indent);
    const values = list(items);
    if (!values.length) return `${pad}[]`;
    return values.map((item) => `${pad}- ${yamlScalar(item)}`).join("\n");
  }

  function yamlMapOfLists(map, indent = 0) {
    const pad = " ".repeat(indent);
    const lines = [];
    Object.entries(map).forEach(([key, values]) => {
      const items = list(values);
      if (!items.length) return;
      lines.push(`${pad}${key}:`);
      lines.push(yamlList(items, indent + 2));
    });
    return lines.length ? lines.join("\n") : `${pad}{}`;
  }

  function yamlMapOfMaps(map, indent = 0) {
    const pad = " ".repeat(indent);
    const lines = [];
    Object.entries(map).forEach(([key, values]) => {
      const entries = Object.entries(values || {}).filter(([, value]) => value !== "" && value !== null && value !== undefined);
      if (!entries.length) return;
      lines.push(`${pad}${key}:`);
      entries.forEach(([entryKey, value]) => {
        lines.push(`${pad}  ${entryKey}: ${yamlScalar(value)}`);
      });
    });
    return lines.length ? lines.join("\n") : `${pad}{}`;
  }

  function multiline(value) {
    const text = String(value || "").trim();
    return text || "待补充。";
  }

  function profileSteps(profile) {
    return PROFILE_STAGE_STEPS[profile] || PROFILE_STAGE_STEPS.preloan_credit_card;
  }

  function contractForState(state) {
    const inherited = {
      sample_location: state.sample_location || PROJECT_CONTRACT.source_table,
      feature_location: PROJECT_CONTRACT.feature_location,
      target_column: PROJECT_CONTRACT.target_column,
      id_columns: PROJECT_CONTRACT.id_columns,
      time_column: PROJECT_CONTRACT.time_column,
      period_column: PROJECT_CONTRACT.period_column,
      split_column: PROJECT_CONTRACT.split_column,
      dev_values: PROJECT_CONTRACT.dev_values,
      oos_values: PROJECT_CONTRACT.oos_values,
      oot_values: PROJECT_CONTRACT.oot_values,
      sample_definition: state.sample_definition || PROJECT_CONTRACT.sample_definition,
    };

    if (state.contract_mode !== "override") return inherited;

    return {
      sample_location: state.sample_location || PROJECT_CONTRACT.source_table,
      feature_location: state.feature_location || PROJECT_CONTRACT.feature_location,
      target_column: state.target_column || PROJECT_CONTRACT.target_column,
      id_columns: list(state.id_columns).length ? list(state.id_columns) : PROJECT_CONTRACT.id_columns,
      time_column: state.time_column || PROJECT_CONTRACT.time_column,
      period_column: state.period_column || PROJECT_CONTRACT.period_column,
      split_column: state.split_column || PROJECT_CONTRACT.split_column,
      dev_values: list(state.dev_values).length ? list(state.dev_values) : PROJECT_CONTRACT.dev_values,
      oos_values: list(state.oos_values).length ? list(state.oos_values) : PROJECT_CONTRACT.oos_values,
      oot_values: list(state.oot_values).length ? list(state.oot_values) : PROJECT_CONTRACT.oot_values,
      sample_definition: state.sample_definition || PROJECT_CONTRACT.sample_definition,
    };
  }

  function stageStepsForState(state) {
    const scope = new Set(WORKFLOW_STAGE_SCOPE[state.workflow] || WORKFLOW_STAGE_SCOPE.full_modeling);
    const profile = profileSteps(state.scenario_profile);
    const result = {};
    STAGES.forEach((stage) => {
      if (!scope.has(stage)) return;
      const values = profile[stage] || [];
      if (values.length) result[stage] = [...values];
    });
    return result;
  }

  function selectedStepSet(stageSteps) {
    return new Set(Object.values(stageSteps).flat());
  }

  function addParam(target, selected, step, key, value) {
    if (!selected.has(step) || value === "" || value === null || value === undefined) return;
    target[step] = target[step] || {};
    target[step][key] = value;
  }

  function stepParamsForState(state) {
    const selected = selectedStepSet(stageStepsForState(state));
    const params = {};
    addParam(params, selected, "feature_quality_prescreen", "require_sql_approval", true);
    addParam(params, selected, "sql_review_gate", "block_on_high_risk", Boolean(state.sql_block_high_risk));
    addParam(params, selected, "missing_rate_filter", "threshold", state.missing_rate_threshold);
    addParam(params, selected, "constant_value_filter", "max_unique_values", state.constant_max_unique_values);
    addParam(params, selected, "iv_filter", "min_iv", state.iv_min);
    addParam(params, selected, "psi_filter", "max_psi", state.psi_max);
    addParam(params, selected, "correlation_dedup", "method", state.correlation_method || "spearman");
    addParam(params, selected, "correlation_dedup", "max_abs_corr", state.correlation_max_abs);
    addParam(params, selected, "null_importance_filter", "null_rounds", 20);
    addParam(params, selected, "null_importance_filter", "null_percentile", 75);
    addParam(params, selected, "null_importance_filter", "score_threshold", 1.0);
    addParam(params, selected, "baseline_importance_filter", "importance_type", "gain");
    addParam(params, selected, "baseline_importance_filter", "keep_top_n", 500);
    addParam(params, selected, "lightgbm_binary_training", "early_stopping_rounds", 50);
    addParam(params, selected, "lightgbm_binary_training", "max_auc_gap", 0.02);
    addParam(params, selected, "score_psi", "warn_psi", state.score_psi_warn);
    addParam(params, selected, "scale_pos_weight", "mode", "negative_over_positive");
    return params;
  }

  function featureRounds(stageSteps) {
    const rounds = [];
    if (stageSteps.feature_metadata) rounds.push("metadata");
    if (stageSteps.feature_prescreen) rounds.push("prescreen");
    if (stageSteps.feature_refine) rounds.push("refine");
    return rounds;
  }

  function experimentForState(state) {
    const name = String(state.experiment_name || "").trim() || "main_lgbm";
    return {
      name,
      method: state.experiment_method || "lightgbm",
      segment: "all",
      description: state.experiment_description || "标准 baseline 实验。",
    };
  }

  function completeState(state) {
    const contract = contractForState(state);
    const stageSteps = stageStepsForState(state);
    return {
      ...state,
      request_id: ensureRequestId(state.request_id),
      project: PROJECT_CONTRACT.project,
      task_mode: TASK_MODE_LABELS[state.workflow] || state.workflow,
      business_domain: PROFILE_BUSINESS_DOMAINS[state.scenario_profile] || state.business_domain,
      contract,
      stage_steps: stageSteps,
      step_params: stepParamsForState(state),
      feature_rounds: featureRounds(stageSteps),
      experiment: experimentForState(state),
      reports: {
        sections: state.report_sections || DEFAULTS.report_sections,
        outputs: DEFAULTS.report_outputs,
      },
    };
  }

  function buildMarkdown(rawState) {
    const state = completeState(rawState);
    const contract = state.contract;
    const lines = [
      "---",
      `request_id: ${yamlScalar(state.request_id)}`,
      `title: ${yamlScalar(state.title)}`,
      `project: ${yamlScalar(state.project)}`,
      `workflow: ${yamlScalar(state.workflow)}`,
      `task_mode: ${yamlScalar(state.task_mode)}`,
      `owner: ${yamlScalar(state.owner)}`,
      `business_domain: ${yamlScalar(state.business_domain)}`,
      `scenario_profile: ${yamlScalar(state.scenario_profile)}`,
      "request_source:",
      "  channel: html_preview_builder",
      "  human_interaction: html_only",
      "  downstream_executor: ai_agent_with_rmw",
      "",
      `data_source_mode: ${yamlScalar(state.data_source_mode)}`,
      `sample_location: ${yamlScalar(contract.sample_location)}`,
      `feature_location: ${yamlScalar(contract.feature_location)}`,
      `target_column: ${yamlScalar(contract.target_column)}`,
      "id_columns:",
      yamlList(contract.id_columns, 2),
      `time_column: ${yamlScalar(contract.time_column)}`,
      `period_column: ${yamlScalar(contract.period_column)}`,
      `split_column: ${yamlScalar(contract.split_column)}`,
      "splits:",
      "  dev:",
      "    values:",
      yamlList(contract.dev_values, 6),
      "  oos:",
      "    values:",
      yamlList(contract.oos_values, 6),
      "  oot:",
      "    values:",
      yamlList(contract.oot_values, 6),
      "",
      "sample_checks:",
      yamlList(["sample_check_profile", "sample_check_stability"], 2),
      "",
      "stage_steps:",
      yamlMapOfLists(state.stage_steps, 2),
      "",
      "step_params:",
      yamlMapOfMaps(state.step_params, 2),
      "",
      "feature_selection:",
      "  rounds:",
      yamlList(state.feature_rounds, 4),
      "  require_sql_approval: true",
      "",
      "candidate_targets:",
      yamlList(list(state.candidate_targets || contract.target_column), 2),
      "sample_variants:",
      yamlList(list(state.sample_variants || "all"), 2),
      "experiment_description: " + yamlScalar(state.experiment_description),
      "experiments:",
      `  - name: ${yamlScalar(state.experiment.name)}`,
      `    method: ${yamlScalar(state.experiment.method)}`,
      `    segment: ${yamlScalar(state.experiment.segment)}`,
      `    description: ${yamlScalar(state.experiment.description)}`,
      "",
      "evaluation:",
      "  metrics:",
      yamlList(state.metrics, 4),
      "  champions:",
      yamlList(list(state.champions || PROJECT_CONTRACT.champions), 4),
      "  comparison_dimensions:",
      yamlList(state.comparison_dimensions, 4),
      "  risk_profile_dimensions:",
      yamlList(list(state.risk_profile_dimensions || PROJECT_CONTRACT.risk_profile_dimensions), 4),
      "",
      "reports:",
      "  sections:",
      yamlList(state.reports.sections, 4),
      "  outputs:",
      yamlList(state.reports.outputs, 4),
      "---",
      "",
      "# 建模目标",
      "",
      multiline(state.objective),
      "",
      "# 样本与切分",
      "",
      multiline(contract.sample_definition),
      "",
      "# 本次变化",
      "",
      `口径模式：${state.contract_mode === "override" ? "本次覆盖项目口径" : "沿用项目口径"}。`,
      "",
      `实验：${state.experiment.description}`,
      "",
      "# 评估与报告要求",
      "",
      `重点比较维度：${list(state.comparison_dimensions).join(", ") || "待补充"}。`,
      "",
      `风险画像维度：${list(state.risk_profile_dimensions || PROJECT_CONTRACT.risk_profile_dimensions).join(", ") || "待补充"}。`,
      "",
      "# Agent 执行说明",
      "",
      "该 Markdown 是需求合同。AI Agent 应先执行 `rmw request validate` 和 `rmw plan create`，再使用工作台 CLI 完成后续流程；涉及真实 DP/TMLSQLClient 拉数时必须先生成 SQL 并获得明确审批。",
      "",
      "# 补充说明",
      "",
      multiline(state.extra_notes),
      "",
    ];
    return lines.join("\n");
  }

  function stateFromDefaults() {
    return {
      ...DEFAULTS,
      request_id: ensureRequestId(DEFAULTS.request_id),
    };
  }

  function checkedValues(form, name) {
    return Array.from(form.querySelectorAll(`input[name="${name}"]:checked`)).map((input) => input.value);
  }

  function field(form, name) {
    const el = form.elements[name];
    if (!el) return "";
    if (el instanceof RadioNodeList) return el.value;
    if (el.type === "checkbox") return el.checked;
    return el.value;
  }

  function collectFormState(form) {
    return {
      request_id: requestId,
      title: field(form, "title"),
      owner: field(form, "owner"),
      workflow: field(form, "workflow"),
      business_domain: field(form, "business_domain"),
      scenario_profile: field(form, "scenario_profile"),
      contract_mode: field(form, "contract_mode"),
      data_source_mode: field(form, "data_source_mode"),
      sample_location: field(form, "sample_location"),
      feature_location: field(form, "feature_location"),
      target_column: field(form, "target_column"),
      id_columns: field(form, "id_columns"),
      time_column: field(form, "time_column"),
      period_column: field(form, "period_column"),
      split_column: field(form, "split_column"),
      dev_values: field(form, "dev_values"),
      oos_values: field(form, "oos_values"),
      oot_values: field(form, "oot_values"),
      sample_definition: field(form, "sample_definition"),
      objective: field(form, "objective"),
      experiment_name: field(form, "experiment_name"),
      experiment_method: field(form, "experiment_method"),
      experiment_description: field(form, "experiment_description"),
      candidate_targets: field(form, "candidate_targets"),
      sample_variants: field(form, "sample_variants"),
      champions: field(form, "champions"),
      risk_profile_dimensions: field(form, "risk_profile_dimensions"),
      comparison_dimensions: checkedValues(form, "comparison_dimensions"),
      metrics: checkedValues(form, "metrics"),
      report_sections: DEFAULTS.report_sections,
      report_outputs: DEFAULTS.report_outputs,
      extra_notes: field(form, "extra_notes"),
      missing_rate_threshold: field(form, "missing_rate_threshold"),
      constant_max_unique_values: DEFAULTS.constant_max_unique_values,
      iv_min: field(form, "iv_min"),
      psi_max: DEFAULTS.psi_max,
      correlation_method: DEFAULTS.correlation_method,
      correlation_max_abs: field(form, "correlation_max_abs"),
      score_psi_warn: field(form, "score_psi_warn"),
      sql_block_high_risk: Boolean(field(form, "sql_block_high_risk")),
    };
  }

  function setField(form, name, value) {
    const el = form.elements[name];
    if (!el) return;
    if (el instanceof RadioNodeList) {
      Array.from(el).forEach((input) => {
        input.checked = input.value === value;
      });
      return;
    }
    if (el.type === "checkbox") {
      el.checked = Boolean(value);
      return;
    }
    el.value = value ?? "";
  }

  function setCheckedValues(form, name, values) {
    const selected = new Set(values || []);
    form.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
      input.checked = selected.has(input.value);
    });
  }

  function applyState(form, state) {
    Object.entries(state).forEach(([key, value]) => {
      if (key === "comparison_dimensions" || key === "metrics") return;
      setField(form, key, Array.isArray(value) ? value.join(", ") : value);
    });
    setCheckedValues(form, "comparison_dimensions", state.comparison_dimensions);
    setCheckedValues(form, "metrics", state.metrics);
  }

  function setList(el, items, emptyText = "暂无") {
    if (!el) return;
    el.innerHTML = "";
    const values = items.length ? items : [emptyText];
    values.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      if (!items.length) li.className = "empty";
      el.appendChild(li);
    });
  }

  function renderInheritedContract(state) {
    const el = document.querySelector("#inheritedContract");
    if (!el) return;
    const contract = contractForState(state);
    const items = [
      ["标签字段", contract.target_column],
      ["主键字段", contract.id_columns.join(", ")],
      ["时间字段", contract.time_column],
      ["分区字段", contract.period_column],
      ["切分字段", contract.split_column],
      ["DEV/OOS/OOT", `${contract.dev_values.join(", ")} / ${contract.oos_values.join(", ")} / ${contract.oot_values.join(", ")}`],
    ];
    el.innerHTML = items
      .map(([label, value]) => `<div class="contract-item"><span>${label}</span><strong>${value}</strong></div>`)
      .join("");
  }

  function updateSummaries(state) {
    const complete = completeState(state);
    const stageSteps = complete.stage_steps;
    const stepCount = Object.values(stageSteps).flat().length;
    const profileLabel = PROFILE_LABELS[state.scenario_profile] || state.scenario_profile;
    const domainLabel = BUSINESS_DOMAIN_LABELS[complete.business_domain] || complete.business_domain;
    const liveTitle = document.querySelector("#liveTitle");
    const liveSummary = document.querySelector("#liveSummary");
    const stepCountEl = document.querySelector("#stepCount");
    const filenameHint = document.querySelector("#filenameHint");
    const preview = document.querySelector("#markdownPreview");

    if (liveTitle) liveTitle.textContent = state.title || "未命名需求";
    if (liveSummary) {
      liveSummary.textContent = `${domainLabel} / ${profileLabel} / ${TASK_MODE_LABELS[state.workflow] || state.workflow}。口径模式：${
        state.contract_mode === "override" ? "本次覆盖" : "沿用项目"
      }。`;
    }
    if (stepCountEl) stepCountEl.textContent = String(stepCount);
    if (filenameHint) filenameHint.textContent = `${complete.request_id}.md`;
    if (preview) preview.textContent = buildMarkdown(state);

    setList(
      document.querySelector("#stageSummary"),
      Object.entries(stageSteps).map(([stage, steps]) => `${STAGE_LABELS[stage] || stage}: ${steps.length} 项`),
      "当前任务没有执行步骤。",
    );
    setList(document.querySelector("#userSummary"), [
      `任务模式：${TASK_MODE_LABELS[state.workflow] || state.workflow}`,
      `建模场景：${profileLabel}`,
      `数据源：${state.data_source_mode === "local_feather" ? "本地 feather" : "DP 表或 SQL"}`,
      `实验：${complete.experiment.name}`,
    ]);
    setList(document.querySelector("#inheritSummary"), [
      `项目：${PROJECT_CONTRACT.project}`,
      `标签：${complete.contract.target_column}`,
      `切分：${complete.contract.split_column}`,
      `Champion：${list(state.champions || PROJECT_CONTRACT.champions).join(", ")}`,
    ]);
    setList(document.querySelector("#generatedSummary"), [
      `stage_steps：${stepCount} 项`,
      `feature rounds：${complete.feature_rounds.join(", ") || "[]"}`,
      `step_params：${Object.keys(complete.step_params).length} 组`,
      `报告输出：${complete.reports.outputs.join(", ")}`,
    ]);

    const guardrails = [];
    if (!String(state.objective || "").trim()) guardrails.push("建模目标为空。");
    if (state.data_source_mode === "local_feather" && !String(complete.contract.sample_location || "").endsWith(".feather")) {
      guardrails.push("本地 feather 模式下，样本位置应以 .feather 结尾。");
    }
    if (!list(state.metrics).length) guardrails.push("未选择评估指标。");
    if (!list(state.comparison_dimensions).length) guardrails.push("未选择重点比较维度。");
    if (!guardrails.length) guardrails.push("可以下载 Markdown。");
    setList(document.querySelector("#guardrailSummary"), guardrails);
  }

  function updateContractMode(state) {
    document.body.classList.toggle("override-mode", state.contract_mode === "override");
  }

  function persistState(state) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...state, request_id: ensureRequestId(state.request_id) }));
  }

  function showToast(message) {
    const toast = document.querySelector("#toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
  }

  function updateAll(form, { persist = true } = {}) {
    const state = collectFormState(form);
    updateContractMode(state);
    renderInheritedContract(state);
    updateSummaries(state);
    if (persist) persistState(state);
  }

  function activateStep(step) {
    activeStep = step;
    const index = ["task", "contract", "changes", "review"].indexOf(step);
    document.querySelectorAll(".flow-step").forEach((section) => {
      const active = section.dataset.step === step;
      section.hidden = !active;
      section.classList.toggle("is-active", active);
    });
    document.querySelectorAll("[data-step-target]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.stepTarget === step);
    });
    const prev = document.querySelector("#prevStep");
    const next = document.querySelector("#nextStep");
    const progress = document.querySelector("#stepProgress");
    if (prev) prev.disabled = index <= 0;
    if (next) next.textContent = index >= 3 ? "回到开头" : "下一步";
    if (progress) progress.textContent = `${index + 1} / 4`;
  }

  function nextStep() {
    const steps = ["task", "contract", "changes", "review"];
    const index = steps.indexOf(activeStep);
    activateStep(steps[index >= steps.length - 1 ? 0 : index + 1]);
  }

  function prevStep() {
    const steps = ["task", "contract", "changes", "review"];
    const index = steps.indexOf(activeStep);
    activateStep(steps[Math.max(0, index - 1)]);
  }

  async function copyMarkdown(form) {
    updateAll(form, { persist: false });
    const markdown = buildMarkdown(collectFormState(form));
    try {
      await navigator.clipboard.writeText(markdown);
      showToast("Markdown 已复制");
    } catch (error) {
      const helper = document.createElement("textarea");
      helper.value = markdown;
      helper.setAttribute("readonly", "");
      helper.style.position = "fixed";
      helper.style.opacity = "0";
      document.body.appendChild(helper);
      helper.select();
      document.execCommand("copy");
      helper.remove();
      showToast("Markdown 已复制");
    }
  }

  function downloadMarkdown(form) {
    updateAll(form, { persist: false });
    const state = completeState(collectFormState(form));
    const blob = new Blob([buildMarkdown(state)], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.request_id}.md`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    showToast("已生成下载文件");
  }

  function resetDraft(form) {
    localStorage.removeItem(STORAGE_KEY);
    requestId = generateRequestId();
    applyState(form, stateFromDefaults());
    updateAll(form, { persist: false });
    activateStep("task");
    showToast("已重置预览版表单");
  }

  function restoreDraft(form) {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) {
      showToast("没有可恢复的草稿");
      return;
    }
    try {
      const state = JSON.parse(saved);
      requestId = state.request_id || generateRequestId();
      applyState(form, { ...stateFromDefaults(), ...state });
      updateAll(form, { persist: false });
      showToast("已恢复草稿");
    } catch (error) {
      localStorage.removeItem(STORAGE_KEY);
      showToast("草稿已损坏，已清除");
    }
  }

  function applyProfileDefaults(form) {
    const profile = field(form, "scenario_profile");
    setField(form, "business_domain", PROFILE_BUSINESS_DOMAINS[profile] || "preloan");
    if (profile === "fujie_gcard_main_lgbm") {
      const current = collectFormState(form);
      applyState(form, {
        ...DEFAULTS,
        ...current,
        title: current.title || DEFAULTS.title,
        objective: current.objective || DEFAULTS.objective,
        sample_location: current.sample_location || PROJECT_CONTRACT.source_table,
        champions: current.champions || PROJECT_CONTRACT.champions.join(", "),
        risk_profile_dimensions: current.risk_profile_dimensions || PROJECT_CONTRACT.risk_profile_dimensions.join(", "),
      });
    }
  }

  function boot() {
    const form = document.querySelector("#previewForm");
    if (!form) return;
    requestId = generateRequestId();
    applyState(form, stateFromDefaults());
    activateStep("task");
    updateAll(form, { persist: false });

    form.addEventListener("input", () => updateAll(form));
    form.addEventListener("change", (event) => {
      if (event.target.name === "scenario_profile") applyProfileDefaults(form);
      updateAll(form);
    });

    document.querySelectorAll("[data-step-target]").forEach((button) => {
      button.addEventListener("click", () => activateStep(button.dataset.stepTarget));
    });
    document.querySelector("#nextStep")?.addEventListener("click", nextStep);
    document.querySelector("#prevStep")?.addEventListener("click", prevStep);
    document.querySelector("#copyMarkdown")?.addEventListener("click", () => copyMarkdown(form));
    document.querySelector("#downloadMarkdown")?.addEventListener("click", () => downloadMarkdown(form));
    document.querySelector("#resetDraft")?.addEventListener("click", () => resetDraft(form));
    document.querySelector("#restoreDraft")?.addEventListener("click", () => restoreDraft(form));
  }

  return {
    boot,
    buildMarkdown,
    completeState,
    stateFromDefaults,
    stageStepsForState,
    stepParamsForState,
  };
});
