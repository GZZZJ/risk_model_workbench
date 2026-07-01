document.addEventListener('DOMContentLoaded', function () {
  var body = document.querySelector('.report-body');
  if (!body) return;

  var idByTitle = [
    ['总结', 'summary'],
    ['Summary', 'summary'],
    ['模型描述', 'model-description'],
    ['变量筛选', 'feature-selection'],
    ['核心效果', 'core-comparison'],
    ['模型效果', 'model-performance'],
    ['模型稳定性', 'model-stability'],
    ['重要变量', 'important-features'],
    ['Top变量WOE', 'top-woe'],
    ['待补充事项', 'missing-results']
  ];

  var headings = Array.prototype.slice.call(body.querySelectorAll(':scope > h2'));
  headings.forEach(function (heading, index) {
    var section = document.createElement('section');
    section.className = 'report-section';
    var title = heading.textContent.trim();
    var match = idByTitle.find(function (pair) { return title.indexOf(pair[0]) !== -1; });
    section.id = match ? match[1] : 'section-' + (index + 1);

    var badge = document.createElement('span');
    badge.className = 'section-number';
    badge.textContent = String(index + 1);
    heading.prepend(badge);

    body.insertBefore(section, heading);
    var current = heading;
    while (current) {
      var next = current.nextSibling;
      section.appendChild(current);
      if (next && next.nodeType === 1 && next.matches('h2')) break;
      current = next;
    }
  });

  function parseNumeric(text) {
    var cleaned = String(text || '').replace(/[,%+]/g, '').trim();
    if (!cleaned || /^N\/A$/i.test(cleaned)) return null;
    var value = parseFloat(cleaned);
    return Number.isFinite(value) ? value : null;
  }

  function tableContext(table) {
    var parts = [];
    var sectionTitle = table.closest('.report-section') && table.closest('.report-section').querySelector('h2');
    if (sectionTitle) parts.push(sectionTitle.textContent.trim());
    var node = table.previousElementSibling;
    var hops = 0;
    while (node && hops < 4) {
      if (/^(H2|H3|P|BLOCKQUOTE|UL)$/i.test(node.tagName)) parts.push(node.textContent.trim());
      node = node.previousElementSibling;
      hops += 1;
    }
    return parts.join(' | ');
  }

  function columnValues(table, columnIndex) {
    return Array.prototype.slice.call(table.querySelectorAll('tr')).slice(1).map(function (row) {
      return parseNumeric(row.children[columnIndex] && row.children[columnIndex].textContent);
    }).filter(function (value) {
      return value !== null;
    });
  }

  function shouldSkipNumericHeader(header) {
    return /旧→新|样本数|n_samples|positive|index|序号|排名|split|分组|月份|month|版本|客群|样本$|feature|varname|desc|变量|字段/i.test(header);
  }

  function heatColor(header, context, value, min, max) {
    var absMax = Math.max(Math.abs(min), Math.abs(max));
    var norm = max === min ? 0.75 : (value - min) / (max - min);
    var alpha = 0.08 + Math.max(0, Math.min(1, norm)) * 0.30;
    if (/提升|uplift|delta|Δ|gap/i.test(header)) {
      var deltaStrength = absMax ? Math.min(1, Math.abs(value) / absMax) : 0;
      var deltaAlpha = 0.08 + deltaStrength * 0.30;
      if (value > 0) return 'rgba(24,169,87,' + deltaAlpha.toFixed(3) + ')';
      if (value < 0) return 'rgba(207,68,68,' + deltaAlpha.toFixed(3) + ')';
      return 'rgba(101,117,138,.10)';
    }
    if (/PSI/i.test(header + ' ' + context)) return 'rgba(207,68,68,' + alpha.toFixed(3) + ')';
    if (/风险|overdue|bad_rate/i.test(header + ' ' + context)) return 'rgba(230,126,34,' + alpha.toFixed(3) + ')';
    return 'rgba(45,156,219,' + alpha.toFixed(3) + ')';
  }

  function addHeatScale(table, headers, context, columns) {
    columns.forEach(function (columnIndex) {
      var values = columnValues(table, columnIndex);
      if (values.length < 2) return;
      var min = Math.min.apply(null, values);
      var max = Math.max.apply(null, values);
      Array.prototype.slice.call(table.querySelectorAll('tr')).slice(1).forEach(function (row) {
        var cell = row.children[columnIndex];
        var value = parseNumeric(cell && cell.textContent);
        if (!cell || value === null) return;
        cell.classList.add('heat-cell');
        cell.style.setProperty('--heat-bg', heatColor(headers[columnIndex], context, value, min, max));
      });
    });
  }

  function addDataBars(table, headers, columns) {
    columns.forEach(function (columnIndex) {
      var values = columnValues(table, columnIndex).map(Math.abs);
      if (values.length < 2) return;
      var max = Math.max.apply(null, values);
      if (!max) return;
      Array.prototype.slice.call(table.querySelectorAll('tr')).slice(1).forEach(function (row) {
        var cell = row.children[columnIndex];
        var value = parseNumeric(cell && cell.textContent);
        if (!cell || value === null) return;
        var width = Math.max(3, Math.min(100, Math.abs(value) / max * 100));
        cell.classList.add('bar-cell');
        if (/lift/i.test(headers[columnIndex])) cell.classList.add('bar-good');
        if (/剩余|提升/i.test(headers[columnIndex])) cell.classList.add('bar-warn');
        cell.style.setProperty('--bar-width', width.toFixed(1) + '%');
      });
    });
  }

  function enhanceNumericTable(table, headers, context, changeColumns) {
    var slopingTable = /sloping|lift|高分10%|累计发起率|剩余发起率/i.test(context + ' ' + headers.join(' '));
    var comparisonTable = /by月|按月|每月|OOS|整体效果|分客群整体效果|效果对比|AUC|KS/i.test(context);
    var psiTable = /PSI|稳定性/i.test(context + ' ' + headers.join(' '));
    var barColumns = [];
    var heatColumns = [];

    headers.forEach(function (header, columnIndex) {
      var values = columnValues(table, columnIndex);
      if (values.length < 2 || shouldSkipNumericHeader(header)) return;
      if (slopingTable && /发起率|lift|提升/i.test(header)) {
        barColumns.push(columnIndex);
        return;
      }
      if (changeColumns.indexOf(columnIndex) !== -1) {
        heatColumns.push(columnIndex);
        return;
      }
      if (psiTable && /psi/i.test(header)) {
        heatColumns.push(columnIndex);
        return;
      }
      if (comparisonTable && (/AUC|KS|本轮|model_score|bad_rate|发起率|风险率/i.test(header) || /AUC|KS/.test(context))) {
        heatColumns.push(columnIndex);
      }
    });

    if (barColumns.length || heatColumns.length) table.closest('.table-wrap').classList.add('visual-table');
    addDataBars(table, headers, barColumns);
    addHeatScale(table, headers, context, heatColumns);
  }

  body.querySelectorAll('table').forEach(function (table) {
    if (!table.parentElement || table.parentElement.classList.contains('table-wrap')) return;
    var context = tableContext(table);
    var wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    if (table.querySelectorAll('th').length >= 7) wrap.classList.add('wide-table');
    if (table.querySelectorAll('tr').length >= 12) wrap.classList.add('dense-table');
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);

    var headers = Array.prototype.slice.call(table.querySelectorAll('tr:first-child th')).map(function (th) {
      return th.textContent.trim();
    });
    var changeColumns = headers.reduce(function (acc, header, columnIndex) {
      if (/提升|gap|delta|uplift|Δ/i.test(header)) acc.push(columnIndex);
      return acc;
    }, []);

    table.querySelectorAll('tr').forEach(function (row, rowIndex) {
      if (rowIndex === 0) return;
      Array.prototype.slice.call(row.children).forEach(function (cell, columnIndex) {
        if (changeColumns.indexOf(columnIndex) === -1) return;
        var value = parseFloat(cell.textContent.replace(/[,%+]/g, ''));
        if (Number.isNaN(value)) return;
        if (value > 0) cell.classList.add('is-positive');
        if (value < 0) cell.classList.add('is-negative');
        if (value === 0) cell.classList.add('is-neutral');
      });
    });

    enhanceNumericTable(table, headers, context, changeColumns);
  });

  var navItems = Array.prototype.slice.call(document.querySelectorAll('.nav-item'));
  var sections = Array.prototype.slice.call(document.querySelectorAll('.report-section'));
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        navItems.forEach(function (item) {
          item.classList.toggle('is-active', item.getAttribute('href') === '#' + entry.target.id);
        });
      });
    }, {rootMargin: '-35% 0px -55% 0px', threshold: 0});
    sections.forEach(function (section) { observer.observe(section); });
  }
});
