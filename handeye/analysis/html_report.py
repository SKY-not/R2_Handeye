"""Portable HTML summaries, using the same evaluated numbers and plot files."""
import base64
from html import escape
from pathlib import Path

import numpy as np

_STYLE = """
:root{color-scheme:light;font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#243247;background:#f4f6f8}
*{box-sizing:border-box}body{margin:0;font-size:14px;line-height:1.65}main{max-width:1180px;margin:auto;padding:30px 24px 48px}
h1,h2,h3,p,figure,dl{margin:0}h1{font-size:26px;font-weight:650}h2{font-size:19px;margin-bottom:15px}h3{font-size:15px;margin-bottom:8px}
header,section{background:#fff;border:1px solid #dce2e8;border-radius:8px;padding:24px;margin-bottom:20px;min-width:0}
.muted,figcaption{color:#647084;font-size:13px}.intro{margin:7px 0 16px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.metric{padding:15px;background:#f7f9fb;border:1px solid #e3e8ee;border-radius:6px}.metric dt{font-size:13px;color:#59677a}
.metric dd{margin:5px 0 0;font-size:27px;font-weight:650;font-variant-numeric:tabular-nums}.metric small{font-size:13px;font-weight:400}
a{color:#245a94;text-underline-offset:3px}.links{display:flex;gap:18px;flex-wrap:wrap;margin-top:13px}.badge{display:inline-block;background:#edf3f8;color:#45627d;border-radius:4px;padding:2px 8px;font-size:12px}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;border-bottom:1px solid #e4e8ed;text-align:right;vertical-align:top;white-space:nowrap}
th{background:#f7f9fb;color:#4a586b;font-weight:600}th:first-child,td:first-child{text-align:left}tbody tr:last-child td{border-bottom:0}
.parameters{display:grid;grid-template-columns:1.2fr 1fr;gap:24px}.parameters>div{min-width:0}.matrix{width:auto;font-family:ui-monospace,Consolas,monospace;margin:10px 0}
.matrix td{border:0;padding:5px 9px;text-align:right}.source{overflow-wrap:anywhere}.reference-value{font-family:ui-monospace,Consolas,monospace;padding:10px 12px;background:#f7f9fb;border-radius:5px;margin:10px 0}
.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.charts figure{padding:14px;border:1px solid #e3e8ee;border-radius:6px;min-width:0}
.charts figure:last-child:nth-child(odd){grid-column:1/-1}.plot-button{display:block;width:100%;padding:0;border:0;background:white;cursor:zoom-in}.plot-button img{display:block;width:100%;height:auto}
figcaption{margin-top:8px}details{margin-top:16px}summary{cursor:pointer;font-weight:550;color:#30465f;padding:7px 0}.scroll-table{max-height:430px;overflow:auto}.scroll-table th{position:sticky;top:0}
.caption{margin-top:10px}.legend{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:12px}.legend span{font-size:13px;color:#647084}
dialog{max-width:96vw;max-height:96vh;border:1px solid #dce2e8;border-radius:8px;padding:16px}dialog::backdrop{background:#192b40aa}
dialog img{display:block;max-width:90vw;max-height:85vh;object-fit:contain}.close{float:right;padding:5px 13px;background:#f4f6f8;border:1px solid #dce2e8;border-radius:4px;cursor:pointer}
footer{font-size:12px;color:#6b7583;padding:0 6px}
@media(max-width:800px){main{padding:16px 12px}header,section{padding:18px}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.parameters,.charts{grid-template-columns:1fr}.charts figure:last-child:nth-child(odd){grid-column:auto}}
@media print{body{background:white}main{max-width:none;padding:0}.links,.close{display:none}section,figure{break-inside:avoid}}
"""
_SCRIPT = """
const modal=document.getElementById('plot-modal');
document.querySelectorAll('.plot-button').forEach(button=>button.addEventListener('click',()=>{
  const image=button.querySelector('img');
  const target=modal.querySelector('img');target.src=image.src;target.alt=image.alt;modal.showModal();
}));
modal.querySelector('button').addEventListener('click',()=>modal.close());
modal.addEventListener('click',event=>{if(event.target===modal)modal.close();});
"""


def _number(value, digits=3):
    return '—' if value is None or not np.isfinite(float(value)) else f'{float(value):.{digits}f}'


def _data_uri(data, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def _metric_specs(report):
    calibration = 'reference' in report
    specs = [
        ('位置一致性','mm',report.get('spatial_position_mm' if calibration else 'position_mm'),'translation_mm'),
        ('旋转一致性','deg',report.get('spatial_rotation_deg' if calibration else 'rotation_deg'),'rotation_deg'),
        ('观测重投影','px',report.get('observed_reprojection_px'),'observed_reprojection_px'),
    ]
    if calibration:
        reference = report.get('reference') or {}
        specs.extend([
            ('Chain 重投影','px',report.get('chain_reprojection_px'),'chain_reprojection_px'),
            ('参考位置偏差','mm',reference.get('position_mm'),'reference_position_mm'),
            ('参考旋转偏差','deg',reference.get('rotation_deg'),'reference_rotation_deg'),
        ])
    return specs


def _verify_summary(report, rows, specs):
    if report['sample_count'] != len(rows):
        raise ValueError('HTML report sample count differs from per-frame rows')
    for _, _, stats, key in specs:
        if stats is None:
            continue
        values = np.array([np.nan if row.get(key) is None else row[key] for row in rows],dtype=float)
        valid = values[np.isfinite(values)]
        if stats['count'] != len(valid) or stats['invalid_count'] != len(values)-len(valid):
            raise ValueError(f'HTML report count mismatch: {key}')
        for name, operation in [('mean',np.mean),('max',np.max),('min',np.min),('std',np.std),('median',np.median)]:
            expected = None if not len(valid) else float(operation(valid))
            if expected is None:
                matches = stats[name] is None
            else:
                matches = stats[name] is not None and np.isclose(stats[name],expected,rtol=1e-10,atol=1e-10)
            if not matches:
                raise ValueError(f'HTML report statistic mismatch: {key}.{name}')


def write_html_report(root, report, rows, *, context=None):
    root = Path(root)
    context = {} if context is None else context
    calibration = 'reference' in report
    specs = _metric_specs(report)
    _verify_summary(report,rows,specs)
    mode = context.get('mode')
    mode_label = {'eye_to_hand':'眼在手外 · Eye-to-Hand','eye_on_hand':'眼在手上 · Eye-on-Hand'}.get(mode,'静态视觉观测')
    scope = {'training_sample_fit':'原标定数据拟合','separate_dataset_consistency':'另一批数据一致性',
             'static_visual_repeatability':'静态视觉重复性'}.get(report.get('scope'),report.get('scope',''))
    title = '手眼标定报告' if calibration else '静态观测报告'
    card_indices = [0,1,3,4] if calibration else [0,1,2]
    cards = ''.join(f'<dl class="metric"><dt>{escape(specs[i][0])} · 均值</dt><dd>{_number((specs[i][2] or {}).get("mean"))} <small>{specs[i][1]}</small></dd></dl>' for i in card_indices)
    metric_rows = ''.join('<tr><td>'+escape(label)+f' <span class="muted">({unit})</span></td>' +
        ''.join('<td>'+_number((stats or {}).get(key))+'</td>' for key in ('mean','max','std'))+
        '<td>'+('—' if stats is None else f'{stats["count"]}/{len(rows)}')+'</td></tr>' for label,unit,stats,_ in specs)
    columns = [('frame','帧'),('translation_mm','一致性 mm'),('rotation_deg','一致性 deg'),
               ('observed_reprojection_px','观测 px')]
    if calibration:
        columns.extend([('chain_reprojection_px','Chain px'),('reference_position_mm','参考 mm'),('reference_rotation_deg','参考 deg')])
    frame_head = ''.join('<th>'+label+'</th>' for _,label in columns)
    frame_rows = ''.join('<tr>'+''.join('<td>'+(escape(str(row.get(key,''))) if key=='frame' else _number(row.get(key)))+'</td>' for key,_ in columns)+'</tr>' for row in rows)
    downloads = ''.join(f'<a download="{name}" href="{_data_uri((root/name).read_bytes(),mime)}">{label}</a>' for name,mime,label in
                        [('evaluation.json','application/json','指标 JSON'),('residuals.csv','text/csv','逐帧 CSV')])
    matrix_section = ''
    if context.get('transform') is not None:
        matrix = np.asarray(context['transform'],dtype=float)
        if matrix.shape != (4,4) or not np.isfinite(matrix).all():
            raise ValueError('HTML report requires a finite 4x4 hand-eye matrix')
        matrix_text = '\n'.join(' '.join(f'{x:.8f}' for x in row) for row in matrix)+'\n'
        matrix_rows = ''.join('<tr>'+''.join(f'<td>{value:.8f}</td>' for value in row)+'</tr>' for row in matrix)
        matrix_name = 'T_base_camera' if mode=='eye_to_hand' else 'T_tcp_camera'
        reference = report.get('reference') or {}
        definition = reference.get('definition') or {}
        reference_key = 'board_to_tcp_rough' if mode=='eye_to_hand' else 'board_to_base_rough'
        reference_name = 'T_tcp_board_ref' if mode=='eye_to_hand' else 'T_base_board_ref'
        pose = definition.get(reference_key)
        reference_value = '未提供参考位姿' if pose is None else '['+', '.join(f'{float(x):g}' for x in pose)+']'
        units = '' if pose is None else '平移 '+str(definition.get('translation_unit',''))+' · 旋转向量 '+str(definition.get('rotation_unit',''))
        matrix_uri = _data_uri(matrix_text.encode('utf-8'),'text/plain')
        matrix_section = f"""
<section class="parameters"><div><h2>手眼矩阵 · {matrix_name}</h2><div class="table-wrap"><table class="matrix"><tbody>{matrix_rows}</tbody></table></div>
<p class="muted">将{('相机坐标转换到机器人基座坐标' if mode=='eye_to_hand' else '相机坐标转换到 TCP 坐标')}，平移单位 m。</p>
<a href="{matrix_uri}" download="handeye_transform.txt">下载矩阵 · 8 位小数</a></div>
<div><h2>参考位姿 · {reference_name}</h2><p class="reference-value">{escape(reference_value)}</p><p class="muted">{escape(units)}</p>
<p class="source muted">{escape(str(reference.get('source','')))}</p></div></section>
"""
    plot_specs = [
        ('pose_errors.png','均值一致性残差','位置和旋转相对样本均值的残差。'),
        ('components.png','一致性分量','XYZ 位移与旋转向量分量。'),
        ('reprojection.png','重投影误差','蓝线：视觉位姿拟合；橙线：机器人与手眼链路预测，使用平均安装关系。' if calibration else '视觉观测的角点重投影误差。'),
        ('reference_pose_errors.png','参考位姿偏差','估计板位姿与配置参考位姿的差。'),
        ('reference_components.png','参考偏差分量','分量表达在参考板坐标系，单位 mm、deg。'),
    ]
    plot_html = []
    for filename,label,caption in plot_specs:
        path = root/'figures'/filename
        if not path.is_file():
            continue
        uri = _data_uri(path.read_bytes(),'image/png')
        plot_html.append(f'<figure><h3>{label}</h3><button type="button" class="plot-button" aria-label="放大{label}"><img alt="{label}" src="{uri}"></button><figcaption>{caption}</figcaption></figure>')
    source = escape(str(context.get('source_session','')))
    footer_note = ('参考指标为相对配置参考值的偏差；' if calibration else '') + '本页与 JSON、CSV 使用同一组评价结果。'
    note = '一致性反映内部自洽；参考偏差包含参考测量误差。' if calibration else '静态重复性不代表手眼标定精度。'
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title} · {escape(root.name)}</title><style>{_STYLE}</style></head>
<body><main><header><h1>{title}</h1><p class="intro muted">{escape(mode_label)} · {report['sample_count']} 帧 · {escape(str(scope))} · {escape(root.name)}</p>
<div class="cards">{cards}</div><p class="caption muted">{note}</p></header>
{matrix_section}
<section><h2>误差汇总</h2><div class="table-wrap"><table id="metrics"><thead><tr><th>指标</th><th>均值</th><th>最大帧值</th><th>标准差</th><th>有效帧</th></tr></thead><tbody>{metric_rows}</tbody></table></div>
<p class="caption"><span class="badge" id="verification">汇总与逐帧数据一致</span> <span class="muted">按帧统计；重投影每帧先对角点取平均，标准差为总体标准差。</span></p>
<div class="links">{downloads}</div><details id="frame-details"><summary>逐帧明细 · {len(rows)} 帧</summary><div class="scroll-table"><table id="frames"><thead><tr>{frame_head}</tr></thead><tbody>{frame_rows}</tbody></table></div></details></section>
<section><h2>结果图 <span class="muted">点击放大</span></h2><div class="charts">{''.join(plot_html)}</div></section>
<footer class="source">数据来源：{source}<br>{footer_note}</footer></main>
<dialog id="plot-modal"><button class="close" type="button">关闭</button><img alt="放大结果图"></dialog><script>{_SCRIPT}</script></body></html>
"""
    (root/'report.html').write_text(document,encoding='utf-8')
