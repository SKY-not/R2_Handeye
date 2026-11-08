"""Portable reports keep displayed values and downloadable numeric artifacts together."""
import base64
import csv
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
import numpy as np
from handeye.analysis.html_report import write_html_report
from handeye.analysis.report import statistics
from handeye.session import write_json


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images, self.downloads = [], {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img' and attrs.get('src'):
            self.images.append(attrs['src'])
        if tag == 'a' and attrs.get('download'):
            self.downloads[attrs['download']] = attrs['href']


class HtmlReportTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(frame='f'+str(i),translation_mm=x,rotation_deg=x/10,
                     observed_reprojection_px=x/5,chain_reprojection_px=x+1,
                     reference_position_mm=x+4,reference_rotation_deg=(x+4)/10)
                for i,x in enumerate([1.,3.])]
        report = {'sample_count':2,'scope':'training_sample_fit',
            'spatial_position_mm':statistics([1.,3.]),'spatial_rotation_deg':statistics([.1,.3]),
            'observed_reprojection_px':statistics([.2,.6]),'chain_reprojection_px':statistics([2.,4.]),
            'reference':{'position_mm':statistics([5.,7.]),'rotation_deg':statistics([.5,.7]),
                'source':'reference.json','definition':{'board_to_tcp_rough':[0,0,.07,0,-90,0],
                    'translation_unit':'m','rotation_unit':'deg'}}}
        write_json(root/'evaluation.json',report)
        with (root/'residuals.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (root/'figures').mkdir()
        # A tiny valid PNG is sufficient for verifying portable asset encoding.
        png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a3ioAAAAASUVORK5CYII=')
        for name in ['pose_errors','components','reprojection','reference_pose_errors','reference_components']:
            (root/'figures'/f'{name}.png').write_bytes(png)
        context={'mode':'eye_to_hand','source_session':'data/test','transform':np.eye(4).tolist()}
        return report,rows,context,png

    def test_portable_downloads_and_charts_match_numeric_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report,rows,context,png=self.fixture(root)
            write_html_report(root,report,rows,context=context)
            text=(root/'report.html').read_text()
            parser=ReportParser();parser.feed(text)
            self.assertEqual(len(parser.images),5)
            for uri in parser.images:
                self.assertTrue(uri.startswith('data:image/png;base64,'))
                self.assertEqual(base64.b64decode(uri.split(',',1)[1]),png)
            for name in ['evaluation.json','residuals.csv']:
                data=base64.b64decode(parser.downloads[name].split(',',1)[1])
                self.assertEqual(data,(root/name).read_bytes())
            matrix=base64.b64decode(parser.downloads['handeye_transform.txt'].split(',',1)[1]).decode()
            for value in matrix.split():self.assertRegex(value,r'^-?[0-9]+[.][0-9]{8}$')
            self.assertIn('T_base_camera',text)
            self.assertIn('1.00000000',text)

    def test_inconsistent_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report,rows,context,_=self.fixture(root)
            report['spatial_position_mm']['mean']+=1
            with self.assertRaisesRegex(ValueError,'statistic mismatch'):
                write_html_report(root,report,rows,context=context)
            self.assertFalse((root/'report.html').exists())

    def test_source_text_is_escaped_and_missing_reference_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report,rows,context,_=self.fixture(root)
            report['reference']=None
            context['source_session']='<script>alert(1)</script>'
            for row in rows:row['reference_position_mm']=row['reference_rotation_deg']=None
            write_html_report(root,report,rows,context=context)
            text=(root/'report.html').read_text()
            self.assertNotIn('<script>alert(1)</script>',text)
            self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;',text)
            self.assertIn('\u672a\u63d0\u4f9b\u53c2\u8003\u4f4d\u59ff',text)


if __name__=='__main__':unittest.main()
