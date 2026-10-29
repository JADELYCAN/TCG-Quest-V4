import unittest, json, pathlib, sys, types, base64, io
try: import requests
except ImportError: sys.modules['requests']=types.SimpleNamespace(RequestException=Exception)
from unittest.mock import patch
from PIL import Image
import evidence
class EvidenceGeometry(unittest.TestCase):
 def test_real_off_card_regression(self):
  self.assertFalse(evidence.on_card([850,20,940,110],[[.153939,.13],[.810909,.124545],[.814545,.822727],[.150303,.821818]]))
 def test_tilt_and_invalid(self):
  q=[[.5,.1],[.9,.5],[.5,.9],[.1,.5]]
  self.assertFalse(evidence.on_card([100,100,160,160],q))
  self.assertTrue(evidence.on_card([450,450,550,550],q))
  self.assertFalse(evidence.on_card([450,450,550,550],None))
 def test_no_provider_call_for_off_card(self):
  c=dict(side='front',type='corner_wear',bbox=[850,20,940,110],observation='Possible corner wear',confidence='high')
  with patch.object(evidence.requests,'post',create=True) as post:
   self.assertEqual(evidence.verify_evidence([c],{},'fake',card_quads={'front':[[.1,.1],[.8,.1],[.8,.9],[.1,.9]]}),[]);post.assert_not_called()
 def test_exact_region_highlighted(self):
  im=Image.new('RGB',(1000,1000),'white');buf=io.BytesIO();im.save(buf,format='JPEG')
  url=evidence._crop_data_url(base64.b64encode(buf.getvalue()).decode(),[400,400,450,450])
  out=Image.open(io.BytesIO(base64.b64decode(url.split(',')[1])))
  self.assertTrue(any(r>180 and b>150 and g<130 for r,g,b in out.getdata()))
if __name__=='__main__':unittest.main()
