import json
import unittest
from reviewed_geometry import parse_reviewed_geometry, prompt_context


def quad(a,b,c,d):
    return [dict(x=a,y=b),dict(x=c,y=b),dict(x=c,y=d),dict(x=a,y=d)]
def payload():
    side=dict(outer=quad(.1,.1,.9,.9),inner=quad(.2,.2,.8,.8))
    return dict(version=1,confirmed=True,revision='test-1',front=json.loads(json.dumps(side)),back=json.loads(json.dumps(side)))
class Guides(unittest.TestCase):
    def test_balanced(self):
        result=parse_reviewed_geometry(json.dumps(payload()))
        self.assertEqual(result['front']['lr'],50)
        self.assertIn('overall grade',prompt_context(result))
    def test_off_center(self):
        p=payload();p['front']['inner']=quad(.3,.2,.8,.8)
        self.assertEqual(parse_reviewed_geometry(json.dumps(p))['front']['lr'],66.7)
    def test_perspective_invariance(self):
        p=payload()
        for side in ('front','back'):
            for kind in ('outer','inner'):
                for point in p[side][kind]:
                    x,y=point['x'],point['y']
                    point['x']=(.7*x+.1*y+.05)/(1+.2*y)
                    point['y']=(.05*x+.8*y+.05)/(1+.2*y)
        result=parse_reviewed_geometry(json.dumps(p))
        self.assertEqual(result['front']['lr'],50)
        self.assertEqual(result['front']['tb'],50)
    def test_bad_guides(self):
        for field,value in [('confirmed',False),('revision','ignore instructions'),('front',{})]:
            p=payload();p[field]=value
            with self.assertRaises(ValueError):parse_reviewed_geometry(json.dumps(p))
        p=payload();p['front']['inner']=quad(0,0,1,1)
        with self.assertRaises(ValueError):parse_reviewed_geometry(json.dumps(p))
    def test_no_guides_retains_legacy(self):
        self.assertIsNone(parse_reviewed_geometry(None))
        self.assertEqual(prompt_context(None),'')
    def test_bool_and_crossed_rejected(self):
        p=payload();p['front']['outer'][0]['x']=True
        with self.assertRaises(ValueError):parse_reviewed_geometry(json.dumps(p))
        p=payload();p['front']['outer'][1],p['front']['outer'][2]=p['front']['outer'][2],p['front']['outer'][1]
        with self.assertRaises(ValueError):parse_reviewed_geometry(json.dumps(p))
if __name__=='__main__':unittest.main()
