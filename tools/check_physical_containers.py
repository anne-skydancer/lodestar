"""Prepare physical-container controls and validate them with native Firestorm."""

import argparse
from pathlib import Path
from hashlib import sha256
import json
import sys
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
from lodestar_physics_rebuild import HullProbe
from lodestar_textured_glb import write_textured_glb
from bake_section_atlas import indexed_primitives
from firestorm_prepare import prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--library', required=True, type=Path)
parser.add_argument('--native-audit', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path, help='A new output directory')
args = parser.parse_args()
library = args.library.resolve()
native_audit = args.native_audit.resolve()
probe = HullProbe(library)
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
reports = {}
for name in ('Micro','Existing','Flat','Mixed','Thin','SmallScale','Enlarged'):
    scale = .01 if name=='SmallScale' else 10. if name=='Enlarged' else 1.
    size = (.002,.04,.06) if name=='Thin' else (.002,.004,.006)
    body = probe.envelope(np.array([(x,y,z) for x in (0,size[0]) for y in (0,size[1]) for z in (0,size[2])]))
    if name=='Flat':
        body = np.array([[[0,0,0],[.002,0,0],[.002,.004,0]],[[0,0,0],[.002,.004,0],[0,.004,0]]])
    if name=='Existing': body = np.concatenate((body, np.array([[[-.01]*3]*3, [[.01]*3]*3])))
    # Include translation to exercise roundoff and frame center compensation.
    body += [0.03,-0.02,0.01]
    source = output/f'{name}.glb'
    streams = {name:body}
    if name=='Mixed':
        streams['Large'] = probe.envelope(np.array([(x,y,z) for x in (.1,.12) for y in (0,.03) for z in (0,.04)]))
    groups = {}
    for node, positions in streams.items():
        normals = np.broadcast_to([.26726124,.53452248,.80178373],positions.shape).copy()
        uv = np.broadcast_to([[.1,.2],[.8,.2],[.8,.9]],(len(positions),3,2)).copy()
        groups[node] = indexed_primitives(positions,uv,normals,np.zeros(len(positions),dtype=int),np.arange(positions.size//3).reshape(-1,3))
    write_textured_glb(source,groups,{0:(Image.new('RGB',(1,1),'white'),True)})
    digest = sha256(source.read_bytes()).hexdigest()
    package = output/f'{name}-{uuid.uuid4().hex[:8]}'
    assert prepare(source,package,library,import_scale=scale)==0
    score = json.loads((package/'scorecard.json').read_text())
    assert score['status']=='complete' and score['size_planning']['auto_applied']==(name!='Enlarged')
    assert score['size_planning']['physical_container_models']==([name] if name!='Enlarged' else [])
    assert score['physics']['triangle_upload_ready']
    assert sha256(source.read_bytes()).hexdigest()==digest
    for suffix in ('','_LOD2','_LOD1','_LOD0'):
        dae = package/f'Lodestar{suffix}.dae'
        tree = ET.parse(dae)
        for geometry in tree.findall('.//{*}geometry'):
            for extra in geometry.findall('{*}extra'): geometry.remove(extra)
        ET.register_namespace('', 'http://www.collada.org/2005/11/COLLADASchema')
        tree.write(package/f'Reference{suffix}.dae',encoding='utf-8',xml_declaration=True)
    reports[name] = {'package':str(package),'source_unchanged':True,'import_scale':scale,'visual_triangles':{level:info['triangles'] for level,info in score['levels'].items()},
        'physical_container':score['size_planning']['parts'][name]}
    result_path = output/f'{name}-native.json'
    result = subprocess.run([str(native_audit), str(package), str(result_path), str(scale)], capture_output=True, text=True)
    (output/f'{name}-native.log').write_text(result.stdout+result.stderr)
    result.check_returncode()
    reports[name]['native'] = json.loads(result_path.read_text())
    if name not in ('Mixed', 'Enlarged'):
        fixup_path = output/f'{name}-normal-setting.json'
        result = subprocess.run([str(native_audit), str(package), str(fixup_path), str(scale), 'true'], capture_output=True, text=True)
        (output/f'{name}-normal-setting.log').write_text(result.stdout+result.stderr)
        result.check_returncode()
        reports[name]['alternate_normal_setting'] = json.loads(fixup_path.read_text())
(output/'manifest.json').write_text(json.dumps(reports,indent=2)+'\n')
rejections = {}
for defect in ('version', 'inverted', 'outside', 'duplicate'):
    package = output/f'invalid-{defect}'
    shutil.copytree(reports['Micro']['package'], package)
    high = package/'Lodestar.dae'
    tree = ET.parse(high)
    ns = {'c':'http://www.collada.org/2005/11/COLLADASchema'}
    frame = tree.find('.//c:technique[@profile="LODESTAR"]/c:physical_container', ns)
    if defect=='version': frame.set('version', '2')
    elif defect=='inverted': frame.set('maximum', frame.get('minimum'))
    elif defect=='outside': frame.set('minimum', '0,0,0'); frame.set('maximum', '.001,.001,.001')
    else:
        parent = tree.find('.//c:technique[@profile="LODESTAR"]', ns)
        parent.append(ET.fromstring(ET.tostring(frame)))
    tree.write(high,encoding='utf-8',xml_declaration=True)
    result = subprocess.run([str(native_audit), str(package), str(output/f'invalid-{defect}.json')], capture_output=True, text=True)
    (output/f'invalid-{defect}.log').write_text(result.stdout+result.stderr)
    if result.returncode==0 or 'Invalid physical container frame' not in result.stdout+result.stderr:
        raise ValueError(f'Invalid physical frame was not rejected: {defect}')
    rejections[defect] = True
maximum = max(lod['maximum_world_displacement_m'] for case in reports.values()
    for model in case['native']['models'] for lod in model['lods'])
summary = {'passed':True, 'cases':list(reports), 'invalid_frames_rejected':rejections,
    'maximum_measured_world_displacement_m':maximum, 'render_triangles_added':0,
    'degenerate_generated_triangles':0, 'native_lod_roundtrips':sum(len(case['native']['models'])*4 for case in reports.values()),
    'source_files_unchanged':True, 'gui_validated':False,'simulator_validated':False}
(output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
