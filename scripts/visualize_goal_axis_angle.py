"""Render one real Goal action, before VAE, without touching existing figures."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,default=114)
    p.add_argument('--start',type=int,default=30)
    p.add_argument('--offset',type=int,default=1,help='Raymap index: 0=current state, 1=action[start].')
    a=p.parse_args()
    if not 0<=a.offset<=16:raise ValueError('offset must be 0..16')
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    import pyarrow.parquet as pq
    from PIL import Image,ImageDraw,ImageFont
    from omegaconf import OmegaConf
    from fastwam.representations.libero_rothko import LiberoRothkoCodec,LiberoRothkoCodecConfig
    from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec
    torch.set_num_threads(2)
    reference=ROOT/'runs/libero_goal_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-21_18-42-23/config.yaml'
    cfg=OmegaConf.load(reference)
    base=LiberoRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)),ROOT/cfg.data.train.rothko_norm_stats)
    codec=LiberoAxisAngleCodec(base)
    dataset_root=ROOT/cfg.data.train.dataset_dirs[0]
    source=next(dataset_root.glob(f'data/chunk-*/episode_{a.episode:06d}.parquet'))
    episode_meta=next(json.loads(line) for line in (dataset_root/'meta/episodes.jsonl').read_text().splitlines()
                      if json.loads(line)['episode_index']==a.episode)
    columns=['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz','observation.state.gripper_open','action']
    table=pq.read_table(source,columns=columns)
    data={k:torch.tensor(table[k].to_pylist(),dtype=torch.float32) for k in columns}
    if not 0<=a.start<len(table):raise ValueError('Invalid frame start')
    ix=torch.arange(a.start,a.start+16).clamp_max(len(table)-1)
    pose=torch.cat((data[columns[0]][a.start:a.start+1],data[columns[1]][ix]))
    grip=torch.cat((data[columns[2]][a.start:a.start+1].reshape(1,1),data['action'][ix,-1:])).clamp(0,1)
    original=base.encode(pose,grip)
    encoded=codec.encode(pose,grip)
    center,_,_,border=codec.masks(encoded.device)
    assert torch.equal(original[...,:224][...,center|border],encoded[...,:224][...,center|border])
    assert torch.equal(encoded[...,:224],encoded[...,224:])
    decoded,opening=codec.decode(encoded,pose[0])
    q=torch.nn.functional.normalize(decoded[:,3:].double(),dim=-1)
    gt=torch.nn.functional.normalize(pose[:,3:].double(),dim=-1)
    rotation_error=torch.rad2deg(2*torch.acos((q*gt).sum(-1).abs().clamp(0,1)))
    rotation_vector=codec.rotation_vectors(pose[None])[0,a.offset]
    rotation_degrees=float(torch.rad2deg(rotation_vector.norm()))
    scale=3;size=224*scale;margin=36;gap=72;top=84;bottom=104
    canvas=Image.new('RGB',(2*margin+2*size+gap,top+size+bottom),'white')
    draw=ImageDraw.Draw(canvas)
    font_path='/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf'
    title=ImageFont.truetype(font_path,40)
    small=ImageFont.truetype(font_path,26)
    for i,(label,pixels) in enumerate([('Action Raymap',original),('Constant Axis-Angle',encoded)]):
        image=((pixels[:,a.offset,:,:224].clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).numpy()
        left=margin+i*(size+gap)
        draw.text((left+size/2,top/2),label,font=title,fill='black',anchor='mm')
        canvas.paste(Image.fromarray(image).resize((size,size),Image.Resampling.NEAREST),(left,top))
    code=(rotation_vector/torch.pi).tolist()
    grip_label='open' if float(grip[a.offset,0])>=.5 else 'closed'
    footer=f'Same target; {grip_label} gripper. One tile shown; actual input has two identical tiles.'
    draw.text((canvas.width/2,top+size+30),footer,font=small,fill='black',anchor='mm')
    draw.text((canvas.width/2,top+size+68),
        f'Rotation: {rotation_degrees:.2f} deg; axis-angle / pi = ['+', '.join(f'{v:.5f}' for v in code)+']',
        font=small,fill='black',anchor='mm')
    a.output.mkdir(parents=True)
    path=a.output/'comparison.png'
    canvas.save(path,dpi=(300,300))
    metadata=dict(source=str(source.relative_to(ROOT)),episode=a.episode,frame_start=a.start,
        raymap_offset=a.offset,target_action_frame=None if a.offset==0 else int(ix[a.offset-1]),
        task=episode_meta['tasks'][0],rotation_magnitude_degrees=rotation_degrees,
        gripper_open=float(grip[a.offset,0]),rotation_vector_radians=rotation_vector.tolist(),
        rotation_code=code,translation_and_gripper_bit_identical=True,tiles_bit_identical=True,
        max_direct_rotation_error_degrees=float(rotation_error.max()),
        max_direct_position_error_mm=float((decoded[:,:3]-pose[:,:3]).norm(dim=-1).max()*1000),
        stats_fingerprint=base.norm_stats.fingerprint(),codec_metadata=codec.metadata(),
        display='Actual normalized input [-1,1] mapped to uint8 RGB. No per-image color rescaling. Nearest-neighbor display only.',
        stage='Codec and visualization only; not connected to existing training/eval defaults or caches; no VAE run.')
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(path,flush=True)


if __name__=='__main__':main()
