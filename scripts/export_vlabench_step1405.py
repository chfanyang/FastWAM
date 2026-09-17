"""Extract full BF16 DiT module from ZeRO state without changing the source."""
from pathlib import Path
import torch

run=Path('runs/vlabench_select_book_rothko_3cam192_wan21/2026-09-15_17-33-31')
out=run/'checkpoints/weights/step_001405.pt'
if out.exists(): raise FileExistsError(out)
source=run/'checkpoints/state/step_001405/pytorch_model/mp_rank_00_model_states.pt'
state=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
template=torch.load(run/'checkpoints/weights/step_000843.pt',map_location='cpu',weights_only=False,mmap=True)
assert state['global_steps']==1405, state['global_steps']
weights={k.removeprefix('video_expert.'):v for k,v in state['module'].items() if k.startswith('video_expert.')}
assert weights.keys()==template['dit'].keys()
assert all(weights[k].shape==v.shape for k,v in template['dit'].items())
template.update(dit=weights,step=1405,exported_from_state=str(source))
tmp=out.with_suffix('.tmp');torch.save(template,tmp);tmp.replace(out)
print('Exported',out,flush=True)
