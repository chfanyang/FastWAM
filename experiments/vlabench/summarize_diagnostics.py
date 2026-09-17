"""Make a frame-linked EE/IK comparison table from a recorded episode."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.directory / 'steps.jsonl').read_text().splitlines()]
    bad = [r['step'] for r in rows if not r['ik_success']]
    print('Recorded steps:', len(rows), 'IK failures:', bad)
    for key in ('requested_rotation_deg', 'ik_rotation_error_deg', 'rotation_error_deg',
                'requested_translation_m', 'ik_position_error_m', 'position_error_m'):
        if rows:
            row = max(rows, key=lambda r: r[key])
            print(key, 'max=',row[key], 'step=',row['step'])
    lines = ['# VLABench 单场景 EE / IK 诊断', '',
             f'已记录 {len(rows)} 步；IK 未收敛步骤（从0开始）：{bad}。', '',
             '请求变化：本步目标相对执行前实际 EE；IK误差：IK解做FK后相对请求目标；'
             '执行误差：执行一次环境step后实际EE相对请求目标。旋转采用SO(3)测地角，不是Euler分量差。', '',
             '图像是本步动作执行前的两路观测；完整xyz/wxyz/关节值见steps.jsonl，预测chunk见replan_*.json。', '',
             '|step / 图像|replan/index|IK成功|请求位移mm|请求转角°|IK位置误差mm|IK旋转误差°|执行位置误差mm|执行旋转误差°|',
             '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"|[{r['step']}]({r['frame']})|{r['replan']}/{r['action_index']}|{r['ik_success']}|"
                     f"{1000*r['requested_translation_m']:.2f}|{r['requested_rotation_deg']:.2f}|"
                     f"{1000*r['ik_position_error_m']:.2f}|{r['ik_rotation_error_deg']:.2f}|"
                     f"{1000*r['position_error_m']:.2f}|{r['rotation_error_deg']:.2f}|")
    (args.directory / 'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
