with open('models_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'dist_xx = xx + xx.T - 2 * torch.matmul(x, x.T)' in line:
        line = line.replace('dist_xx = xx + xx.T - 2 * torch.matmul(x, x.T)', 
                           'dist_xx = torch.clamp(xx + xx.T - 2 * torch.matmul(x, x.T), min=0.0)')
    if 'dist_yy = yy + yy.T - 2 * torch.matmul(y, y.T)' in line:
        line = line.replace('dist_yy = yy + yy.T - 2 * torch.matmul(y, y.T)', 
                           'dist_yy = torch.clamp(yy + yy.T - 2 * torch.matmul(y, y.T), min=0.0)')
    if 'return hsic' in line and 'clamp' not in line:
        line = line.replace('return hsic', 'return torch.clamp(hsic, min=0.0, max=100.0)')
    new_lines.append(line)

with open('models_v2.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('修复完成！')