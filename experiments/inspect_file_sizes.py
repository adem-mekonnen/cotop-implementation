import os

def get_dir_size(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

for d in ['data', 'figures', 'results', 'manuscript', 'docs', 'experiments']:
    sz = get_dir_size(d)
    print(f'{d}: {sz/1024:.1f} KB ({sz/(1024*1024):.2f} MB)')

print('\nDetailed files in data/:')
for root, dirs, files in os.walk('data'):
    for f in files:
        fp = os.path.join(root, f)
        print(f'  {fp} ({os.path.getsize(fp)/1024:.1f} KB)')

print('\nCheck root untracked files:')
for f in os.listdir('.'):
    if os.path.isfile(f) and (f.endswith('.pdf') or f.endswith('.txt')):
        print(f'  {f} ({os.path.getsize(f)/1024:.1f} KB)')
