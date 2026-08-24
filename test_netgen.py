import subprocess
net_cmd = [
    'netgenerate',
    '--grid',
    '--grid.x-number', '6',
    '--grid.y-number', '1',
    '--grid.length', '400',
    '--default.lanenumber', '3',
    '--default.speed', '13.89',
    '--output-file', 'hangzhou.net.xml',
    '--no-turnarounds', 'true',
    '--tls.guess', 'true',
]
res = subprocess.run(net_cmd, capture_output=True, text=True)
print('RC:', res.returncode)
print('STDOUT:', res.stdout)
print('STDERR:', res.stderr)
