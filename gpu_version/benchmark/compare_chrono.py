"""Compare native floating CSV with extracted DualSPHysics body CSVs.

python compare_chrono.py native_Floating.csv dual_directory [2|3]
Reported differences compare solvers; they are not experimental errors.
"""
import bisect
import csv
import json
import math
import sys
from pathlib import Path

dimension = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rows = list(csv.DictReader(open(sys.argv[1],newline='')))
reference = {}
for group in {int(row['GroupMarker']) for row in rows}:
    source = list(csv.DictReader(open(Path(sys.argv[2])/f'floating_mk{group}.csv',newline='')))
    reference[group] = ([float(r['time [s]']) for r in source],source)
position,pitch = [],[]
for row in rows:
    time = float(row['Time'])
    times,source = reference[int(row['GroupMarker'])]
    index = max(0,min(len(times)-2,bisect.bisect_right(times,time)-1))
    fraction = max(0,min(1,(time-times[index])/(times[index+1]-times[index])))
    def sample(column):
        return (1-fraction)*float(source[index][column])+fraction*float(source[index+1][column])
    axes = ['x','z'] if dimension == 2 else ['x','y','z']
    position.append(math.sqrt(sum((float(row[f'Center:{d}'])-sample(f'center.{axis} [m]'))**2 for d,axis in enumerate(axes))))
    if dimension == 2:
        pitch.append(-math.degrees(float(row['Angle']))-sample('pitch [deg]'))
metrics = {'body_center_rms_difference_m':math.sqrt(sum(x*x for x in position)/len(position)),
           'body_center_max_difference_m':max(position),'body_samples':len(position)}
if pitch:
    metrics.update(pitch_rms_difference_deg=math.sqrt(sum(x*x for x in pitch)/len(pitch)),pitch_max_difference_deg=max(map(abs,pitch)))
print(json.dumps(metrics,indent=2))
