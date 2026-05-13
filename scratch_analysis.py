"""Quick analysis of box volumes and theoretical packing limits."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.data_loader import load_data

boxes, container = load_data("data/project10_container_dataset.json")
cvol = container.L * container.W * container.H

vols = sorted([b.l * b.w * b.h for b in boxes], reverse=True)
total_vol = sum(vols)
print(f"Container: {container.L}x{container.W}x{container.H} = {cvol:,} cm³")
print(f"Total box volume: {total_vol:,} cm³")
print(f"Theoretical max utilization: {total_vol/cvol*100:.2f}%")
print(f"Avg box volume: {total_vol/len(boxes):,.0f} cm³")
print(f"Median box vol: {vols[len(vols)//2]:,} cm³")

# Check how many boxes fit if we greedily sum volumes
cumvol = 0
for i, v in enumerate(vols):
    if cumvol + v > cvol:
        print(f"\nVolume-limit: first {i} largest boxes fill {cumvol/cvol*100:.1f}%")
        break
    cumvol += v
else:
    print(f"\nAll {len(vols)} boxes fit by volume: {cumvol/cvol*100:.1f}%")

# Smallest 24 boxes
small24 = sorted(vols)[:24]
print(f"\n24 smallest box volumes sum: {sum(small24):,} cm³ = {sum(small24)/cvol*100:.2f}%")
print(f"If we placed all 386: {total_vol/cvol*100:.2f}%")
print(f"362 boxes placed → unplaced vol unknown (depends on which 24)")
