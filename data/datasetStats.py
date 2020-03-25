import json
import matplotlib.pyplot as plt
import os

with open('data/jersey_number_labelling_via_project.json') as f:
    viaProject = json.load(f)

# counts for digits 0 through 9, with subcounts for home vs away colors
digitCounts = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
for key in viaProject["_via_img_metadata"]:
    fileData = viaProject["_via_img_metadata"][key]
    homeAwayIdx = 0 if fileData["filename"].startswith("auburn") else 1
    for region in fileData["regions"]:
        digitIdx = int(region["region_attributes"]["digit"])
        digitCounts[digitIdx][homeAwayIdx] += 1

xValues = range(10)
homeCounts = list(map(lambda val: val[0], digitCounts))
awayCounts = list(map(lambda val: val[1], digitCounts))
barWidth = 0.35

fig, ax = plt.subplots()
ax.bar(xValues, homeCounts, barWidth, color='#9E1B32', label=f"Home (Total: {sum(homeCounts)})")
ax.bar(xValues, awayCounts, barWidth, bottom=homeCounts, color='#828A8F', label=f"Away (Total: {sum(awayCounts)})")

ax.set_title('Digit Counts Grouped by Jersey Color')
ax.set_ylabel('Count')
ax.set_xticks(xValues)
ax.set_yticks(range(0, 701, 50))
ax.legend()

plt.savefig(os.path.join('data', 'readme_assets', 'dataVisualization.png'))

plt.show()