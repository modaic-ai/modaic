from pathlib import Path

p0 = Path("./scratchpad.py")
p1 = Path("scratchpad2.py")
p3 = Path("package.json")
p4 = Path("/Users/tytodd/Desktop/Modaic/code/modaic/src/modaic/databases/vector_database/vector_database.py")
p5 = Path("src/modaic/databases/vector_database/vector_database.py")
p6 = Path("/Users/tytodd/Desktop/Modaic/code/modaic/scratchpad.py")
for p in [p0, p1, p3, p4, p5, p6]:
    print(repr(p))

print()
print()

for p in [p0, p1, p3, p4, p5, p6]:
    print(str(p))