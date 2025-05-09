from pulp import *

# ulazni podaci
skladista = ["TV1", "TV2", "TV3", "TV4"]
trgovine = ["TR1", "TR2", "TR3", "TR4"]

ponuda = {
    "TV1": 100,
    "TV2": 50,
    "TV3": 60,
    "TV4": 70
}

potraznja = {
    "TR1": 30,
    "TR2": 120,
    "TR3": 80,
    "TR4": 50
}

cijene = {
    "TV1": {"TR1": 5, "TR2": 5, "TR3": 6, "TR4": 1},
    "TV2": {"TR1": 1, "TR2": 8, "TR3": 2, "TR4": 5},
    "TV3": {"TR1": 9, "TR2": 5, "TR3": 5, "TR4": 7},
    "TV4": {"TR1": 5, "TR2": 3, "TR3": 2, "TR4": 3}
}

# definicija problema - minimiziramo cijenu
prob = LpProblem("Transportation_Problem", LpMinimize)

# varijable odluke: kolicina robe koju šaljemo iz skladišta u trgovine
route_vars = LpVariable.dicts("Route", (skladista, trgovine), 0, None, LpInteger)

# funkcija cilja: minimizacija ukupne cijene transporta
prob += lpSum([route_vars[w][b] * cijene[w][b] for w in skladista for b in trgovine]), "Total_Transportation_Cost"

# ograničenja: ponuda iz svakog skladišta
for w in skladista:
    prob += lpSum([route_vars[w][b] for b in trgovine]) <= ponuda[w], f"Supply_{w}"

# ograničenja: potražnja svake trgovine
for b in trgovine:
    prob += lpSum([route_vars[w][b] for w in skladista]) >= potraznja[b], f"Demand_{b}"

# riješi problem
prob.solve(PULP_CBC_CMD())

# ispis rezultata
print(f"Status problema: {LpStatus[prob.status]}")

for w in skladista:
    for b in trgovine:
        if route_vars[w][b].varValue > 0:
            print(f"Pošalji {route_vars[w][b].varValue} jedinica robe iz {w} u {b}")

print(f"Ukupna cijena = {value(prob.objective)}")
