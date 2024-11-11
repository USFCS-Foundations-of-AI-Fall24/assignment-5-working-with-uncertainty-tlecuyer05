from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("KeyPresent","Starts")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_keypresent = TabularCPD(
    variable="KeyPresent", variable_card=2,
    values=[[0.7], [0.3]],
    state_names={"KeyPresent": ["yes", "no"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],[0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "KeyPresent":["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)


# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keypresent)

car_infer = VariableElimination(car_model)

print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

#My queries for Problem 2 part 2
p1 = car_infer.query(variables=["Battery"],evidence={"Moves":"no"})
p2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})
p3 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Empty"})
p4 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"})
p5 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"})
p6 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", "Gas": "Full"})
p7 = car_infer.query(variables=["Starts"], evidence={"Radio":"turns on", "Gas": "Full"})
p8 = car_infer.query(variables=["KeyPresent"], evidence={"Moves":"no"})
if __name__ == "__main__":
    print("Probability battery is not working, given car won't move\n", p1)
    print("Probability that the car will not start given that radio is not working\n", p2)
    print("Probability of radio working without gas\n", p3)
    print("Probability of radio working with gas\n", p4)
    print("Probability of ignition failing if car has no gas\n", p5)
    print("Probability of ignition failing if car has gas\n", p6)
    print("Probability of car starting if radio works and has gas in it\n", p7)
    print("Probability that the key is not present given that the car does not move.\n", p8)