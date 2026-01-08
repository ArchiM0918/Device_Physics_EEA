from devsim import *
from devsim.python_packages.simple_physics import *
import numpy as np
import matplotlib.pyplot as plt


# ===================================================================
# MESH GENERATION
# ===================================================================
device = "MyDevice"
region = "silicon_region"

# Define Dimensions (50 microns long)
xmin = 0.0
xmax = 50e-4      # 50 microns in cm
x_points = 101    # Number of nodes

create_1d_mesh(mesh="mesh1d")
xs = np.linspace(xmin, xmax, x_points)
step_size = (xmax - xmin) / (x_points - 1)

for i, pos in enumerate(xs):
    if i == 0:
        add_1d_mesh_line(mesh="mesh1d", pos=pos, ps=step_size, ns=step_size, tag="top")
    elif i == len(xs) - 1:
        add_1d_mesh_line(mesh="mesh1d", pos=pos, ps=step_size, ns=step_size, tag="bot")
    else:
        add_1d_mesh_line(mesh="mesh1d", pos=pos, ps=step_size, ns=step_size)

add_1d_contact(mesh="mesh1d", name="top", tag="top", material="metal")
add_1d_contact(mesh="mesh1d", name="bot", tag="bot", material="metal")
add_1d_region(mesh="mesh1d", material="Silicon", region=region, tag1="top", tag2="bot")

finalize_mesh(mesh="mesh1d")
create_device(mesh="mesh1d", device=device)

# ==================================================================
# MATERIAL PARAMETERS at 300 K
# ==================================================================

SetSiliconParameters(device, region, 300)

# Set Carrier Lifetimes 
set_parameter(device=device, region=region, name="taun", value=1e-6)
set_parameter(device=device, region=region, name="taup", value=1e-6)

# Create the solution variables (Potential, Electrons, Holes)
CreateSolution(device, region, "Potential")
CreateSolution(device, region, "Electrons")
CreateSolution(device, region, "Holes")

# ==================================================================
# DOPING PROFILE OF RESISTOR (STEP FUNCTION)
# ==================================================================
# Left Side (0-25um): High Doping (1e17)
# Right Side (25-50um): Low Doping (1e16)
N_high = 1.0e17
N_low  = 1.0e16
mid_point = 25e-4 

# Create the Equation String
# step(mid_point - x) is 1 when x < 50um (Left Side -> P-Type)
# step(x - mid_point) is 1 when x > 50um (Right Side -> N-Type)
doping_eq = f"{N_high} * step({mid_point} - x) + {N_low} * step(x - {mid_point})"

# Apply to the Device
CreateNodeModel(device, region,"Donors", doping_eq)
CreateNodeModel(device, region, "Acceptors", "0.0")
CreateNodeModel(device, region, "NetDoping", "Donors - Acceptors")

# ==================================================================
# INITIAL SOLUTION (EQUILIBRIUM / ZERO BIAS)
# ==================================================================

# Solves Poisson's Equation only to find the starting potential

CreateSiliconPotentialOnly(device, region)

# Set contacts biases to 0.0V
for i in get_contact_list(device=device):
    set_parameter(device=device, name=GetContactBiasName(i), value=0.0)
    CreateSiliconPotentialOnlyContact(device, region, i)

print("Solving Initial Equilibrium (Poisson Only)...")
solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)

# Initialize n and p from the Potential solution
set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
set_node_values(device=device, region=region, name="Holes",     init_from="IntrinsicHoles")

# SAVE DATA FOR ZERO BIAS PLOTS
x_val_0V     = np.array(get_node_model_values(device=device, region=region, name="x"))
potential_0V = np.array(get_node_model_values(device=device, region=region, name="Potential"))
electrons_0V = np.array(get_node_model_values(device=device, region=region, name="Electrons"))
holes_0V     = np.array(get_node_model_values(device=device, region=region, name="Holes"))
donors_0V    = np.array(get_node_model_values(device=device, region=region, name="Donors"))

# Calculate Excess (Spillover)
excess_n = electrons_0V - donors_0V


# ==================================================================
# DRIFT-DIFFUSION SIMULATION (0.3V BIAS)
# ==================================================================

# Set up Full Drift-Diffusion Equations
CreateSiliconDriftDiffusion(device, region)

for i in get_contact_list(device=device):
    CreateSiliconDriftDiffusionAtContact(device, region, i)

# Apply Bias (0.3V at top)
set_parameter(device=device, name=GetContactBiasName("top"), value=0.3)
set_parameter(device=device, name=GetContactBiasName("bot"), value=0.0)

solve(type="dc", absolute_error=1e10, relative_error=1e-5, maximum_iterations=50)

# SAVE DATA FOR BIASED PLOT
x_val_bias     = np.array(get_node_model_values(device=device, region=region, name="x"))
potential_bias = np.array(get_node_model_values(device=device, region=region, name="Potential"))


# ==================================================================
# PLOTTING RESULTS
# ==================================================================

# 1: BUILT-IN POTENTIAL (0V)
plt.figure(figsize=(8, 6))
plt.plot(x_val_0V * 1e4, potential_0V, linewidth=2, color='blue')
plt.title("Built-in Potential (Equilibrium / 0V)")
plt.xlabel("Position (microns)")
plt.ylabel("Potential (V)")
plt.grid(True)
plt.savefig("resistor_potential_0v.png") # Saves the image
plt.show()

# 2: ELECTRON SPILLOVER (EXCESS CARRIERS)
plt.figure(figsize=(8, 6))
plt.plot(x_val_0V * 1e4, excess_n, linewidth=2, color='green', label='n - Nd')
plt.title("Electron Spillover (Excess Carriers)")
plt.xlabel("Position (microns)")
plt.ylabel("Excess Concentration (cm^-3)")
plt.axhline(0, color='black', linewidth=1, linestyle='--') 
plt.legend()
plt.grid(True)
plt.savefig("resistor_spillover.png")
plt.show()

# 3: POTENTIAL WITH 0.3V BIAS 
plt.figure(figsize=(8, 6))
plt.plot(x_val_bias * 1e4, potential_bias, linewidth=3, color='red')
plt.title("Potential Profile with 0.3V Bias Applied")
plt.xlabel("Position (microns)")
plt.ylabel("Potential (V)")
plt.grid(True)
plt.savefig("resistor_potential_bias.png")
plt.show()

