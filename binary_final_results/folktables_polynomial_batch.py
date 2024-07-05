import subprocess

if __name__ == "__main__":
    cmd1 = "python folktables_polynomial_sol.py income"
    cmd2 = "python folktables_polynomial_sol.py employment"
    subprocess.run(cmd1.split(" "))
    subprocess.run(cmd2.split(" "))
