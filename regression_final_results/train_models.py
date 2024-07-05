import subprocess

if __name__ == "__main__":
    states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

    for state in states:
        # train ce model
        ce_command = f"python folktables_notepo.py income {state}_CE_income {state}"
        subprocess.run(ce_command.split(" "))

    for state in states:
        # train spo model
        spo_command = f"python folktables_epo.py income {state}_SPO_income {state}"
        subprocess.run(spo_command.split(" "))
