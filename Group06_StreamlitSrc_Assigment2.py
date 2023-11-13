#Author: Elisabeth Oeljeklaus Group 6
# Date: 2023-11-13

import subprocess

# This script will run 'streamlit run Start.py' when executed.

def run_streamlit():
    # Define the command to start the Streamlit app
    command = "streamlit run Start.py"

    # Execute the command
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    run_streamlit()
