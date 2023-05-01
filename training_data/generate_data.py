import csv
import glob
import shutil

# Get a list of all CSV files in the current directory
csv_files = glob.glob('*.csv')

# Define the original file name and the new file name
combined = 'combined.csv'

# Make a copy of the original file with the new name
shutil.copyfile(csv_files[0], combined)

# Write the combined data to a new CSV file
with open('combined.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Loop through each CSV file and read the data into a list
    for file in csv_files[1:]:
        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the first row (header row)
            #next(reader)
            file_data = [row for row in reader]
            writer.writerows(file_data)
