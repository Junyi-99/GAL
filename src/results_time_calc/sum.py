import csv

# Dictionary to store the total time and count for each group
group_times = {}

# Read the CSV file
with open('output.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    
    # Iterate over each row in the CSV file
    for row in reader:
        method = row[0]
        split = row[2]
        weight = row[3]
        time = row[4]
        
        # Skip rows with missing time value
        if not time:
            continue
        
        # Create a unique key for the group
        key = (method, split, weight)
        
        # Convert the time to seconds
        hours, minutes, seconds = map(int, time.split(':'))
        total_seconds = (hours * 3600) + (minutes * 60) + seconds
        
        # Update the total time and count for the group
        if key in group_times:
            group_times[key][0] += total_seconds
            group_times[key][1] += 1
        else:
            group_times[key] = [total_seconds, 1]

# Calculate the average time for each group
average_times = {}
for key, (total_time, count) in group_times.items():
    average_time = total_time / count
    average_times[key] = average_time

# Print the average time for each group
for key, average_time in average_times.items():
    hours = average_time / 3600
    print(f'Group: {key}, Average Time: {hours}')